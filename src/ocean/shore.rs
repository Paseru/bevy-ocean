//! Vagues côtières : précalcul, au démarrage, du trajet de la houle autour de l'île.
//!
//! La houle arrive du large en ondes planes. Elle ralentit quand le fond remonte
//! (c = ω/k avec ω² = g·k·tanh(k·h)), ce qui la fait tourner vers le rivage (réfraction) et
//! resserre les crêtes. On résout l'équation eikonale |∇T| = 1/c par "fast marching" :
//! T(x) est l'heure d'arrivée de la crête en x, les crêtes sont les lignes de même T.
//!
//! On propage ensuite la hauteur des vagues le long des rayons, dans l'ordre d'arrivée :
//! - conservation du flux d'énergie (H²·cg constant) → les vagues grossissent en eau peu
//!   profonde (shoaling) ;
//! - étalement ou concentration des rayons (divergence de la direction de propagation) ;
//! - déferlement : H ne peut pas dépasser γ·h (γ ≈ 0,78) ; ce qui est perdu fait de l'écume,
//!   qui dérive ensuite vers la plage en s'estompant ;
//! - frottement sur le platier.
//!
//! Résultat : deux textures (lues par les shaders de l'eau et du fond) et la même donnée
//! côté CPU (hauteur de l'eau sous la caméra).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::TAU;

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::terrain;

/// Nombre de cellules par côté de la grille.
pub const SHORE_N: usize = 1024;
/// Demi-côté du domaine (m), centré sur l'île.
pub const SHORE_HALF: f32 = 640.0;
/// Périodes des deux trains de houle (s) : leur battement donne des séries de vagues.
pub const SHORE_PERIODS: [f32; 2] = [8.5, 9.7];
/// Part de la hauteur portée par chaque train.
pub const SHORE_SPLIT: [f32; 2] = [0.78, 0.55];
/// Indice de déferlement (hauteur max / profondeur).
pub const BREAK_GAMMA: f32 = 0.78;
const G: f32 = 9.81;
/// Profondeur en dessous de laquelle une cellule est considérée comme émergée.
const DRY_DEPTH: f32 = 0.04;

/// Nombre d'onde (approximation d'Eckart, ~5 %) pour la pulsation ω à la profondeur d.
fn wavenumber(omega: f32, d: f32) -> f32 {
    let k0 = omega * omega / G;
    let d = d.max(0.01);
    k0 / (k0 * d).tanh().sqrt()
}

/// Vitesse de phase et vitesse de groupe.
fn speeds(omega: f32, d: f32) -> (f32, f32) {
    let k = wavenumber(omega, d);
    let c = omega / k;
    let kd = (k * d.max(0.01)).min(20.0);
    let n = 0.5 * (1.0 + 2.0 * kd / (2.0 * kd).sinh().max(1e-6));
    (c, c * n)
}

#[derive(Copy, Clone, PartialEq)]
struct Item {
    t: f32,
    idx: u32,
}
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        // tas min
        other.t.partial_cmp(&self.t).unwrap_or(Ordering::Equal).then(other.idx.cmp(&self.idx))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Données précalculées des vagues côtières.
#[derive(Resource, Clone)]
pub struct ShoreField {
    pub n: usize,
    pub half: f32,
    pub cell: f32,
    /// (T : heure d'arrivée des crêtes (s), H : hauteur des vagues (m), F : écume, hauteur du fond)
    pub tex0: Vec<[f32; 4]>,
    /// (∂T/∂x, ∂T/∂z, Hb : hauteur des derniers brisants (jet de rive), distance à terre (m))
    pub tex1: Vec<[f32; 4]>,
    pub image0: Handle<Image>,
    pub image1: Handle<Image>,
}

impl ShoreField {
    pub fn bake(swell_dir: Vec2, h0: f32, images: &mut Assets<Image>) -> Self {
        let t_start = std::time::Instant::now();
        let n = SHORE_N;
        let half = SHORE_HALF;
        let cell = 2.0 * half / n as f32;
        let pos = |i: usize| -half + (i as f32 + 0.5) * cell;

        // --- relief (calcul parallèle)
        let mut height = vec![0.0f32; n * n];
        let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
        let rows_per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, chunk) in height.chunks_mut(rows_per * n).enumerate() {
                s.spawn(move || {
                    for (k, h) in chunk.iter_mut().enumerate() {
                        let idx = ci * rows_per * n + k;
                        let (j, i) = (idx / n, idx % n);
                        *h = terrain::height_at(pos(i), pos(j));
                    }
                });
            }
        });
        let depth: Vec<f32> = height.iter().map(|h| (-h).max(0.0)).collect();
        let wet: Vec<bool> = depth.iter().map(|&d| d > DRY_DEPTH).collect();

        // vitesses pour la période de référence
        let omega = TAU / (0.5 * (SHORE_PERIODS[0] + SHORE_PERIODS[1]));
        let mut slowness = vec![0.0f32; n * n];
        let mut cg = vec![0.0f32; n * n];
        for idx in 0..n * n {
            if wet[idx] {
                let (c, g) = speeds(omega, depth[idx]);
                slowness[idx] = 1.0 / c;
                cg[idx] = g;
            }
        }

        // --- fast marching depuis le bord amont (ondes planes)
        let dir = swell_dir.normalize();
        let mut t = vec![f32::INFINITY; n * n];
        let mut state = vec![0u8; n * n]; // 0 loin, 1 essai, 2 connu
        let mut heap = BinaryHeap::new();
        let mut d_min = f32::INFINITY;
        for j in 0..n {
            for i in 0..n {
                d_min = d_min.min(Vec2::new(pos(i), pos(j)).dot(dir));
            }
        }
        let c_deep = 1.0 / slowness[0].max(1e-6);
        for j in 0..n {
            for i in 0..n {
                let border = i == 0 || j == 0 || i == n - 1 || j == n - 1;
                let idx = j * n + i;
                let p = Vec2::new(pos(i), pos(j));
                if border && wet[idx] && p.dot(dir) < 0.0 {
                    t[idx] = (p.dot(dir) - d_min) / c_deep;
                    state[idx] = 1;
                    heap.push(Item { t: t[idx], idx: idx as u32 });
                }
            }
        }
        let mut order: Vec<u32> = Vec::with_capacity(n * n);
        let neighbors = |idx: usize| -> [Option<usize>; 4] {
            let (j, i) = (idx / n, idx % n);
            [
                (i > 0).then(|| idx - 1),
                (i + 1 < n).then(|| idx + 1),
                (j > 0).then(|| idx - n),
                (j + 1 < n).then(|| idx + n),
            ]
        };
        let solve = |t: &Vec<f32>, state: &Vec<u8>, idx: usize, f: f32| -> f32 {
            let nb = neighbors(idx);
            let get = |o: Option<usize>| o.filter(|&k| state[k] == 2).map(|k| t[k]).unwrap_or(f32::INFINITY);
            let a = get(nb[0]).min(get(nb[1]));
            let b = get(nb[2]).min(get(nb[3]));
            let (a, b) = if a < b { (a, b) } else { (b, a) };
            if !a.is_finite() {
                return f32::INFINITY;
            }
            if !b.is_finite() || b - a >= f {
                return a + f;
            }
            0.5 * (a + b + (2.0 * f * f - (a - b) * (a - b)).max(0.0).sqrt())
        };
        while let Some(Item { t: ti, idx }) = heap.pop() {
            let idx = idx as usize;
            if state[idx] == 2 || ti > t[idx] {
                continue;
            }
            state[idx] = 2;
            order.push(idx as u32);
            for nb in neighbors(idx).into_iter().flatten() {
                if state[nb] == 2 || !wet[nb] {
                    continue;
                }
                let tn = solve(&t, &state, nb, slowness[nb] * cell);
                if tn < t[nb] {
                    t[nb] = tn;
                    state[nb] = 1;
                    heap.push(Item { t: tn, idx: nb as u32 });
                }
            }
        }

        // --- lissage léger de T (le fast marching du premier ordre crénelle un peu les crêtes)
        for _ in 0..2 {
            let src = t.clone();
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let idx = j * n + i;
                    if !wet[idx] || !src[idx].is_finite() {
                        continue;
                    }
                    let mut acc = 0.0;
                    let mut w = 0.0;
                    for (dj, di, wk) in [(0i32, 0i32, 4.0f32), (0, -1, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (1, 0, 1.0)] {
                        let k = ((j as i32 + dj) as usize) * n + (i as i32 + di) as usize;
                        if wet[k] && src[k].is_finite() {
                            acc += src[k] * wk;
                            w += wk;
                        }
                    }
                    t[idx] = acc / w;
                }
            }
        }

        // --- gradient de T (direction et lenteur locales)
        let mut grad = vec![Vec2::ZERO; n * n];
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                if !wet[idx] || !t[idx].is_finite() {
                    continue;
                }
                let sample = |ii: i32, jj: i32| -> Option<f32> {
                    if ii < 0 || jj < 0 || ii >= n as i32 || jj >= n as i32 {
                        return None;
                    }
                    let k = jj as usize * n + ii as usize;
                    (wet[k] && t[k].is_finite()).then(|| t[k])
                };
                let (ii, jj) = (i as i32, j as i32);
                let gx = match (sample(ii - 1, jj), sample(ii + 1, jj)) {
                    (Some(a), Some(b)) => (b - a) / (2.0 * cell),
                    (Some(a), None) => (t[idx] - a) / cell,
                    (None, Some(b)) => (b - t[idx]) / cell,
                    _ => 0.0,
                };
                let gz = match (sample(ii, jj - 1), sample(ii, jj + 1)) {
                    (Some(a), Some(b)) => (b - a) / (2.0 * cell),
                    (Some(a), None) => (t[idx] - a) / cell,
                    (None, Some(b)) => (b - t[idx]) / cell,
                    _ => 0.0,
                };
                grad[idx] = Vec2::new(gx, gz);
            }
        }
        // divergence de la direction de propagation (étalement des rayons), calculée sur un
        // champ de directions lissé (sinon le bruit du fast marching fait des stries)
        let mut dirs: Vec<Vec2> = grad.iter().map(|g| g.normalize_or_zero()).collect();
        for _ in 0..3 {
            let src = dirs.clone();
            for j in 2..n - 2 {
                for i in 2..n - 2 {
                    let idx = j * n + i;
                    if !wet[idx] {
                        continue;
                    }
                    let mut acc = Vec2::ZERO;
                    for dj in -2i32..=2 {
                        for di in -2i32..=2 {
                            let k = ((j as i32 + dj) as usize) * n + (i as i32 + di) as usize;
                            if wet[k] {
                                acc += src[k];
                            }
                        }
                    }
                    dirs[idx] = acc.normalize_or_zero();
                }
            }
        }
        let mut div = vec![0.0f32; n * n];
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let idx = j * n + i;
                if !wet[idx] {
                    continue;
                }
                let dx = (dirs[idx + 1].x - dirs[idx - 1].x) / (2.0 * cell);
                let dz = (dirs[idx + n].y - dirs[idx - n].y) / (2.0 * cell);
                div[idx] = dx + dz;
            }
        }

        // --- hauteur des vagues, écume, brisants, dans l'ordre d'arrivée
        let mut hh = vec![0.0f32; n * n];
        let mut foam = vec![0.0f32; n * n];
        let mut hb = vec![0.0f32; n * n];
        let mut done = vec![false; n * n];
        for &idx in &order {
            let idx = idx as usize;
            let mut wsum = 0.0;
            let mut h_up = 0.0;
            let mut cg_up = 0.0;
            let mut f_up = 0.0;
            let mut hb_up = 0.0;
            for nb in neighbors(idx).into_iter().flatten() {
                if done[nb] && t[nb] < t[idx] {
                    let w = t[idx] - t[nb];
                    wsum += w;
                    h_up += w * hh[nb];
                    cg_up += w * cg[nb];
                    f_up += w * foam[nb];
                    hb_up += w * hb[nb];
                }
            }
            let d = depth[idx];
            let (h_in, f_in, hb_in) = if wsum > 0.0 {
                let h_up = h_up / wsum;
                let cg_up = cg_up / wsum;
                let ds = cell;
                // shoaling + étalement des rayons + frottement sur le platier
                let shoal = (cg_up / cg[idx].max(1e-3)).sqrt();
                let spread = (-0.5 * div[idx] * ds).exp().clamp(0.96, 1.04);
                let friction = if d < 4.0 { (-ds / 450.0).exp() } else { 1.0 };
                (h_up * shoal * spread * friction, f_up / wsum * (-ds / 28.0).exp(), hb_up / wsum * (-ds / 30.0).exp())
            } else {
                (h0, 0.0, 0.0)
            };
            let h_max = BREAK_GAMMA * d;
            let h_out = h_in.min(h_max);
            let loss = if h_in > 1e-4 { (h_in - h_out) / h_in } else { 0.0 };
            hh[idx] = h_out;
            foam[idx] = f_in.max((loss * 9.0).min(1.0) * (h_in / 0.25).min(1.0));
            hb[idx] = hb_in.max(if loss > 0.0 { h_in.min(1.2) } else { 0.0 });
            done[idx] = true;
        }

        // --- prolongement sur la plage (jet de rive) : on recopie l'heure d'arrivée au rivage,
        // la direction et la hauteur des brisants de la cellule mouillée la plus proche
        let mut land_dist = vec![f32::INFINITY; n * n];
        let mut heap = BinaryHeap::new();
        for idx in 0..n * n {
            if wet[idx] && done[idx] {
                if neighbors(idx).into_iter().flatten().any(|k| !wet[k]) {
                    land_dist[idx] = 0.0;
                    heap.push(Item { t: 0.0, idx: idx as u32 });
                }
            }
        }
        while let Some(Item { t: di, idx }) = heap.pop() {
            let idx = idx as usize;
            if di > land_dist[idx] || di > 40.0 {
                continue;
            }
            for nb in neighbors(idx).into_iter().flatten() {
                if wet[nb] {
                    continue;
                }
                let nd = di + cell;
                if nd < land_dist[nb] {
                    land_dist[nb] = nd;
                    t[nb] = t[idx];
                    grad[nb] = grad[idx];
                    hb[nb] = hb[idx];
                    heap.push(Item { t: nd, idx: nb as u32 });
                }
            }
        }

        let mut tex0 = vec![[0.0f32; 4]; n * n];
        let mut tex1 = vec![[0.0f32; 4]; n * n];
        for idx in 0..n * n {
            let ti = if t[idx].is_finite() { t[idx] } else { -1.0 };
            let ld = if wet[idx] { 0.0 } else if land_dist[idx].is_finite() { land_dist[idx] } else { 99.0 };
            tex0[idx] = [ti, hh[idx], foam[idx], height[idx]];
            tex1[idx] = [grad[idx].x, grad[idx].y, hb[idx], ld];
        }
        let image0 = images.add(make_image(n, &tex0));
        let image1 = images.add(make_image(n, &tex1));
        info!(
            "Vagues côtières : propagation de la houle précalculée en {:.2} s",
            t_start.elapsed().as_secs_f32()
        );
        let field = Self { n, half, cell, tex0, tex1, image0, image1 };
        if let Ok(dir) = std::env::var("OCEAN_DUMP") {
            field.dump(&dir);
        }
        field
    }

    /// Écrit les deux textures en flottants bruts (pour les inspecter).
    fn dump(&self, dir: &str) {
        let _ = std::fs::create_dir_all(dir);
        for (name, data) in [("shore0.f32", &self.tex0), ("shore1.f32", &self.tex1)] {
            let mut bytes = Vec::with_capacity(data.len() * 16);
            for v in data.iter() {
                for c in v {
                    bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
            let _ = std::fs::write(format!("{dir}/{name}"), bytes);
        }
    }

    /// Interpolation bilinéaire des deux textures au point (x, z) du monde.
    pub fn sample(&self, x: f32, z: f32) -> Option<([f32; 4], [f32; 4])> {
        let n = self.n;
        let fx = (x + self.half) / self.cell - 0.5;
        let fz = (z + self.half) / self.cell - 0.5;
        if fx < 0.0 || fz < 0.0 || fx >= (n - 1) as f32 || fz >= (n - 1) as f32 {
            return None;
        }
        let (i, j) = (fx.floor() as usize, fz.floor() as usize);
        let (u, v) = (fx - i as f32, fz - j as f32);
        let lerp4 = |data: &Vec<[f32; 4]>| {
            let a = data[j * n + i];
            let b = data[j * n + i + 1];
            let c = data[(j + 1) * n + i];
            let d = data[(j + 1) * n + i + 1];
            let mut o = [0.0; 4];
            for k in 0..4 {
                o[k] = (a[k] * (1.0 - u) + b[k] * u) * (1.0 - v) + (c[k] * (1.0 - u) + d[k] * u) * v;
            }
            o
        };
        Some((lerp4(&self.tex0), lerp4(&self.tex1)))
    }
}

fn make_image(n: usize, data: &[[f32; 4]]) -> Image {
    let mut bytes = Vec::with_capacity(n * n * 16);
    for v in data {
        for c in v {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    let mut image = Image::new(
        Extent3d {
            width: n as u32,
            height: n as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::nearest());
    image
}

/// Profil d'une vague côtière : crête pointue, front raide (asymétrie `a` dans [0, 1[).
/// Retourne (hauteur normalisée, dérivée par rapport à la phase, phase déformée).
pub fn shore_profile(theta: f32, a: f32) -> (f32, f32, f32) {
    let u = theta.rem_euclid(TAU);
    let w = u - a * (1.0 - u.cos());
    (w.cos(), -w.sin() * (1.0 - a * u.sin()), w)
}

/// État de l'eau en (x, z) dû aux vagues côtières (même calcul que `shore_wave` dans
/// `ocean_common.wesl`, sans le déplacement horizontal).
pub struct ShoreState {
    /// poids de la houle du large (0 près de l'île, où les vagues côtières prennent le relais)
    pub swell_w: f32,
    /// poids de la mer du vent (0 au ras du rivage)
    pub wind_w: f32,
    /// élévation due aux vagues côtières (m)
    pub eta: f32,
    /// sur la plage : hauteur de la nappe du jet de rive si elle couvre ce point
    pub beach_y: Option<f32>,
}

/// Bruit de phase le long des crêtes (`shore_phase_noise` côté shader). Approximation CPU :
/// le bruit de valeur des shaders n'est pas reproduit exactement, on s'en passe (l'écart est
/// de quelques centimètres sur la hauteur de l'eau).
fn phase_noise(_x: f32, _z: f32) -> f32 {
    0.0
}

pub fn shore_state(field: &ShoreField, x: f32, z: f32, t: f32) -> ShoreState {
    let mut st = ShoreState { swell_w: 1.0, wind_w: 1.0, eta: 0.0, beach_y: None };
    let Some((s0, s1)) = field.sample(x, z) else {
        return st;
    };
    let (arrival, h, _foam, ground) = (s0[0], s0[1], s0[2], s0[3]);
    let depth = (-ground).max(0.0);
    let w_shore = 1.0 - smoothstep(9.0, 14.5, depth);
    st.swell_w = 1.0 - w_shore;
    st.wind_w = smoothstep(0.0, 1.2, depth);
    if arrival < 0.0 {
        return st;
    }
    let along = phase_noise(x, z);
    let theta = |k: usize| TAU / SHORE_PERIODS[k] * (t - arrival) + k as f32 * 2.1 + along * (1.0 + k as f32);
    if depth > DRY_DEPTH {
        let b = (h / (BREAK_GAMMA * depth).max(1e-3)).clamp(0.0, 1.0);
        let asym = 0.15 + 0.6 * smoothstep(0.55, 1.0, b);
        let mut eta = 0.0;
        for k in 0..2 {
            let (prof, _, _) = shore_profile(theta(k), asym);
            eta += 0.5 * h * SHORE_SPLIT[k] * prof;
        }
        let limit = 0.55 * BREAK_GAMMA * depth + 0.02;
        let kk = 0.3 * limit + 0.01;
        let hh = (0.5 + 0.5 * (limit - eta) / kk).clamp(0.0, 1.0);
        st.eta = (limit + (eta - limit) * hh - kk * hh * (1.0 - hh)) * w_shore;
        return st;
    }
    // jet de rive sur la plage
    let run_up = 0.55 * s1[2] + 0.04;
    let mut front: f32 = -1.0;
    for k in 0..2 {
        let tau = (theta(k) / TAU).rem_euclid(1.0);
        front = front.max(run_up * SHORE_SPLIT[k] / SHORE_SPLIT[0] * swash_curve(tau));
    }
    if ground < front {
        st.beach_y = Some(ground + 0.03 + 0.08 * ((front - ground) / run_up.max(0.01)).min(1.0));
    }
    st
}

/// Courbe du jet de rive sur une période (0..1) : montée rapide, retrait plus lent.
pub fn swash_curve(tau: f32) -> f32 {
    let u = (tau / 0.85).clamp(0.0, 1.0);
    (std::f32::consts::PI * u.powf(0.6)).sin().max(0.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
