//! Grandes vagues : somme de vagues de Gerstner tirées d'un spectre JONSWAP (houle + mer du vent).
//! Les vagues plus petites que 2,5 m sont simulées par FFT (voir `fft.rs`).
//!
//! Le même champ est évalué sur le CPU (hauteur de l'eau à une position, utile pour savoir
//! si la caméra est sous l'eau, ou plus tard pour faire flotter des objets) et sur le GPU.

use std::f32::consts::{PI, TAU};

use bevy::math::{Vec2, Vec3};

use crate::noise::Rng;

pub const GRAVITY: f32 = 9.81;
/// Nombre maximal de vagues envoyées au GPU (doit correspondre à `MAX_WAVES` dans les shaders).
pub const MAX_WAVES: usize = 40;
/// Longueur d'onde en dessous de laquelle les vagues sont simulées par FFT.
pub const FFT_SPLIT_WAVELENGTH: f32 = 2.5;
/// Nombre de composantes de houle en tête de liste (même valeur que `SWELL_WAVES` côté shader).
pub const SWELL_COUNT: usize = 3;

/// Une composante du champ de vagues.
#[derive(Clone, Copy, Debug)]
pub struct Wave {
    /// Direction de propagation (unitaire) dans le plan XZ.
    pub dir: Vec2,
    /// Nombre d'onde k = 2π/λ (rad/m).
    pub k: f32,
    /// Amplitude (m).
    pub amplitude: f32,
    /// Pulsation (rad/s).
    pub omega: f32,
    /// Phase initiale (rad).
    pub phase0: f32,
    /// Raideur de Gerstner (déplacement horizontal relatif).
    pub steepness: f32,
}

impl Wave {
    pub fn wavelength(&self) -> f32 {
        TAU / self.k
    }
}

/// Paramètres de l'état de la mer.
#[derive(Clone, Debug)]
pub struct SeaState {
    /// Direction de propagation de la mer du vent.
    pub wind_dir: Vec2,
    /// Longueur d'onde au pic du spectre de la mer du vent (m).
    pub peak_wavelength: f32,
    /// Hauteur significative de la mer du vent (m).
    pub wind_sea_hs: f32,
    /// Direction de propagation de la houle.
    pub swell_dir: Vec2,
    /// Hauteur significative de la houle (m).
    pub swell_hs: f32,
    /// Multiplicateur appliqué aux petites vagues (FFT).
    pub detail_boost: f32,
    pub seed: u64,
}

impl Default for SeaState {
    fn default() -> Self {
        Self {
            // alizé d'est : les vagues partent vers l'ouest-sud-ouest
            wind_dir: Vec2::new(-1.0, 0.35).normalize(),
            peak_wavelength: 7.5,
            wind_sea_hs: 0.22,
            // houle modérée venant du sud-est : elle déferle sur la crête du récif
            swell_dir: Vec2::new(-0.55, -1.0).normalize(),
            swell_hs: 0.75,
            detail_boost: 1.0,
            seed: 7,
        }
    }
}

/// Champ de grandes vagues.
#[derive(Clone, Debug)]
pub struct WaveField {
    /// Vagues triées : houle, puis mer du vent des plus longues aux plus courtes.
    pub waves: Vec<Wave>,
    /// Phase courante de chaque vague (calculée en f64 pour garder la précision dans le temps).
    pub phases: Vec<f32>,
    /// Coefficient du spectre JONSWAP (normalisé pour la hauteur significative demandée).
    pub alpha: f32,
    /// Pulsation au pic du spectre.
    pub peak_omega: f32,
    pub wind_dir: Vec2,
}

/// Relation de dispersion en eau profonde avec tension de surface.
pub fn dispersion(k: f32) -> f32 {
    (GRAVITY * k + 7.4e-5 * k * k * k).sqrt()
}

/// Spectre JONSWAP (sans normalisation d'amplitude).
pub fn jonswap(omega: f32, omega_p: f32) -> f32 {
    if omega <= 0.0 {
        return 0.0;
    }
    let gamma: f32 = 3.3;
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = (-(omega - omega_p).powi(2) / (2.0 * sigma * sigma * omega_p * omega_p)).exp();
    GRAVITY * GRAVITY / omega.powi(5) * (-1.25 * (omega_p / omega).powi(4)).exp() * gamma.powf(r)
}

fn rotate(v: Vec2, angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

impl WaveField {
    pub fn new(state: &SeaState) -> Self {
        let mut rng = Rng::new(state.seed);
        let mut waves = Vec::with_capacity(MAX_WAVES);

        // --- Houle : quelques longues vagues régulières -------------------------------------
        let swell_var = (state.swell_hs / 4.0).powi(2);
        let swell = [(74.0, 0.50, -0.10), (55.0, 0.30, 0.06), (41.0, 0.20, 0.16)];
        for (lambda, weight, angle) in swell {
            let k = TAU / lambda;
            waves.push(Wave {
                dir: rotate(state.swell_dir, angle),
                k,
                amplitude: (2.0 * swell_var * weight).sqrt(),
                omega: dispersion(k),
                phase0: rng.range(0.0, TAU),
                steepness: 0.0,
            });
        }

        // --- Mer du vent : spectre JONSWAP ---------------------------------------------------
        let k_p = TAU / state.peak_wavelength;
        let omega_p = dispersion(k_p);
        let mut m0 = 0.0;
        let n_int = 4000;
        let (w_min, w_max) = (0.3 * omega_p, 60.0 * omega_p);
        for i in 0..n_int {
            let w = w_min + (w_max - w_min) * (i as f32 + 0.5) / n_int as f32;
            m0 += jonswap(w, omega_p) * (w_max - w_min) / n_int as f32;
        }
        let alpha = (state.wind_sea_hs / 4.0).powi(2) / m0;

        let count = MAX_WAVES - waves.len();
        let (lambda_max, lambda_min) = (26.0f32, FFT_SPLIT_WAVELENGTH);
        let ln_max = lambda_max.ln();
        let ln_min = lambda_min.ln();
        for i in 0..count {
            let t = (i as f32 + rng.range(0.2, 0.8)) / count as f32;
            let lambda = (ln_max + (ln_min - ln_max) * t).exp();
            let k = TAU / lambda;
            let omega = dispersion(k);
            let dl = (ln_max - ln_min) / count as f32;
            let k_hi = TAU / (lambda * (-0.5 * dl).exp());
            let k_lo = TAU / (lambda * (0.5 * dl).exp());
            let d_omega = dispersion(k_hi) - dispersion(k_lo);
            // amplitude de Rayleigh autour de la valeur du spectre (aspect moins régulier)
            let rayleigh = (-2.0 * rng.f32().max(1e-4).ln()).sqrt() / 1.2533;
            let amplitude = (2.0 * alpha * jonswap(omega, omega_p) * d_omega).sqrt() * (0.55 + 0.45 * rayleigh);
            let spread = 0.42 + 0.35 * (1.0 - (lambda / state.peak_wavelength).min(1.0));
            let angle = (rng.normal() * spread).clamp(-1.3, 1.3);
            waves.push(Wave {
                dir: rotate(state.wind_dir, angle),
                k,
                amplitude,
                omega,
                phase0: rng.range(0.0, TAU),
                steepness: 0.0,
            });
        }

        // raideur de Gerstner : vagues pointues sans jamais se replier
        let slope_sum: f32 = waves.iter().map(|w| w.k * w.amplitude).sum();
        let q = (0.7 / slope_sum).min(1.5);
        for w in waves.iter_mut() {
            w.steepness = q;
        }

        let phases = waves.iter().map(|w| w.phase0).collect();
        Self {
            waves,
            phases,
            alpha,
            peak_omega: omega_p,
            wind_dir: state.wind_dir,
        }
    }

    /// Met à jour la phase de chaque vague pour l'instant `t` (secondes).
    pub fn update_phases(&mut self, t: f64) {
        for (i, w) in self.waves.iter().enumerate() {
            let ph = (w.phase0 as f64 - w.omega as f64 * t).rem_euclid(std::f64::consts::TAU);
            self.phases[i] = ph as f32;
        }
    }

    /// Déplacement (dx, dy, dz) d'un point de repos `p` de la surface.
    pub fn displacement(&self, p: Vec2) -> Vec3 {
        let mut d = Vec3::ZERO;
        for (w, &ph) in self.waves.iter().zip(&self.phases) {
            let theta = w.k * w.dir.dot(p) + ph;
            let (s, c) = theta.sin_cos();
            d.x -= w.steepness * w.amplitude * w.dir.x * s;
            d.z -= w.steepness * w.amplitude * w.dir.y * s;
            d.y += w.amplitude * c;
        }
        d
    }

    /// Déplacement avec des poids sur la houle (3 premières vagues) et la mer du vent.
    pub fn displacement_weighted(&self, p: Vec2, swell_w: f32, wind_w: f32) -> Vec3 {
        let mut d = Vec3::ZERO;
        for (i, (w, &ph)) in self.waves.iter().zip(&self.phases).enumerate() {
            let wt = if i < SWELL_COUNT { swell_w } else { wind_w };
            if wt <= 0.0 {
                continue;
            }
            let theta = w.k * w.dir.dot(p) + ph;
            let (s, c) = theta.sin_cos();
            let a = w.amplitude * wt;
            d.x -= w.steepness * a * w.dir.x * s;
            d.z -= w.steepness * a * w.dir.y * s;
            d.y += a * c;
        }
        d
    }

    /// Hauteur de l'eau (au-dessus du niveau moyen) avec les poids de `displacement_weighted`.
    pub fn height_at_weighted(&self, x: f32, z: f32, swell_w: f32, wind_w: f32) -> f32 {
        let target = Vec2::new(x, z);
        let mut p = target;
        for _ in 0..5 {
            let d = self.displacement_weighted(p, swell_w, wind_w);
            p = target - Vec2::new(d.x, d.z);
        }
        self.displacement_weighted(p, swell_w, wind_w).y
    }

    /// Hauteur de la surface de l'eau (au-dessus du niveau moyen) au point (x, z) du monde.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let target = Vec2::new(x, z);
        let mut p = target;
        // inversion du déplacement horizontal par point fixe
        for _ in 0..5 {
            let d = self.displacement(p);
            p = target - Vec2::new(d.x, d.z);
        }
        self.displacement(p).y
    }

    /// Écart-type de la hauteur (m).
    pub fn height_std(&self) -> f32 {
        self.waves.iter().map(|w| 0.5 * w.amplitude * w.amplitude).sum::<f32>().sqrt()
    }

    /// Variance des pentes portée par ces vagues.
    pub fn slope_variance(&self) -> f32 {
        self.waves
            .iter()
            .map(|w| 0.5 * (w.k * w.amplitude).powi(2))
            .sum()
    }
}

#[allow(dead_code)]
pub fn describe(field: &WaveField) -> String {
    let mut s = String::new();
    for (i, w) in field.waves.iter().enumerate() {
        s.push_str(&format!(
            "{i:3}: λ={:7.3} m  A={:7.4} m  ka={:6.4}  dir=({:5.2},{:5.2})  T={:5.2} s\n",
            w.wavelength(),
            w.amplitude,
            w.k * w.amplitude,
            w.dir.x,
            w.dir.y,
            2.0 * PI / w.omega
        ));
    }
    s
}
