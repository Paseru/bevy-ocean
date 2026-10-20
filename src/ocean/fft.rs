//! Petites vagues (de 2 cm à 2,5 m) simulées par transformée de Fourier (méthode de
//! Tessendorf) sur le CPU, en deux "cascades" de tailles différentes.
//!
//! Chaque image calcule, pour chaque cascade, les pentes et les dérivées secondes de la
//! hauteur de l'eau, envoyées au GPU dans deux textures (tableaux de 2 couches) :
//! - `slopes`  : (dh/dx, dh/dz, d²h/dx², d²h/dz²)
//! - `extra`   : (d²h/dxdz, h, 0, 0)
//! Les pentes servent aux normales (reflets, réfraction), les dérivées secondes aux caustiques.

use std::f32::consts::{PI, TAU};

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    math::Vec2,
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
    },
};

use crate::noise::Rng;

#[derive(Clone, Copy, Default, Debug)]
struct C {
    re: f32,
    im: f32,
}

impl C {
    #[inline]
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    #[inline]
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    #[inline]
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    #[inline]
    fn scale(self, s: f32) -> C {
        C::new(self.re * s, self.im * s)
    }
}

/// FFT inverse (exposant positif, sans normalisation), radix 2, itérative.
struct Fft {
    n: usize,
    twiddles: Vec<C>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let twiddles = (0..n / 2)
            .map(|i| {
                let a = TAU * i as f32 / n as f32;
                C::new(a.cos(), a.sin())
            })
            .collect();
        Self { n, twiddles, rev }
    }

    fn inverse(&self, data: &mut [C]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i];
            if i < j {
                data.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let w = self.twiddles[k * step];
                    let u = data[start + k];
                    let v = data[start + k + half].mul(w);
                    data[start + k] = u.add(v);
                    data[start + k + half] = u.sub(v);
                }
            }
            len *= 2;
        }
    }

    /// FFT 2D inverse en place (lignes puis colonnes).
    fn inverse_2d(&self, data: &mut [C], column: &mut [C]) {
        let n = self.n;
        for row in data.chunks_mut(n) {
            self.inverse(row);
        }
        for x in 0..n {
            for z in 0..n {
                column[z] = data[z * n + x];
            }
            self.inverse(column);
            for z in 0..n {
                data[z * n + x] = column[z];
            }
        }
    }
}

/// Une cascade de vagues simulées par FFT.
pub struct FftCascade {
    /// Taille du motif (m) : il se répète tous les `size` mètres.
    pub size: f32,
    n: usize,
    h0: Vec<C>,
    h0_neg_conj: Vec<C>,
    omega: Vec<f32>,
    kx: Vec<f32>,
    kz: Vec<f32>,
    /// Variance totale des pentes portée par cette cascade.
    pub slope_variance: f32,
    /// Bande de nombres d'onde simulée.
    pub k_band: (f32, f32),
    work: [Vec<C>; 3],
}

/// Spectre de la mer du vent (même forme que pour les grandes vagues).
#[derive(Clone, Copy, Debug)]
pub struct WindSpectrum {
    pub wind_dir: Vec2,
    pub peak_omega: f32,
    /// Coefficient d'amplitude (alpha de JONSWAP déjà normalisé).
    pub alpha: f32,
    pub boost: f32,
}

fn dispersion(k: f32) -> f32 {
    (9.81 * k + 7.4e-5 * k * k * k).sqrt()
}

fn d_dispersion(k: f32) -> f32 {
    (9.81 + 3.0 * 7.4e-5 * k * k) / (2.0 * dispersion(k))
}

fn jonswap(omega: f32, omega_p: f32) -> f32 {
    if omega <= 1e-4 {
        return 0.0;
    }
    let gamma: f32 = 3.3;
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = (-(omega - omega_p).powi(2) / (2.0 * sigma * sigma * omega_p * omega_p)).exp();
    9.81 * 9.81 / omega.powi(5) * (-1.25 * (omega_p / omega).powi(4)).exp() * gamma.powf(r)
}

impl FftCascade {
    /// `k_band` : bande de nombres d'onde gardée par cette cascade.
    pub fn new(size: f32, n: usize, k_band: (f32, f32), spec: &WindSpectrum, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let dk = TAU / size;
        let mut h0 = vec![C::default(); n * n];
        let mut omega = vec![0.0; n * n];
        let mut kx = vec![0.0; n * n];
        let mut kz = vec![0.0; n * n];
        let wind_angle = spec.wind_dir.y.atan2(spec.wind_dir.x);
        let mut slope_variance = 0.0;
        for j in 0..n {
            for i in 0..n {
                let fi = if i < n / 2 { i as i32 } else { i as i32 - n as i32 };
                let fj = if j < n / 2 { j as i32 } else { j as i32 - n as i32 };
                let k = Vec2::new(fi as f32 * dk, fj as f32 * dk);
                let idx = j * n + i;
                kx[idx] = k.x;
                kz[idx] = k.y;
                let km = k.length();
                // tirage aléatoire même hors bande pour garder la même séquence
                let (g1, g2) = (rng.normal(), rng.normal());
                if km < k_band.0 || km >= k_band.1 || km < 1e-6 {
                    continue;
                }
                let w = dispersion(km);
                omega[idx] = w;
                // spectre 2D en nombre d'onde : S(w) dw/dk D(theta) / k
                let s_w = spec.alpha * jonswap(w, spec.peak_omega);
                let theta = k.y.atan2(k.x) - wind_angle;
                // étalement directionnel large (cos^2s du demi-angle, s = 3)
                let s_par = 3.0f32;
                let d = (theta * 0.5).cos().abs().powf(2.0 * s_par);
                let norm = 2.0 * PI * gamma_ratio(s_par);
                let psi = s_w * d_dispersion(km) * d / norm / km;
                let amp = (psi * dk * dk * 0.25).sqrt() * spec.boost;
                h0[idx] = C::new(g1 * amp, g2 * amp);
                slope_variance += 2.0 * (g1 * g1 + g2 * g2) * amp * amp * km * km;
            }
        }
        let mut h0_neg_conj = vec![C::default(); n * n];
        for j in 0..n {
            for i in 0..n {
                let ni = (n - i) % n;
                let nj = (n - j) % n;
                h0_neg_conj[j * n + i] = h0[nj * n + ni].conj();
            }
        }
        Self {
            size,
            n,
            h0,
            h0_neg_conj,
            omega,
            kx,
            kz,
            slope_variance,
            k_band,
            work: [vec![C::default(); n * n], vec![C::default(); n * n], vec![C::default(); n * n]],
        }
    }

    /// Spectre des pentes et dérivées secondes à l'instant `t` (avant les FFT inverses).
    fn spectrum(&mut self, t: f64) {
        let n = self.n;
        let [w0, w1, w2] = &mut self.work;
        for idx in 0..n * n {
            let ph = (self.omega[idx] as f64 * t).rem_euclid(std::f64::consts::TAU) as f32;
            let (s, c) = ph.sin_cos();
            let e = C::new(c, s);
            let h = self.h0[idx].mul(e).add(self.h0_neg_conj[idx].mul(e.conj()));
            let (kx, kz) = (self.kx[idx], self.kz[idx]);
            // i*kx*h + i*(i*kz*h)
            let ikx_h = C::new(-kx * h.im, kx * h.re);
            let ikz_h = C::new(-kz * h.im, kz * h.re);
            w0[idx] = ikx_h.add(C::new(-ikz_h.im, ikz_h.re));
            // -kx² h + i (-kz² h)
            let a = h.scale(-kx * kx);
            let b = h.scale(-kz * kz);
            w1[idx] = a.add(C::new(-b.im, b.re));
            // -kx kz h + i h
            let a = h.scale(-kx * kz);
            w2[idx] = a.add(C::new(-h.im, h.re));
        }
    }
}

/// Rapport Γ(s+1)² / Γ(2s+1) * 2^(2s) / (2 sqrt(pi))... normalisation de cos^2s(θ/2).
fn gamma_ratio(s: f32) -> f32 {
    // intégrale de |cos(θ/2)|^(2s) sur [-π, π] divisée par 2π, calculée numériquement
    let n = 512;
    let mut sum = 0.0;
    for i in 0..n {
        let th = -PI + TAU * (i as f32 + 0.5) / n as f32;
        sum += (th * 0.5).cos().abs().powf(2.0 * s);
    }
    sum / n as f32
}

/// Ressource : les cascades FFT et leurs textures.
#[derive(Resource)]
pub struct FftWaves {
    pub cascades: Vec<FftCascade>,
    fft: Fft,
    pub slopes: Handle<Image>,
    pub extra: Handle<Image>,
    n: usize,
    mips: u32,
}

/// Convertit un f32 en f16 (bits), arrondi au plus proche.
fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = sign as u32 | ((e as u32) << 10) | (mant >> 13);
    // arrondi
    let round = (mant >> 12) & 1;
    (h + round) as u16
}

fn make_image(n: usize, layers: usize, mips: u32) -> Image {
    let mut size = 0;
    for _ in 0..layers {
        for m in 0..mips {
            let s = (n >> m).max(1);
            size += s * s * 8;
        }
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width: n as u32,
            height: n as u32,
            depth_or_array_layers: layers as u32,
        },
        TextureDimension::D2,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(vec![0u8; size]);
    image.texture_descriptor.mip_level_count = mips;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 8,
        ..default()
    });
    image
}

impl FftWaves {
    pub fn new(cascades: Vec<FftCascade>, images: &mut Assets<Image>) -> Self {
        let n = cascades[0].n;
        let mips = n.trailing_zeros() + 1;
        let layers = cascades.len();
        Self {
            fft: Fft::new(n),
            slopes: images.add(make_image(n, layers, mips)),
            extra: images.add(make_image(n, layers, mips)),
            cascades,
            n,
            mips,
        }
    }

    /// Simule les cascades à l'instant `t` et met à jour les textures.
    /// Le travail est réparti sur plusieurs cœurs : spectres, puis les 6 FFT inverses
    /// (3 par cascade), puis la conversion des deux textures, chacun en parallèle.
    pub fn update(&mut self, t: f64, images: &mut Assets<Image>) {
        let n = self.n;
        let mips = self.mips;
        let layers = self.cascades.len();
        let fft = &self.fft;
        std::thread::scope(|s| {
            for c in self.cascades.iter_mut() {
                s.spawn(move || {
                    c.spectrum(t);
                    let [w0, w1, w2] = &mut c.work;
                    std::thread::scope(|s2| {
                        for w in [w0, w1, w2] {
                            s2.spawn(move || {
                                let mut column = vec![C::default(); n];
                                fft.inverse_2d(w, &mut column);
                            });
                        }
                    });
                });
            }
        });
        // remplissage des textures (mip 0 puis réductions 2x2)
        let fill = |which: usize, cascades: &Vec<FftCascade>| -> Vec<u8> {
            let mut data = Vec::with_capacity(layers * n * n * 8 * 4 / 3 + 64);
            let mut level = vec![[0.0f32; 4]; n * n];
            for c in cascades.iter().take(layers) {
                for idx in 0..n * n {
                    level[idx] = if which == 0 {
                        [c.work[0][idx].re, c.work[0][idx].im, c.work[1][idx].re, c.work[1][idx].im]
                    } else {
                        [c.work[2][idx].re, c.work[2][idx].im, 0.0, 0.0]
                    };
                }
                let mut size = n;
                for m in 0..mips {
                    if m > 0 {
                        let ns = (size / 2).max(1);
                        let mut next = vec![[0.0f32; 4]; ns * ns];
                        for z in 0..ns {
                            for x in 0..ns {
                                let mut acc = [0.0f32; 4];
                                for (dz, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                                    let v = level[(2 * z + dz) * size + 2 * x + dx];
                                    for ch in 0..4 {
                                        acc[ch] += 0.25 * v[ch];
                                    }
                                }
                                next[z * ns + x] = acc;
                            }
                        }
                        level = next;
                        size = ns;
                    }
                    for v in level.iter().take(size * size) {
                        for ch in 0..4 {
                            data.extend_from_slice(&f32_to_f16(v[ch]).to_le_bytes());
                        }
                    }
                }
                level = vec![[0.0f32; 4]; n * n];
            }
            data
        };
        let cascades = &self.cascades;
        let (slopes, extra) = std::thread::scope(|s| {
            let a = s.spawn(|| fill(0, cascades));
            let b = s.spawn(|| fill(1, cascades));
            (a.join().unwrap(), b.join().unwrap())
        });
        for (handle, bytes) in [(&self.slopes, slopes), (&self.extra, extra)] {
            if let Some(mut img) = images.get_mut(handle) {
                if let Some(data) = img.data.as_mut() {
                    let len = data.len().min(bytes.len());
                    data[..len].copy_from_slice(&bytes[..len]);
                }
            }
        }
    }
}
