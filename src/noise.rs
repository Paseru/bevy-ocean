//! Bruits procéduraux déterministes (CPU) utilisés pour générer l'île et le fond marin.

use bevy::math::Vec2;

#[inline]
fn hash2(ix: i32, iy: i32, seed: u32) -> u32 {
    let mut h = (ix as u32).wrapping_mul(0x8da6_b343)
        ^ (iy as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h
}

#[inline]
fn grad(ix: i32, iy: i32, seed: u32) -> Vec2 {
    let h = hash2(ix, iy, seed);
    let a = (h as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    Vec2::new(a.cos(), a.sin())
}

#[inline]
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Bruit de gradient (Perlin) 2D, valeurs dans ~[-0.7, 0.7].
pub fn perlin(p: Vec2, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let (ix, iy) = (i.x as i32, i.y as i32);
    let g00 = grad(ix, iy, seed).dot(f);
    let g10 = grad(ix + 1, iy, seed).dot(f - Vec2::new(1.0, 0.0));
    let g01 = grad(ix, iy + 1, seed).dot(f - Vec2::new(0.0, 1.0));
    let g11 = grad(ix + 1, iy + 1, seed).dot(f - Vec2::new(1.0, 1.0));
    let u = quintic(f.x);
    let v = quintic(f.y);
    let a = g00 + (g10 - g00) * u;
    let b = g01 + (g11 - g01) * u;
    a + (b - a) * v
}

/// Somme fractale de bruits de Perlin, normalisée dans ~[-1, 1].
pub fn fbm(p: Vec2, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut q = p;
    // rotation entre octaves pour casser l'alignement sur la grille
    let (c, s) = (0.8f32, 0.6f32);
    for o in 0..octaves {
        sum += amp * perlin(q, seed.wrapping_add(o * 1013));
        norm += amp;
        amp *= 0.5;
        q = Vec2::new(c * q.x - s * q.y, s * q.x + c * q.y) * 2.03 + Vec2::splat(17.3);
    }
    1.4 * sum / norm
}

/// Bruit "ridged" (crêtes) dans [0, 1].
pub fn ridged(p: Vec2, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut q = p;
    for o in 0..octaves {
        let n = 1.0 - (perlin(q, seed.wrapping_add(o * 7919)) * 1.4).abs();
        sum += amp * n * n;
        norm += amp;
        amp *= 0.5;
        q = q * 2.07 + Vec2::splat(5.1);
    }
    sum / norm
}

/// Petit générateur pseudo-aléatoire déterministe (SplitMix64).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniforme dans [0, 1).
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f32()
    }
    /// Loi normale centrée réduite (Box-Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
