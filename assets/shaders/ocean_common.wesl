//! Paramètres et fonctions partagés par tous les shaders de la scène :
//! vagues, ciel, optique de l'eau, caustiques, brume sous-marine.

import bevy_pbr::render::mesh_view_bindings::{view, lights, globals};
import bevy_pbr::render::shadows::fetch_directional_shadow;
import bevy_pbr::render::mesh_view_types::{
    DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT, DIRECTIONAL_LIGHT_FLAGS_CONTACT_SHADOWS_ENABLED_BIT,
};
@if(SCREEN_SPACE_AMBIENT_OCCLUSION)
import bevy_pbr::render::mesh_view_bindings::screen_space_ambient_occlusion_texture;
@if(BLUE_NOISE_TEXTURE)
import bevy_pbr::render::mesh_view_bindings::blue_noise_texture;
@if(CONTACT_SHADOWS && DEPTH_PREPASS)
import bevy_pbr::render::pbr_functions::calculate_contact_shadow;
@if(CONTACT_SHADOWS && DEPTH_PREPASS)
import bevy_pbr::render::mesh_view_bindings::contact_shadows_settings;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const IOR_WATER: f32 = 1.333;
const MAX_WAVES: u32 = 40u;
/// Les premières vagues de la liste sont la houle du large (remplacée près de l'île par
/// les vagues côtières).
const SWELL_WAVES: u32 = 3u;
const FFT_N: f32 = 128.0;
// mêmes constantes que src/ocean/shore.rs
const SHORE_N: f32 = 1024.0;
const SHORE_HALF: f32 = 640.0;
const SHORE_PERIOD_0: f32 = 8.5;
const SHORE_PERIOD_1: f32 = 9.7;
const SHORE_SPLIT_0: f32 = 0.78;
const SHORE_SPLIT_1: f32 = 0.55;
const BREAK_GAMMA: f32 = 0.78;
const DRY_DEPTH: f32 = 0.04;

struct OceanUniforms {
    // x : temps, y : caméra sous l'eau, z : niveau de la mer, w : nombre de vagues
    params0: vec4<f32>,
    // x : libre, y : caustiques, z : écume, w : échelle de la cubemap du ciel
    params1: vec4<f32>,
    // x : variance des pentes capillaires, y : profondeur du fond, z : hauteur de l'eau sous la caméra
    params2: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    absorption: vec4<f32>,
    scattering: vec4<f32>,
    // cascades FFT : (taille du motif, variance des pentes, octave haute, étendue en octaves)
    fft_a: vec4<f32>,
    fft_b: vec4<f32>,
    // ambiance : x saturation du ciel, y force des ombres des nuages, z couverture nuageuse
    look: vec4<f32>,
    // ambiance : x « dé-brume » de l'horizon, y soleil doré, z éclat des scintillements
    look2: vec4<f32>,
    sh: array<vec4<f32>, 9>,
    waves: array<vec4<f32>, 80>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> ocean: OceanUniforms;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var sky_cube: texture_cube<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var sky_sampler: sampler;
// petites vagues simulées par FFT : (dh/dx, dh/dz, d²h/dx², d²h/dz²) et (d²h/dxdz, h)
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var fft_slopes: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var fft_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var fft_extra: texture_2d_array<f32>;
// vagues côtières précalculées (voir src/ocean/shore.rs) :
// shore_tex0 = (heure d'arrivée des crêtes T, hauteur H, écume, hauteur du fond)
// shore_tex1 = (∂T/∂x, ∂T/∂z, hauteur des brisants, distance à terre)
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var shore_tex0: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var shore_tex1: texture_2d<f32>;
// écume photographiée : (opacité dense, opacité en dentelle, relief dense, relief dentelle)
@group(constants::MATERIAL_BIND_GROUP) @binding(8) var foam_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(9) var foam_sampler: sampler;

// ------------------------------------------------------------------------------------------
// Accès aux paramètres
// ------------------------------------------------------------------------------------------

fn sim_time() -> f32 { return ocean.params0.x; }
fn camera_underwater() -> bool { return ocean.params0.y > 0.5; }
fn sea_level() -> f32 { return ocean.params0.z; }
fn wave_count() -> u32 { return min(u32(ocean.params0.w), MAX_WAVES); }
fn caustics_strength() -> f32 { return ocean.params1.y; }
/// Force du halo lumineux autour du soleil (ambiance).
fn sun_halo_strength() -> f32 { return ocean.look.w; }
/// Effets désactivés pour les mesures de performance (OCEAN_FLAGS).
fn debug_flag(bit: u32) -> bool { return (u32(ocean.params1.x + 0.5) & bit) != 0u; }
fn foam_strength() -> f32 { return ocean.params1.z; }
fn capillary_slope_variance() -> f32 { return ocean.params2.x; }
fn floor_depth() -> f32 { return ocean.params2.y; }
fn debug_mode() -> u32 { return u32(ocean.params2.w + 0.5); }
fn sun_dir() -> vec3<f32> { return ocean.sun_dir.xyz; }
/// Éclairement solaire (lux, RGB) perpendiculaire aux rayons.
fn sun_illuminance() -> vec3<f32> { return ocean.sun_color.rgb * ocean.sun_dir.w; }
fn sun_angular_radius() -> f32 { return ocean.sun_color.w; }
fn water_absorption() -> vec3<f32> { return ocean.absorption.rgb; }
fn water_scattering() -> vec3<f32> { return ocean.scattering.rgb; }
fn water_backscattering() -> vec3<f32> { return ocean.scattering.rgb * ocean.scattering.w; }

fn wave_a(i: u32) -> vec4<f32> { return ocean.waves[2u * i]; }
fn wave_b(i: u32) -> vec4<f32> { return ocean.waves[2u * i + 1u]; }

// ------------------------------------------------------------------------------------------
// Bruits
// ------------------------------------------------------------------------------------------

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

/// Bruit de valeur 2D lissé, dans [0, 1].
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2(1.0, 0.0));
    let c = hash12(i + vec2(0.0, 1.0));
    let d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>, octaves: i32) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i++) {
        s += a * noise2(q);
        q = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * q + vec2(11.7, 3.1);
        a *= 0.5;
    }
    return s;
}

/// Bruit de valeur 3D lissé, dans [0, 1].
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i);
    let n100 = hash13(i + vec3(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm3(p: vec3<f32>, octaves: i32) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < octaves; i++) {
        s += a * noise3(q);
        q = q * 2.03 + vec3(1.7, 9.2, 3.3);
        a *= 0.5;
    }
    return s;
}

// ------------------------------------------------------------------------------------------
// Ciel
// ------------------------------------------------------------------------------------------

/// Luminance du ciel (cd/m², sans le soleil) dans la direction `dir`.
/// La saturation du bleu peut être renforcée comme avec un filtre polarisant (surtout loin
/// du soleil et vers le zénith), réglage d'ambiance `look.x`.
fn sky_radiance(dir: vec3<f32>, lod: f32) -> vec3<f32> {
    var c = textureSampleLevel(sky_cube, sky_sampler, dir, lod).rgb * ocean.params1.w;
    // « dé-brume » : la brume blanchâtre de l'horizon de l'HDRI (estimée sur une version très
    // floue du ciel, pour épargner les nuages) devient un bleu tropical plus profond
    let dh = ocean.look2.x;
    if dh > 0.0 {
        let hz = 1.0 - smoothstep(0.0, 0.4, abs(dir.y));
        let toward_sun = max(dot(dir, sun_dir()), 0.0);
        let blur = textureSampleLevel(sky_cube, sky_sampler, dir, max(lod, 6.0)).rgb * ocean.params1.w;
        let air = min(blur.r, min(blur.g, blur.b));
        let tint = vec3(0.42, 0.72, 1.2);
        let k = dh * hz * (1.0 - 0.25 * toward_sun * toward_sun);
        c = max(c + air * k * (tint - vec3(1.0)), c * 0.3);
    }
    let k = ocean.look.x;
    if k == 1.0 {
        return c;
    }
    let lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    let away = 1.0 - abs(dot(dir, sun_dir()));
    let w = mix(1.0, k, (0.35 + 0.65 * smoothstep(0.0, 0.25, dir.y)) * (0.5 + 0.5 * away));
    // un polariseur assombrit aussi un peu le ciel bleu (pas les nuages blancs)
    let blueness = clamp((c.b - c.r) / max(lum, 1e-4), 0.0, 1.0);
    let dim = 1.0 - (w - 1.0) * 0.35 * blueness;
    return max(vec3(lum) + (c - vec3(lum)) * w, vec3(0.0)) * dim;
}

/// Irradiance (lux) reçue par une surface de normale `n` de la part du ciel (hors soleil).
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let e = ocean.sh[0].rgb * 0.282095
        + ocean.sh[1].rgb * (0.488603 * y)
        + ocean.sh[2].rgb * (0.488603 * z)
        + ocean.sh[3].rgb * (0.488603 * x)
        + ocean.sh[4].rgb * (1.092548 * x * y)
        + ocean.sh[5].rgb * (1.092548 * y * z)
        + ocean.sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0))
        + ocean.sh[7].rgb * (1.092548 * x * z)
        + ocean.sh[8].rgb * (0.546274 * (x * x - y * y));
    return max(e, vec3(0.0));
}

// ------------------------------------------------------------------------------------------
// Optique
// ------------------------------------------------------------------------------------------

/// Facteur de Fresnel exact d'un dioptre, `eta` = n_incident / n_transmis.
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let c = clamp(cos_i, 0.0, 1.0);
    let sin_t2 = eta * eta * (1.0 - c * c);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * c - cos_t) / (eta * c + cos_t);
    let rp = (c - eta * cos_t) / (c + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn vis_smith_ggx(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-6);
}

/// Direction vers le soleil vue depuis l'eau (réfractée à la surface).
fn sun_dir_in_water() -> vec3<f32> {
    let t = refract(-sun_dir(), vec3(0.0, 1.0, 0.0), 1.0 / IOR_WATER);
    return -t;
}

/// Part de l'éclairement solaire qui traverse la surface.
fn sun_surface_transmission() -> f32 {
    return 1.0 - fresnel_dielectric(sun_dir().y, 1.0 / IOR_WATER);
}

/// Phase de Henyey-Greenstein.
fn phase_hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5));
}

/// Phase de diffusion de l'eau : pic avant (particules) + partie isotrope (molécules).
fn water_phase(cos_t: f32) -> f32 {
    return 0.55 * phase_hg(cos_t, 0.88) + 0.45 / (4.0 * PI);
}

/// (1 - exp(-k s)) / k, stable quand k tend vers 0.
fn integrate_exp(k: vec3<f32>, s: f32) -> vec3<f32> {
    let x = k * s;
    let safe = select(k, vec3(1.0), abs(x) < vec3(1e-3));
    return select((1.0 - exp(-x)) / safe, vec3(s) * (1.0 - 0.5 * x), abs(x) < vec3(1e-3));
}

/// Éclairement descendant total sous la surface (lux), soleil + ciel, à la profondeur `depth`.
fn downwelling_irradiance(depth: f32) -> vec3<f32> {
    return downwelling_irradiance_c(depth, 1.0);
}

/// Idem, avec la part du soleil qui passe entre les nuages.
fn downwelling_irradiance_c(depth: f32, cloud: f32) -> vec3<f32> {
    let lw = sun_dir_in_water();
    let k = water_absorption() + water_backscattering();
    let sun = sun_illuminance() * sun_surface_transmission() * lw.y * exp(-k * depth / lw.y) * cloud;
    let sky = sky_irradiance(vec3(0.0, 1.0, 0.0)) * 0.94 * exp(-k * depth * 1.2);
    return sun + sky;
}

/// Luminance diffusée par une colonne d'eau d'épaisseur `s` parcourue le long de `v`
/// depuis la profondeur `z0`, et transmittance correspondante. Retourne (in-scattering, transmittance).
struct Medium {
    inscatter: vec3<f32>,
    transmittance: vec3<f32>,
}

fn water_path(z0: f32, v: vec3<f32>, s: f32, shafts: f32) -> Medium {
    let a = water_absorption();
    let b = water_scattering();
    let bb = water_backscattering();
    // l'image perd la lumière absorbée et une partie de la lumière diffusée
    let c = a + b * 0.8;
    let lw = sun_dir_in_water();
    let k_sun = (a + bb) / lw.y;
    let k_sky = (a + bb) * 1.2;
    let dz = -v.y;
    let e_sun0 = sun_illuminance() * sun_surface_transmission();
    let e_sky0 = sky_irradiance(vec3(0.0, 1.0, 0.0)) * 0.94;
    let j_sun = b * e_sun0 * water_phase(dot(v, lw)) * shafts;
    let j_sky = b * e_sky0 * 0.06;
    var m: Medium;
    m.inscatter = j_sun * exp(-k_sun * z0) * integrate_exp(c + k_sun * dz, s)
        + j_sky * exp(-k_sky * z0) * integrate_exp(c + k_sky * dz, s);
    m.transmittance = exp(-c * s);
    return m;
}

/// Bruit de tramage (bruit bleu si disponible) qui change à chaque image.
fn dither(frag_xy: vec2<f32>) -> f32 {
    @if(BLUE_NOISE_TEXTURE) {
        let size = textureDimensions(blue_noise_texture, 0);
        let layers = textureNumLayers(blue_noise_texture);
        return textureLoad(blue_noise_texture, vec2<i32>(frag_xy) % vec2<i32>(size), i32(globals.frame_count % layers), 0).x;
    }
    @if(!BLUE_NOISE_TEXTURE) {
        return fract(52.9829189 * fract(dot(frag_xy, vec2(0.06711056, 0.00583715)) + f32(globals.frame_count % 64u) * 0.618034));
    }
}

/// Traversée de l'eau entre la caméra (sous l'eau) et un point à la distance `s` dans la
/// direction `v`, avec les rayons de lumière : la diffusion du soleil est modulée le long
/// du trajet par la focalisation des vagues (marche de rayon sur les 40 premiers mètres).
fn water_path_shafts(v: vec3<f32>, s: f32, frag_xy: vec2<f32>) -> Medium {
    let cam = view.world_position;
    let z0 = max(sea_level() - cam.y, 0.0);
    var m = water_path(z0, v, s, 0.0);
    let a = water_absorption();
    let b = water_scattering();
    let bb = water_backscattering();
    let c = a + b * 0.8;
    let lw = sun_dir_in_water();
    let k_sun = (a + bb) / lw.y;
    let dz = -v.y;
    let e_sun0 = sun_illuminance() * sun_surface_transmission() * cloud_shadow(cam);
    let j_sun = b * e_sun0 * water_phase(dot(v, lw));
    var l = min(s, 40.0);
    if debug_flag(2u) {
        l = 0.0;
    }
    let steps = 12;
    let jitter = dither(frag_xy);
    var acc = vec3(0.0);
    // échantillons plus serrés près de la caméra (x = l·u²)
    for (var i = 0; i < steps; i++) {
        let u = (f32(i) + jitter) / f32(steps);
        let x = l * u * u;
        let dx = l * 2.0 * u / f32(steps);
        let q = cam + v * x;
        let z = max(sea_level() - q.y, 0.0);
        let shaft = shaft_pattern(q, z);
        acc += exp(-k_sun * z - c * x) * shaft * dx;
    }
    var sun = j_sun * acc;
    if s > l {
        let zl = z0 + dz * l;
        sun += j_sun * exp(-k_sun * max(zl, 0.0) - c * l) * integrate_exp(c + k_sun * dz, s - l);
    }
    m.inscatter += sun;
    return m;
}

/// Applique la traversée de l'eau entre la caméra (sous l'eau) et le point `p`.
fn underwater_fog(color: vec3<f32>, p: vec3<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    let cam = view.world_position;
    let seg = p - cam;
    let s = length(seg);
    let v = seg / max(s, 1e-4);
    let m = water_path_shafts(v, s, frag_xy);
    return color * m.transmittance + m.inscatter;
}

/// Couleur de l'eau "à l'infini" dans la direction `v` (pour le ciel vu sous l'eau, etc.).
fn underwater_background(v: vec3<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    let m = water_path_shafts(v, 400.0, frag_xy);
    return m.inscatter;
}

// ------------------------------------------------------------------------------------------
// Vagues côtières (précalculées au démarrage, voir src/ocean/shore.rs)
// ------------------------------------------------------------------------------------------

struct ShoreSample {
    /// la houle atteint ce point (heure d'arrivée connue)
    reached: bool,
    arrival: f32,
    height: f32,
    foam: f32,
    ground: f32,
    grad: vec2<f32>,
    breaker: f32,
    land_dist: f32,
}

/// Interpolation bilinéaire des données côtières au point `p` (xz monde).
fn shore_sample(p: vec2<f32>) -> ShoreSample {
    var s: ShoreSample;
    s.reached = false;
    s.arrival = -1.0;
    s.height = 0.0;
    s.foam = 0.0;
    s.ground = sea_level() - floor_depth();
    s.grad = vec2(0.0);
    s.breaker = 0.0;
    s.land_dist = 0.0;
    let cell = 2.0 * SHORE_HALF / SHORE_N;
    let f = (p + vec2(SHORE_HALF)) / cell - 0.5;
    if any(f < vec2(0.0)) || any(f >= vec2(SHORE_N - 1.0)) {
        return s;
    }
    let i = vec2<i32>(floor(f));
    let w = fract(f);
    let a0 = textureLoad(shore_tex0, i, 0);
    let b0 = textureLoad(shore_tex0, i + vec2(1, 0), 0);
    let c0 = textureLoad(shore_tex0, i + vec2(0, 1), 0);
    let d0 = textureLoad(shore_tex0, i + vec2(1, 1), 0);
    let a1 = textureLoad(shore_tex1, i, 0);
    let b1 = textureLoad(shore_tex1, i + vec2(1, 0), 0);
    let c1 = textureLoad(shore_tex1, i + vec2(0, 1), 0);
    let d1 = textureLoad(shore_tex1, i + vec2(1, 1), 0);
    let t0 = mix(mix(a0, b0, w.x), mix(c0, d0, w.x), w.y);
    let t1 = mix(mix(a1, b1, w.x), mix(c1, d1, w.x), w.y);
    s.reached = min(min(a0.x, b0.x), min(c0.x, d0.x)) >= 0.0;
    s.arrival = t0.x;
    s.height = t0.y;
    s.foam = t0.z;
    s.ground = t0.w;
    s.grad = t1.xy;
    s.breaker = t1.z;
    s.land_dist = t1.w;
    return s;
}

/// Profil d'une vague côtière (crête pointue, front raide selon `a`) :
/// (hauteur normalisée, dérivée par rapport à la phase, phase déformée dans [0, 2π[).
fn shore_profile(theta: f32, a: f32) -> vec3<f32> {
    let u = theta - TAU * floor(theta / TAU);
    let w = u - a * (1.0 - cos(u));
    return vec3(cos(w), -sin(w) * (1.0 - a * sin(u)), w);
}

/// Jet de rive sur une période (0..1) : montée rapide, retrait plus lent.
fn swash_curve(tau: f32) -> f32 {
    let u = clamp(tau / 0.85, 0.0, 1.0);
    return max(sin(PI * pow(u, 0.6)), 0.0);
}

fn shore_period(k: u32) -> f32 {
    return select(SHORE_PERIOD_1, SHORE_PERIOD_0, k == 0u);
}

fn shore_split(k: u32) -> f32 {
    return select(SHORE_SPLIT_1, SHORE_SPLIT_0, k == 0u);
}

/// Décalage de phase le long des crêtes (les vagues ne sont pas parfaitement régulières).
fn shore_phase_noise(p: vec2<f32>) -> f32 {
    return (fbm2(p / 85.0 + vec2(31.0, 17.0), 3) - 0.5) * 2.2;
}

/// Phase du train `k` au point `p`.
fn shore_theta(k: u32, arrival: f32, along: f32) -> f32 {
    let omega = TAU / shore_period(k);
    return omega * (sim_time() - arrival) + f32(k) * 2.1 + along * (1.0 + f32(k));
}

struct ShoreWave {
    /// déplacement de la surface (x, y, z)
    disp: vec3<f32>,
    /// pente (∂η/∂x, ∂η/∂z)
    slope: vec2<f32>,
    /// courbure (∂²η/∂x², ∂²η/∂z², ∂²η/∂x∂z), pour les caustiques
    curv: vec3<f32>,
    /// écume des brisants (crête, front, traînée)
    foam: f32,
    /// écume persistante (dérive derrière les zones de déferlement)
    foam_trail: f32,
    /// hauteur relative de la crête (0..1), pour la lumière qui traverse les vagues
    crest: f32,
    /// poids de la houle du large et de la mer du vent
    swell_w: f32,
    wind_w: f32,
    /// nappe d'eau du jet de rive (1) sur la plage
    sheet: f32,
    /// hauteur absolue de l'eau sur la plage (si nappe), sinon juste sous le sable
    beach_y: f32,
    on_land: bool,
    depth: f32,
    ground: f32,
    dir: vec2<f32>,
}

/// Vagues côtières au point de repos `p`.
fn shore_wave(p: vec2<f32>) -> ShoreWave {
    let s = shore_sample(p);
    var o: ShoreWave;
    o.disp = vec3(0.0);
    o.slope = vec2(0.0);
    o.curv = vec3(0.0);
    o.foam = 0.0;
    o.foam_trail = s.foam;
    o.crest = 0.0;
    o.swell_w = 1.0;
    o.wind_w = 1.0;
    o.sheet = 0.0;
    o.ground = s.ground;
    o.depth = max(sea_level() - s.ground, 0.0);
    o.on_land = o.depth <= DRY_DEPTH;
    // hors de la nappe, l'eau passe bien sous le sable (le maillage de l'eau est plus lâche
    // que le relief, il ne doit pas affleurer dans les creux)
    o.beach_y = s.ground - 0.12 - 0.04 * min(s.land_dist, 20.0);
    o.dir = vec2(0.0, 1.0);
    let w_shore = 1.0 - smoothstep(9.0, 14.5, o.depth);
    o.swell_w = 1.0 - w_shore;
    o.wind_w = smoothstep(0.0, 1.2, o.depth);
    if !s.reached || (w_shore <= 0.0 && s.foam <= 0.0) {
        return o;
    }
    let gl = length(s.grad);
    o.dir = s.grad / max(gl, 1e-5);
    let along = shore_phase_noise(p);
    if !o.on_land {
        let depth = o.depth;
        let b = clamp(s.height / max(BREAK_GAMMA * depth, 1e-3), 0.0, 1.0);
        let asym = 0.15 + 0.6 * smoothstep(0.55, 1.0, b);
        var eta = 0.0;
        var slope = vec2(0.0);
        var curv = vec3(0.0);
        var hx = 0.0;
        var foam = 0.0;
        let amp_total = 0.5 * s.height * (SHORE_SPLIT_0 + SHORE_SPLIT_1);
        let q = 0.35 + 0.45 * b;
        for (var k = 0u; k < 2u; k++) {
            let omega = TAU / shore_period(k);
            let pr = shore_profile(shore_theta(k, s.arrival, along), asym);
            let amp = 0.5 * s.height * shore_split(k);
            eta += amp * pr.x;
            slope += amp * pr.y * (-omega) * s.grad;
            let c = -amp * omega * omega * pr.x;
            curv += c * vec3(s.grad.x * s.grad.x, s.grad.y * s.grad.y, s.grad.x * s.grad.y);
            let kk = max(omega * gl, 1e-3);
            hx += q * (amp / max(amp_total, 1e-4)) / kk * sin(pr.z);
            // écume : front (rouleau) et crête des vagues qui déferlent, puis traînée derrière
            let front = smoothstep(TAU - 1.3, TAU - 0.15, pr.z);
            let trail = exp(-pr.z / 1.6);
            foam = max(foam, (front + trail) * shore_split(k) / SHORE_SPLIT_0);
        }
        // les vagues ne peuvent pas dépasser la hauteur de déferlement (min adouci)
        let limit = 0.55 * BREAK_GAMMA * depth + 0.02;
        let kk = 0.3 * limit + 0.01;
        let h = clamp(0.5 + 0.5 * (limit - eta) / kk, 0.0, 1.0);
        let eta_c = mix(limit, eta, h) - kk * h * (1.0 - h);
        o.disp = vec3(o.dir.x * hx, eta_c, o.dir.y * hx) * w_shore;
        o.slope = slope * h * w_shore;
        o.curv = curv * h * w_shore;
        // l'intensité du déferlement varie le long des crêtes
        let vary = 0.78 + 0.22 * smoothstep(0.3, 0.7, fbm2(p / 14.0 + vec2(7.0, 3.0), 3));
        o.foam = foam * smoothstep(0.82, 1.0, b) * w_shore * min(s.height / 0.12, 1.0) * vary;
        o.crest = clamp(eta_c / max(0.5 * s.height, 0.03), 0.0, 1.0) * w_shore;
        return o;
    }
    // --- sur la plage : jet de rive (nappe d'eau qui monte puis se retire)
    let run_up = 0.55 * s.breaker + 0.04;
    var front = -1.0;
    for (var k = 0u; k < 2u; k++) {
        let tau = fract(shore_theta(k, s.arrival, along) / TAU);
        front = max(front, run_up * shore_split(k) / SHORE_SPLIT_0 * swash_curve(tau));
    }
    let rel = s.ground - sea_level();
    if rel < front {
        o.sheet = 1.0;
        o.beach_y = s.ground + 0.03 + 0.08 * min((front - rel) / max(run_up, 0.01), 1.0);
        o.foam = smoothstep(0.12, 0.0, front - rel);
    }
    return o;
}

/// Humidité du sable au point de la plage `p` (hauteur `rel` au-dessus du niveau moyen) :
/// x : pellicule d'eau qui brille juste après le retrait du jet de rive,
/// y : sable humide (sombre), z : dentelle d'écume laissée par le retrait,
/// w : hauteur atteinte par les plus grands jets de rive (laisse de mer juste au-dessus).
fn beach_wetness(p: vec2<f32>, rel: f32) -> vec4<f32> {
    let s = shore_sample(p);
    if !s.reached {
        return vec4(0.0, smoothstep(0.35, 0.0, rel), 0.0, 0.3);
    }
    let along = shore_phase_noise(p);
    let run_up = 0.55 * s.breaker + 0.04;
    var film = 0.0;
    var lace = 0.0;
    var reach = 0.0;
    for (var k = 0u; k < 2u; k++) {
        let r = run_up * shore_split(k) / SHORE_SPLIT_0;
        reach = max(reach, r);
        if rel >= r || r <= 0.0 {
            continue;
        }
        let period = shore_period(k);
        let tau = fract(shore_theta(k, s.arrival, along) / TAU);
        // instant où le jet de rive redescend sous ce point (branche descendante de la courbe)
        let x = clamp(max(rel, 0.0) / r, 0.0, 1.0);
        let u_b = pow((PI - asin(x)) / PI, 1.0 / 0.6);
        let tau_b = 0.85 * u_b;
        let u_a = pow(asin(x) / PI, 1.0 / 0.6);
        let tau_a = 0.85 * u_a;
        var dry = 0.0;
        if tau > tau_b {
            dry = (tau - tau_b) * period;
        } else if tau < tau_a {
            dry = (tau + 1.0 - tau_b) * period;
        } else {
            dry = 0.0;
        }
        film = max(film, exp(-dry / 2.2));
        lace = max(lace, smoothstep(0.0, 0.25, dry) * exp(-dry / 1.6) * smoothstep(0.02, 0.1, r - rel));
    }
    // le sable reste humide jusqu'à la limite des plus grands jets de rive
    let damp = smoothstep(reach + 0.18, reach - 0.08, rel);
    return vec4(film, max(damp, film), lace, reach);
}

// ------------------------------------------------------------------------------------------
// Vagues
// ------------------------------------------------------------------------------------------

/// Position dans le monde d'un sommet du maillage de l'eau (maillage qui suit la caméra) :
/// houle, mer du vent, vagues côtières et jet de rive. xyz : position, w : inutilisé.
/// Partagée par le shader de l'eau et par sa passe de vecteurs de mouvement.
fn water_vertex_world(local: vec3<f32>) -> vec3<f32> {
    let cam = view.world_position;
    let rest = local.xz + cam.xz;
    let r = length(local.xz);
    let spacing = max(r, 0.35) * 0.0196;
    let sw = shore_wave(rest);
    let d = wave_displacement(rest, spacing, sw.swell_w, sw.wind_w);
    var wp = vec3(rest.x + d.x + sw.disp.x, sea_level() + d.y + sw.disp.y, rest.y + d.z + sw.disp.z);
    if sw.on_land {
        // sur la plage : la nappe du jet de rive, sinon sous le sable (d'autant plus bas que
        // le maillage est lâche, pour ne pas affleurer dans les creux)
        let y = select(sw.beach_y - 0.3 * spacing, sw.beach_y, sw.sheet > 0.5);
        wp = vec3(rest.x, y, rest.y);
    }
    return wp;
}

fn wave_weight(i: u32, swell_w: f32, wind_w: f32) -> f32 {
    return select(wind_w, swell_w, i < SWELL_WAVES);
}

/// Déplacement de Gerstner d'un point de repos `p` (filtré selon l'espacement du maillage).
fn wave_displacement(p: vec2<f32>, spacing: f32, swell_w: f32, wind_w: f32) -> vec3<f32> {
    var d = vec3<f32>(0.0);
    let n = wave_count();
    for (var i = 0u; i < n; i++) {
        let a = wave_a(i);
        let b = wave_b(i);
        let fade = smoothstep(2.5 * spacing, 5.0 * spacing, b.w) * wave_weight(i, swell_w, wind_w);
        if fade <= 0.0 {
            continue;
        }
        let theta = a.z * dot(a.xy, p) + b.y;
        let s = sin(theta);
        let c = cos(theta);
        let amp = a.w * fade;
        d.x -= b.z * amp * a.x * s;
        d.z -= b.z * amp * a.y * s;
        d.y += amp * c;
    }
    return d;
}

/// Part de la variance des pentes d'une cascade FFT perdue par le filtrage (mipmaps).
fn fft_lost_fraction(info: vec4<f32>, footprint: f32) -> f32 {
    let dx = info.x / FFT_N;
    let m = max(log2(max(footprint, 1e-6) / dx), 0.0);
    return clamp((info.z + m) / info.w, 0.0, 1.0);
}

struct SurfaceSample {
    normal: vec3<f32>,
    /// variance des pentes non résolues (vagues plus petites que le pixel)
    slope_variance: f32,
    /// hauteur des petites vagues (FFT), utile pour l'écume
    detail_height: f32,
}

/// Normale de la surface au point de repos `p`, filtrée selon l'empreinte du pixel.
/// Doit être appelée dans un flot de contrôle uniforme (échantillonnage de textures).
fn wave_normal(p: vec2<f32>, footprint: f32, sw: ShoreWave) -> SurfaceSample {
    // petites vagues (FFT)
    let fa = textureSample(fft_slopes, fft_sampler, p / ocean.fft_a.x, 0);
    let fb = textureSample(fft_slopes, fft_sampler, p / ocean.fft_b.x, 1);
    let ea = textureSample(fft_extra, fft_sampler, p / ocean.fft_a.x, 0);
    let eb = textureSample(fft_extra, fft_sampler, p / ocean.fft_b.x, 1);
    // la mer du vent s'apaise au ras du rivage et sur la nappe du jet de rive
    let detail = mix(0.35, 1.0, sw.wind_w);
    var lost = (ocean.fft_a.y * fft_lost_fraction(ocean.fft_a, footprint)
        + ocean.fft_b.y * fft_lost_fraction(ocean.fft_b, footprint)) * detail * detail;

    // grandes vagues (Gerstner)
    var sxx = 0.0;
    var szz = 0.0;
    var sxz = 0.0;
    var hx = 0.0;
    var hz = 0.0;
    let n = wave_count();
    for (var i = 0u; i < n; i++) {
        let a = wave_a(i);
        let b = wave_b(i);
        let wt = wave_weight(i, sw.swell_w, sw.wind_w);
        let ka = a.z * a.w * wt;
        let w = smoothstep(1.5 * footprint, 4.0 * footprint, b.w);
        lost += 0.5 * ka * ka * (1.0 - w * w);
        if w <= 0.0 || wt <= 0.0 {
            continue;
        }
        let theta = a.z * dot(a.xy, p) + b.y;
        let s = sin(theta);
        let c = cos(theta);
        let kaw = ka * w;
        hx += kaw * a.x * s;
        hz += kaw * a.y * s;
        let q = b.z * kaw * c;
        sxx += q * a.x * a.x;
        szz += q * a.y * a.y;
        sxz += q * a.x * a.y;
    }
    let tx = vec3(1.0 - sxx, -hx, -sxz);
    let tz = vec3(-sxz, -hz, 1.0 - szz);
    let ng = normalize(cross(tz, tx));
    // pente totale = grandes vagues + vagues côtières + petites vagues
    let slope = -ng.xz / ng.y + sw.slope + (fa.xy + fb.xy) * detail;
    var out: SurfaceSample;
    out.normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    out.slope_variance = lost + capillary_slope_variance();
    out.detail_height = (ea.y + eb.y) * detail;
    return out;
}

// ------------------------------------------------------------------------------------------
// Caustiques
// ------------------------------------------------------------------------------------------

/// Intensité des caustiques (1 = éclairement moyen) sur un point `p` situé à `depth` sous la surface.
/// Calculée à partir de la courbure de la surface (jacobien de la réfraction des rayons du soleil).
fn caustics(p: vec3<f32>, depth: f32, footprint: f32) -> f32 {
    if depth <= 0.02 {
        return 1.0;
    }
    let lw = sun_dir_in_water();
    let path = depth / lw.y;
    let s = p.xz + lw.xz * path;
    let f = path * (1.0 - 1.0 / IOR_WATER);
    // flou : taille apparente du soleil + diffusion vers l'avant, et empreinte du pixel
    let blur = max(footprint * 1.5, 0.02 * path + 0.015);
    // petites vagues (FFT), filtrées au niveau de flou
    let lod_a = log2(blur / (ocean.fft_a.x / FFT_N));
    let lod_b = log2(blur / (ocean.fft_b.x / FFT_N));
    let ha = textureSampleLevel(fft_slopes, fft_sampler, s / ocean.fft_a.x, 0, lod_a);
    let xa = textureSampleLevel(fft_extra, fft_sampler, s / ocean.fft_a.x, 0, lod_a);
    let hb = textureSampleLevel(fft_slopes, fft_sampler, s / ocean.fft_b.x, 1, lod_b);
    let xb = textureSampleLevel(fft_extra, fft_sampler, s / ocean.fft_b.x, 1, lod_b);
    var sw: ShoreWave;
    if debug_flag(4u) {
        sw = shore_wave(vec2(1.0e5));
    } else {
        sw = shore_wave(s);
    }
    let detail = mix(0.35, 1.0, sw.wind_w);
    let wb = 1.0 - smoothstep(0.3, 1.5, lod_b);
    var hxx = (ha.z + hb.z * wb) * detail + sw.curv.x;
    var hzz = (ha.w + hb.w * wb) * detail + sw.curv.y;
    var hxz = (xa.x + xb.x * wb) * detail + sw.curv.z;
    // grandes vagues (on ignore celles dont la courbure est négligeable)
    let n = wave_count();
    for (var i = 0u; i < n; i++) {
        let a = wave_a(i);
        let b = wave_b(i);
        let w = smoothstep(2.0 * blur, 5.0 * blur, b.w) * wave_weight(i, sw.swell_w, sw.wind_w);
        if w <= 0.0 || a.w * a.z * a.z * f < 0.004 {
            continue;
        }
        let c = a.w * a.z * a.z * cos(a.z * dot(a.xy, s) + b.y) * w;
        hxx -= c * a.x * a.x;
        hzz -= c * a.y * a.y;
        hxz -= c * a.x * a.y;
    }
    let j00 = 1.0 + f * hxx;
    let j11 = 1.0 + f * hzz;
    let j01 = f * hxz;
    let det = j00 * j11 - j01 * j01;
    let eps = 0.05 + 0.002 * path;
    let intensity = 1.0 / sqrt(det * det + eps * eps);
    // l'écume des brisants arrête une partie de la lumière
    let foam_block = 1.0 - 0.6 * clamp(sw.foam + 0.5 * sw.foam_trail, 0.0, 1.0);
    return mix(1.0, intensity, caustics_strength()) * foam_block;
}

/// Motif des rayons de lumière dans l'eau au point `q` : la focalisation de la lumière par
/// les vagues (comme les caustiques au fond), un peu plus floue.
fn shaft_pattern(q: vec3<f32>, depth: f32) -> f32 {
    let lw = sun_dir_in_water();
    let path = max(depth, 0.0) / lw.y;
    let s = q.xz + lw.xz * path;
    let f = path * (1.0 - 1.0 / IOR_WATER);
    let blur = 0.025 * path + 0.06;
    let lod_a = log2(blur / (ocean.fft_a.x / FFT_N));
    let lod_b = log2(blur / (ocean.fft_b.x / FFT_N));
    let ha = textureSampleLevel(fft_slopes, fft_sampler, s / ocean.fft_a.x, 0, lod_a);
    let xa = textureSampleLevel(fft_extra, fft_sampler, s / ocean.fft_a.x, 0, lod_a);
    let hb = textureSampleLevel(fft_slopes, fft_sampler, s / ocean.fft_b.x, 1, lod_b);
    let wb = 1.0 - smoothstep(0.3, 1.5, lod_b);
    var hxx = ha.z + hb.z * wb;
    var hzz = ha.w + hb.w * wb;
    var hxz = xa.x;
    // les vagues de Gerstner les plus courtes (ce sont elles qui focalisent le plus)
    let n = wave_count();
    let first = select(0u, n - 6u, n > 6u);
    for (var i = first; i < n; i++) {
        let a = wave_a(i);
        let b = wave_b(i);
        let w = smoothstep(2.0 * blur, 5.0 * blur, b.w);
        let c = a.w * a.z * a.z * cos(a.z * dot(a.xy, s) + b.y) * w;
        hxx -= c * a.x * a.x;
        hzz -= c * a.y * a.y;
        hxz -= c * a.x * a.y;
    }
    let j00 = 1.0 + f * hxx;
    let j11 = 1.0 + f * hzz;
    let j01 = f * hxz;
    let det = j00 * j11 - j01 * j01;
    let eps = 0.12 + 0.004 * path;
    return 1.0 / sqrt(det * det + eps * eps);
}

// ------------------------------------------------------------------------------------------
// Écume : motif en dentelle (cellules de bulles)
// ------------------------------------------------------------------------------------------

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.x, p.y, p.x) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

/// Trous ronds de taille variable (cellules de Voronoï) : 1 dans un trou, 0 sur la mousse.
/// `open` (0..1) agrandit les trous.
fn foam_holes(p: vec2<f32>, open: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var best = 8.0;
    var rad = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let g = vec2(f32(x), f32(y));
            let h = hash22(i + g);
            let r = g + h - f;
            let d = length(r);
            if d < best {
                best = d;
                rad = (0.18 + 0.32 * fract(h.x * 7.13 + h.y * 3.71)) * (0.55 + 0.75 * open);
            }
        }
    }
    return 1.0 - smoothstep(rad - 0.10, rad + 0.04, best);
}

/// Distance au bord de la cellule de Voronoï la plus proche (F2 - F1).
fn voronoi_edge(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var f1 = 8.0;
    var f2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let g = vec2(f32(x), f32(y));
            let o = hash22(i + g);
            let r = g + o - f;
            let d = dot(r, r);
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    return sqrt(f2) - sqrt(f1);
}

/// Échantillonne l'écume photographiée à la position `p` (taille du motif `tile`, en m).
fn foam_texel(p: vec2<f32>, tile: f32, footprint: f32) -> vec4<f32> {
    let lod = log2(max(footprint, 1e-4) / (tile / 1024.0));
    return textureSampleLevel(foam_tex, foam_sampler, p / tile, max(lod, 0.0));
}

/// Dentelle d'écume (écume photographiée), utilisée aussi pour l'écume laissée sur le sable.
fn foam_lace(p: vec2<f32>, open: f32, footprint: f32) -> f32 {
    let v = foam_texel(p, 1.6, footprint).g;
    let thr = 0.35 + 0.45 * open;
    return smoothstep(thr, thr + 0.2, v);
}

/// Écume pour une couverture `coverage` (0..1), à partir de photos d'écume : on garde les
/// zones les plus denses du motif quand la couverture est faible (filaments, dentelle) et
/// tout le motif quand elle est forte (nappe bulleuse). Retourne (masque, relief 0..1).
fn foam_pattern(p: vec2<f32>, coverage: f32, footprint: f32) -> vec2<f32> {
    if coverage <= 0.002 {
        return vec2(0.0);
    }
    let t = sim_time();
    let drift = vec2(t * 0.04, -t * 0.03);
    // déformation du domaine : le motif ne se répète pas et ondule
    let w = vec2(fbm2(p * 0.3 + drift, 3), fbm2(p * 0.3 + vec2(5.2, 1.3) - drift, 3)) - 0.5;
    let a = foam_texel(p + w * 1.4 + drift * 2.0, 2.6, footprint);
    let b = foam_texel(p * 1.0 + w * 0.9 + vec2(0.37, 0.71) - drift * 1.5, 1.45, footprint);
    // répartition en plaques à grande échelle
    let patches = fbm2(p * 0.09 + drift * 0.3, 4);
    let spread = smoothstep(0.85 - coverage, 1.15 - coverage, patches + 0.35 * coverage);
    let v = max(a.r, b.g * 0.95);
    let thr = 1.0 - coverage;
    var m = smoothstep(thr - 0.1, thr + 0.18, v + 0.35 * coverage);
    // de loin, le motif est plus fin que le pixel : on passe à sa valeur moyenne
    let far = smoothstep(0.08, 0.35, footprint);
    m = mix(m, coverage * coverage * 0.7, far) * spread;
    let relief = mix(0.5 * (a.b + b.a), 0.5, far);
    return vec2(clamp(m, 0.0, 1.0), relief);
}

// ------------------------------------------------------------------------------------------
// Ombres des nuages
// ------------------------------------------------------------------------------------------

/// Part du soleil qui passe entre les nuages (couche de cumulus à ~1,2 km qui dérive avec
/// l'alizé). Projetée le long de la direction du soleil.
fn cloud_shadow(p: vec3<f32>) -> f32 {
    let l = sun_dir();
    let h = 1200.0 - p.y;
    let q = p.xz + l.xz / max(l.y, 0.15) * h;
    let drift = vec2(-6.0, 2.0) * sim_time();
    let uv = (q - drift) / 700.0;
    let n = fbm2(uv, 4);
    // couverture : seuil du bruit (look.z = 0,5 → ~20 % du sol à l'ombre)
    let thr = 0.53 + (0.5 - ocean.look.z) * 0.3;
    let cover = smoothstep(thr, thr + 0.15, n);
    return 1.0 - ocean.look.y * cover;
}

// ------------------------------------------------------------------------------------------
// Ombres et éclairage des surfaces opaques
// ------------------------------------------------------------------------------------------

fn sun_shadow(p: vec3<f32>, n: vec3<f32>, frag_xy: vec2<f32>) -> f32 {
    if lights.n_directional_lights == 0u {
        return 1.0;
    }
    if (lights.directional_lights[0].flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) == 0u {
        return 1.0;
    }
    let view_z = (view.view_from_world * vec4(p, 1.0)).z;
    return fetch_directional_shadow(0u, vec4(p, 1.0), n, view_z, frag_xy);
}

/// Occlusion ambiante calculée par Bevy (SSAO), 1 si désactivée.
fn screen_ao(frag_xy: vec2<f32>) -> f32 {
    @if(SCREEN_SPACE_AMBIENT_OCCLUSION) {
        let px = vec2<i32>(frag_xy - view.viewport.xy);
        return textureLoad(screen_space_ambient_occlusion_texture, px, 0).r;
    }
    @if(!SCREEN_SPACE_AMBIENT_OCCLUSION) {
        return 1.0;
    }
}

/// Ombres de contact (petits obstacles au ras des surfaces), 1 si désactivées.
/// Inutiles au loin (plus petites que le pixel) : on ne les calcule que de près.
fn contact_shadow(p: vec3<f32>, frag_xy: vec2<f32>, l: vec3<f32>) -> f32 {
    @if(CONTACT_SHADOWS && DEPTH_PREPASS) {
        let steps = contact_shadows_settings.linear_steps;
        let near = distance(p, view.world_position) < 60.0;
        if near && steps > 0u && lights.n_directional_lights > 0u
            && (lights.directional_lights[0].flags & DIRECTIONAL_LIGHT_FLAGS_CONTACT_SHADOWS_ENABLED_BIT) != 0u {
            return calculate_contact_shadow(p, frag_xy, l, steps);
        }
    }
    return 1.0;
}

/// Occlusion "multi-rebond" (Jimenez 2016) : une surface claire est moins assombrie.
fn ao_multibounce(visibility: f32, albedo: vec3<f32>) -> vec3<f32> {
    let a = 2.0404 * albedo - 0.3324;
    let b = -4.7951 * albedo + 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    let x = vec3(visibility);
    return max(x, ((x * a + b) * x + c) * x);
}

/// Albédo du sol autour d'un point émergé (sable clair, ou eau si on est au-dessus de l'eau).
fn ground_albedo_below(p: vec3<f32>) -> f32 {
    return mix(0.07, 0.55, smoothstep(-0.2, 0.4, p.y - sea_level()));
}

/// Éclairement renvoyé par le sol (lumière qui rebondit sur le sable clair), reçu par la
/// moitié de la surface tournée vers le bas.
fn ground_bounce(p: vec3<f32>, n: vec3<f32>, cloud: f32) -> vec3<f32> {
    let down = clamp(0.5 - 0.5 * n.y, 0.0, 1.0);
    if down <= 0.0 {
        return vec3(0.0);
    }
    let depth = sea_level() - p.y;
    if depth > 0.0 {
        // sous l'eau : le fond de sable renvoie la lumière qui l'atteint
        let floor_y = sea_level() - floor_depth();
        let below = max(p.y - floor_y, 0.0);
        let k = water_absorption() + water_backscattering();
        let e_floor = downwelling_irradiance(depth + below);
        return 0.45 * e_floor * exp(-k * below) * down;
    }
    let e_ground = sun_illuminance() * max(sun_dir().y, 0.0) * cloud + sky_irradiance(vec3(0.0, 1.0, 0.0));
    return ground_albedo_below(p) * e_ground * down;
}

/// Luminance d'une surface diffuse (sable, roche, corail) éclairée par le soleil et le ciel,
/// à travers l'eau si elle est immergée (atténuation, caustiques).
fn lit_surface(
    p: vec3<f32>,
    n: vec3<f32>,
    n_geo: vec3<f32>,
    albedo: vec3<f32>,
    ao: f32,
    gloss: f32,
    footprint: f32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
    let depth = sea_level() - p.y;
    // au fond, loin de la caméra, les seules ombres possibles (coraux) sont invisibles
    var shadow = 1.0;
    if depth < 4.0 || distance(p, view.world_position) < 220.0 {
        shadow = sun_shadow(p, n_geo, frag_xy);
    }
    let v = normalize(view.world_position - p);
    let cloud = cloud_shadow(p);
    // occlusion ambiante : géométrique (sommets) × écran (SSAO)
    let ssao = screen_ao(frag_xy);
    let occl = ao_multibounce(min(ao, 1.0) * ssao, albedo);
    var radiance: vec3<f32>;
    if depth > 0.0 {
        let a = water_absorption();
        let b = water_scattering();
        let bb = water_backscattering();
        let lw = sun_dir_in_water();
        shadow *= contact_shadow(p, frag_xy, lw);
        let path = depth / lw.y;
        let t_direct = exp(-(a + b) * path);
        let t_total = exp(-(a + bb) * path);
        let caus = caustics(p, depth, footprint);
        let e_sun = sun_illuminance() * sun_surface_transmission() * cloud
            * (t_direct * caus + max(t_total - t_direct, vec3(0.0)))
            * max(dot(n, lw), 0.0) * shadow;
        let e_sky = sky_irradiance(n) * 0.94 * exp(-(a + bb) * depth * 1.2);
        radiance = albedo / PI * (e_sun + (e_sky + ground_bounce(p, n, cloud)) * occl);
    } else {
        let l = sun_dir();
        shadow *= contact_shadow(p, frag_xy, l);
        let e_sun = sun_illuminance() * max(dot(n, l), 0.0) * shadow * cloud;
        radiance = albedo / PI * (e_sun + (sky_irradiance(n) + ground_bounce(p, n, cloud)) * occl);
        // reflet spéculaire (sable mouillé)
        if gloss > 0.0 {
            let alpha = mix(0.6, 0.12, gloss);
            let h = normalize(l + v);
            let n_dot_l = max(dot(n, l), 0.0);
            let n_dot_v = max(dot(n, v), 1e-3);
            let spec = d_ggx(max(dot(n, h), 0.0), alpha) * vis_smith_ggx(n_dot_v, n_dot_l, alpha)
                * fresnel_dielectric(max(dot(v, h), 0.0), 1.0 / IOR_WATER) * n_dot_l;
            let r = reflect(-v, n);
            let f_env = fresnel_dielectric(n_dot_v, 1.0 / IOR_WATER);
            radiance += gloss * (sun_illuminance() * spec * shadow * cloud
                + f_env * sky_radiance(r, 6.0 + 4.0 * alpha) * occl.g);
        }
    }
    return radiance;
}

/// Applique la brume sous-marine quand la caméra est sous l'eau (tout ce qu'elle voit
/// directement est alors dans l'eau ; ce qui est au-dessus de la surface n'est visible
/// qu'à travers celle-ci, et c'est le shader de l'eau qui s'en charge).
fn apply_media(color: vec3<f32>, p: vec3<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    if camera_underwater() {
        return underwater_fog(color, p, frag_xy);
    }
    return color;
}
