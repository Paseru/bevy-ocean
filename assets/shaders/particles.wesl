//! Neige marine : points lumineux qui flottent autour de la caméra sous l'eau.

import bevy_pbr::render::mesh_view_bindings::{view, globals};
import super::ocean_common::{
    PI, camera_underwater, sea_level, sim_time, sun_dir_in_water, downwelling_irradiance,
    water_absorption, water_scattering, phase_hg,
};

const BOX: f32 = 18.0;

struct ParticleOut {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) light: vec3<f32>,
}

@vertex
fn vertex(
    @location(0) seed: vec3<f32>,
    @location(2) corner: vec2<f32>,
    @location(5) extra: vec4<f32>,
) -> ParticleOut {
    var out: ParticleOut;
    out.corner = corner;
    out.light = vec3(0.0);
    let cam = view.world_position;
    if !camera_underwater() {
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }
    // dérive lente (courant) et petit mouvement propre
    let t = sim_time();
    let drift = vec3(0.12, -0.015, 0.05) * t
        + vec3(sin(t * 0.7 + extra.z * 40.0), sin(t * 0.5 + extra.z * 17.0), cos(t * 0.6 + extra.z * 29.0)) * 0.04;
    let local = fract((seed * BOX + drift - cam) / BOX + 0.5) - 0.5;
    let center = cam + local * BOX;
    // au-dessus de la surface : rien
    if center.y > sea_level() - 0.05 {
        out.position = vec4(0.0, 0.0, -2.0, 1.0);
        return out;
    }
    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;
    // taille à l'écran : au moins ~1,5 pixel (sinon les points scintillent), la luminosité
    // est réduite d'autant pour garder la même énergie
    let dist0 = distance(center, cam);
    let pixel = 2.0 * dist0 / (view.clip_from_view[1][1] * view.viewport.w);
    let size = max(extra.x, 1.5 * pixel);
    let energy = (extra.x / size) * (extra.x / size);
    let p = center + (right * (corner.x - 0.5) + up * (corner.y - 0.5)) * size * 2.0;
    out.position = view.clip_from_world * vec4(p, 1.0);
    // éclairement : lumière descendante, diffusée vers la caméra (plus forte face au soleil)
    let depth = max(sea_level() - center.y, 0.0);
    let to_cam = cam - center;
    let dist = length(to_cam);
    let v = to_cam / max(dist, 1e-3);
    let lw = sun_dir_in_water();
    let phase = 0.35 + 3.0 * phase_hg(dot(-v, -lw), 0.7);
    let e = downwelling_irradiance(depth);
    let c = water_absorption() + water_scattering() * 0.8;
    // fondu : près (flou, trop gros) et loin (dans la brume, et bord de la boîte)
    let fade = smoothstep(0.25, 0.9, dist) * (1.0 - smoothstep(BOX * 0.32, BOX * 0.5, dist));
    out.light = e * 0.5 / PI * phase * exp(-c * dist) * extra.y * fade * energy;
    return out;
}

@fragment
fn fragment(in: ParticleOut) -> @location(0) vec4<f32> {
    let d = in.corner - 0.5;
    let a = exp(-dot(d, d) * 14.0);
    return vec4(in.light * a * view.exposure, 0.0);
}
