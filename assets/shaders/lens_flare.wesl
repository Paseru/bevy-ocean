//! Reflets d'objectif : fantômes le long de l'axe soleil-centre, halo, éclat en étoile
//! autour du soleil et voile lumineux. La visibilité du soleil est mesurée directement
//! dans l'image (après le rendu des tons, le disque solaire est blanc saturé ; s'il est caché
//! par une feuille ou un rocher, il n'y a pas de reflet).

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct LensFlare {
    sun: vec4<f32>,
    params: vec4<f32>,
}
@group(0) @binding(2) var<uniform> flare: LensFlare;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

/// Fraction visible du disque solaire (0 = caché).
fn sun_visibility(sun_uv: vec2<f32>, aspect: f32) -> f32 {
    var acc = 0.0;
    let r = 0.0035;
    for (var i = 0; i < 12; i++) {
        let a = f32(i) * 0.5236;
        let rr = select(r, r * 0.45, i % 2 == 0);
        let o = vec2(cos(a) / aspect, sin(a)) * rr;
        let lum = luminance(textureSampleLevel(screen_texture, texture_sampler, sun_uv + o, 0.0).rgb);
        acc += smoothstep(flare.params.z, 0.99, lum);
    }
    return acc / 12.0;
}

/// Disque à bord doux (fantôme), légèrement irisé.
fn ghost(uv: vec2<f32>, center: vec2<f32>, radius: f32, aspect: f32) -> vec3<f32> {
    let d = length((uv - center) * vec2(aspect, 1.0)) / radius;
    let disk = smoothstep(1.0, 0.82, d) * (0.55 + 0.45 * d * d);
    // irisation du bord (dispersion des traitements antireflets)
    let rim = vec3(
        smoothstep(1.02, 0.86, d),
        smoothstep(0.99, 0.83, d),
        smoothstep(0.96, 0.80, d)
    );
    return rim * disk;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(screen_texture, texture_sampler, in.uv);
    let fade = flare.sun.z;
    if fade <= 0.0 {
        return base;
    }
    let aspect = flare.params.x;
    let sun_uv = flare.sun.xy;
    if any(sun_uv < vec2(-0.2)) || any(sun_uv > vec2(1.2)) {
        return base;
    }
    let vis = select(0.0, sun_visibility(clamp(sun_uv, vec2(0.0), vec2(1.0)), aspect), all(sun_uv > vec2(0.0)) && all(sun_uv < vec2(1.0)));
    let k = vis * fade * flare.sun.w;
    if k <= 0.001 {
        return base;
    }
    let uv = in.uv;
    let c = vec2(0.5, 0.5);
    let axis = c - sun_uv;
    var add = vec3(0.0);
    // fantômes (positions le long de l'axe, rayons, teintes)
    add += ghost(uv, sun_uv + axis * 0.55, 0.035, aspect) * vec3(0.30, 0.55, 0.40) * 0.05;
    add += ghost(uv, sun_uv + axis * 0.85, 0.060, aspect) * vec3(0.55, 0.35, 0.25) * 0.03;
    add += ghost(uv, sun_uv + axis * 1.25, 0.022, aspect) * vec3(0.35, 0.45, 0.70) * 0.07;
    add += ghost(uv, sun_uv + axis * 1.55, 0.110, aspect) * vec3(0.25, 0.40, 0.55) * 0.015;
    add += ghost(uv, sun_uv + axis * 1.95, 0.045, aspect) * vec3(0.60, 0.45, 0.25) * 0.03;
    // halo annulaire centré sur le soleil
    let ds = length((uv - sun_uv) * vec2(aspect, 1.0));
    let ring = exp(-pow((ds - 0.32) / 0.025, 2.0));
    add += vec3(0.45, 0.55, 0.65) * ring * 0.025;
    // éclat en étoile (diffraction par les lamelles du diaphragme, 6 branches)
    let dir = (uv - sun_uv) * vec2(aspect, 1.0);
    let ang = atan2(dir.y, dir.x);
    let spikes = pow(abs(cos(ang * 3.0)), 60.0);
    let star = spikes * exp(-ds * 9.0) * 0.35 + exp(-ds * 28.0) * 0.5;
    add += vec3(1.0, 0.95, 0.88) * star;
    // voile : la lumière parasite éclaircit un peu toute l'image
    let veil = 0.03 * (1.0 - 0.6 * length(sun_uv - c));
    add += vec3(veil) * vec3(1.0, 0.98, 0.95);
    // les couleurs sont linéaires (affichage) : on ajoute en restant dans [0, 1]
    return vec4(min(base.rgb + add * k, vec3(1.0)), base.a);
}
