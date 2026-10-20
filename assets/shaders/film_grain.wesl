//! Grain de film : bruit fin, animé, plus marqué dans les tons moyens.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct FilmGrain {
    params: vec4<f32>,
}
@group(0) @binding(2) var<uniform> grain: FilmGrain;

fn hash(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(screen_texture, texture_sampler, in.uv);
    let size = vec2<f32>(textureDimensions(screen_texture));
    let px = floor(in.uv * size / max(grain.params.z, 1.0));
    let frame = floor(grain.params.y * 24.0);
    // bruit à peu près gaussien (somme de 3 tirages)
    let n = (hash(vec3(px, frame)) + hash(vec3(px + 17.0, frame + 3.0)) + hash(vec3(px + 41.0, frame + 7.0))) / 3.0 - 0.5;
    // en espace perceptif, plus marqué dans les tons moyens
    let g = pow(max(c.rgb, vec3(0.0)), vec3(1.0 / 2.2));
    let lum = dot(g, vec3(0.2126, 0.7152, 0.0722));
    let amount = grain.params.x * (0.35 + 2.6 * lum * (1.0 - lum));
    let out = pow(max(g + n * amount, vec3(0.0)), vec3(2.2));
    return vec4(out, c.a);
}
