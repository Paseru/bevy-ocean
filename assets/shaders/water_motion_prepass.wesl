//! Vecteurs de mouvement de la surface de l'eau (pour l'anti-crénelage temporel et le flou
//! de mouvement). L'eau elle-même est transparente et ne passe pas par la passe préalable :
//! ce maillage reproduit sa surface, sans écrire de profondeur ni de normale (la réfraction
//! et l'occlusion ambiante ont besoin de la profondeur du fond, pas de celle de l'eau).

import bevy_pbr::render::mesh_view_bindings::view;
@if(MOTION_VECTOR_PREPASS)
import bevy_pbr::prepass::bindings::previous_view_uniforms;
import super::ocean_common::water_vertex_world;

struct MotionVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>) -> MotionVertexOutput {
    let wp = water_vertex_world(position);
    var out: MotionVertexOutput;
    out.world_position = wp;
    out.position = view.clip_from_world * vec4(wp, 1.0);
    return out;
}

struct MotionFragmentOutput {
    @if(NORMAL_PREPASS)
    @location(0) normal: vec4<f32>,
    @if(MOTION_VECTOR_PREPASS)
    @location(1) motion_vector: vec2<f32>,
}

@fragment
fn fragment(in: MotionVertexOutput) -> MotionFragmentOutput {
    var out: MotionFragmentOutput;
    @if(NORMAL_PREPASS)
    out.normal = vec4(0.5, 1.0, 0.5, 1.0);
    @if(MOTION_VECTOR_PREPASS) {
        // la surface est considérée immobile d'une image à l'autre (seule la caméra bouge) :
        // le mouvement propre des vagues est assez lent pour que l'anti-crénelage le suive
        let c = view.unjittered_clip_from_world * vec4(in.world_position, 1.0);
        let pc = previous_view_uniforms.unjittered_clip_from_world * vec4(in.world_position, 1.0);
        out.motion_vector = (c.xy / c.w - pc.xy / pc.w) * vec2(0.5, -0.5);
    }
    return out;
}
