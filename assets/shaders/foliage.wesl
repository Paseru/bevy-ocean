//! Plantes : troncs de cocotiers, palmes, noix de coco, buissons, herbe.
//! Éclairage commun de la scène (lit_surface : soleil, ciel, rebond sur le sable, nuages,
//! occlusion) + translucidité des feuilles à contre-jour + reflets cireux des palmes.

import bevy_pbr::render::{forward_io::VertexOutput, mesh_view_bindings::view};
import super::ocean_common::{
    PI, sea_level, noise3, fbm3, lit_surface, apply_media, sun_dir, sun_illuminance,
    cloud_shadow, sun_shadow, d_ggx, vis_smith_ggx, sky_radiance, debug_mode,
};

struct FoliageParams {
    color: vec4<f32>,
    // x : type (0 tronc, 1 feuille, 2 noix, 3 herbe), y : translucidité, z : rugosité,
    // w : amplitude des variations de teinte
    kind: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(10) var<uniform> params: FoliageParams;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let p = in.world_position.xyz;
    let footprint = max(length(dpdx(p)), length(dpdy(p)));
    var n_geo = normalize(in.world_normal);
    if !is_front {
        n_geo = -n_geo;
    }
    let kind = u32(params.kind.x + 0.5);
    var vcol = vec4(1.0, 0.5, 0.0, 1.0);
    @if(VERTEX_COLORS)
    vcol = in.color;
    var uv = vec2(0.0);
    @if(VERTEX_UVS_A)
    uv = in.uv;

    var albedo = params.color.rgb;
    var n = n_geo;
    let ao = vcol.r;
    let tint = vcol.g;
    let var_amp = params.kind.w;
    // variation de teinte d'une plante à l'autre et le long des feuilles
    let big = fbm3(p * 0.08, 2);
    if kind == 0u {
        // tronc : cicatrices des anciennes palmes (anneaux tous les ~9 cm), écorce grise
        // anneaux irréguliers (espacement et épaisseur variables)
        let warp = noise3(vec3(uv.y * 3.0, uv.x * 2.0, p.x * 0.1)) * 2.5 + noise3(p * 5.0) * 0.8;
        let ring = sin(uv.y * 62.0 + warp);
        let groove = smoothstep(0.75, 0.98, ring) * (0.5 + 0.5 * noise3(vec3(uv.y * 8.0, 1.0, 2.0)));
        let streak = fbm3(vec3(uv.x * 9.0, uv.y * 1.5, 0.0) + p * 0.2, 3);
        albedo *= (0.78 + 0.4 * streak) * (1.0 - 0.3 * groove) * (0.85 + 0.3 * big);
        // relief des anneaux : la normale bascule vers le haut ou le bas
        let d_ring = cos(uv.y * 62.0 + warp) * 0.12;
        n = normalize(n_geo + vec3(0.0, d_ring, 0.0) * (1.0 - smoothstep(0.02, 0.06, footprint)));
        // bas du tronc plus clair (sable, sel)
        albedo = mix(albedo, vec3(0.45, 0.42, 0.37), smoothstep(1.2, 0.0, uv.y) * 0.5);
    } else if kind == 1u || kind == 3u {
        // feuilles : vert jaune → vert foncé, pointes parfois sèches
        let yellow = vec3(0.20, 0.21, 0.05);
        let dark = params.color.rgb * vec3(0.75, 0.85, 0.9);
        albedo = mix(params.color.rgb, mix(dark, yellow, smoothstep(0.55, 0.95, tint)), var_amp * 0.8);
        albedo *= 0.85 + 0.3 * big * var_amp;
        if kind == 1u {
            let tip = smoothstep(0.82, 1.0, uv.y) * smoothstep(0.7, 1.0, tint);
            albedo = mix(albedo, vec3(0.28, 0.21, 0.1), tip * 0.8);
            // nervure centrale plus claire
            let rib = 1.0 - smoothstep(0.0, 0.12, abs(uv.x - 0.5));
            albedo = mix(albedo, albedo * 1.35, rib * 0.6);
        }
    } else {
        // noix de coco : vertes ou brunes
        albedo = mix(params.color.rgb, vec3(0.25, 0.16, 0.08), smoothstep(0.6, 0.9, tint));
    }

    var color = lit_surface(p, n, n_geo, albedo, ao, 0.0, footprint, in.position.xy);

    // translucidité : lumière du soleil qui traverse la feuille (vert-jaune lumineux)
    let l = sun_dir();
    let v = normalize(view.world_position - p);
    let trans = params.kind.y;
    if trans > 0.0 {
        let shadow = sun_shadow(p, n_geo, in.position.xy) * cloud_shadow(p);
        let back = max(dot(-n, l), 0.0);
        let through = pow(max(dot(-v, l), 0.0), 3.0) * 0.6 + 0.4;
        let t_col = albedo * vec3(1.25, 1.45, 0.65);
        color += t_col / PI * sun_illuminance() * back * through * trans * shadow * ao;
    }
    // reflet cireux (palmes, feuilles de Scaevola)
    let rough = params.kind.z;
    if rough < 0.8 && p.y > sea_level() {
        let alpha = rough * rough;
        let h = normalize(l + v);
        let n_dot_l = max(dot(n, l), 0.0);
        let n_dot_v = max(dot(n, v), 1e-3);
        let f = 0.04 + 0.96 * pow(1.0 - max(dot(v, h), 0.0), 5.0);
        let spec = d_ggx(max(dot(n, h), 0.0), alpha) * vis_smith_ggx(n_dot_v, n_dot_l, alpha) * f * n_dot_l;
        let shadow = sun_shadow(p, n_geo, in.position.xy) * cloud_shadow(p);
        color += sun_illuminance() * spec * shadow * ao * 0.6;
        let r = reflect(-v, n);
        color += sky_radiance(r, 6.0) * (0.04 + 0.96 * pow(1.0 - n_dot_v, 5.0)) * 0.1 * ao;
    }
    return vec4(apply_media(color, p, in.position.xy) * view.exposure, 1.0);
}
