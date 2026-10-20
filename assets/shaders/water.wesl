//! Surface de l'eau : vagues de Gerstner (houle, mer du vent), vagues côtières précalculées
//! (réfraction, déferlement, jet de rive), réflexion du ciel et de la scène (réflexions en
//! espace écran), soleil, réfraction du fond, absorption et diffusion de la lumière dans
//! l'eau, lumière qui traverse les crêtes, écume.
//! Vue d'en dessous : fenêtre de Snell et réflexion totale.

import bevy_pbr::render::mesh_view_bindings::view;
import super::ocean_common::{
    PI, TAU, IOR_WATER, ocean, sim_time, camera_underwater, sea_level, floor_depth, foam_strength,
    sun_dir, sun_illuminance, sun_angular_radius, sky_radiance, sky_irradiance,
    fresnel_dielectric, d_ggx, vis_smith_ggx, water_absorption, water_scattering,
    water_backscattering, sun_dir_in_water, sun_surface_transmission, downwelling_irradiance,
    downwelling_irradiance_c, water_path, underwater_fog, wave_displacement, wave_normal,
    fbm2, noise2, debug_mode, shore_wave, ShoreWave, foam_pattern, cloud_shadow, sun_shadow,
    DRY_DEPTH, debug_flag, water_vertex_world, hash12, hash22,
};

struct WaterVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) rest_xz: vec2<f32>,
}

@vertex
fn vertex(@location(0) position: vec3<f32>) -> WaterVertexOutput {
    // le maillage suit la caméra (horizontalement)
    let wp = water_vertex_world(position);
    var out: WaterVertexOutput;
    out.world_position = wp;
    out.rest_xz = position.xz + view.world_position.xz;
    out.position = view.clip_from_world * vec4(wp, 1.0);
    return out;
}

// ------------------------------------------------------------------------------------------
// Accès à la scène opaque (fond) : couleur et profondeur
// ------------------------------------------------------------------------------------------

fn scene_color(uv: vec2<f32>) -> vec3<f32> {
    @if(VIEW_TRANSMISSION_TEXTURE) {
        return textureSampleLevel(
            bevy_pbr::render::mesh_view_bindings::view_transmission_texture,
            bevy_pbr::render::mesh_view_bindings::view_transmission_sampler,
            uv,
            0.0
        ).rgb;
    }
    @if(!VIEW_TRANSMISSION_TEXTURE) {
        return vec3(0.0);
    }
}

fn scene_depth(uv: vec2<f32>) -> f32 {
    let px = vec2<i32>(clamp(uv, vec2(0.0), vec2(0.9999)) * view.viewport.zw + view.viewport.xy);
    @if(DEPTH_PREPASS) {
        return textureLoad(bevy_pbr::render::mesh_view_bindings::depth_prepass_texture, px, 0);
    }
    @if(!DEPTH_PREPASS) {
        return 0.0;
    }
}

fn uv_depth_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth);
    let w = view.world_from_clip * vec4(ndc, 1.0);
    return w.xyz / w.w;
}

fn world_to_uv(p: vec3<f32>) -> vec2<f32> {
    let c = view.clip_from_world * vec4(p, 1.0);
    let ndc = c.xy / c.w;
    return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

// ------------------------------------------------------------------------------------------
// Réflexions en espace écran : l'île, les rochers, les palmiers se reflètent dans l'eau
// ------------------------------------------------------------------------------------------

/// Marche le long du rayon réfléchi dans l'image de profondeur de la scène opaque.
/// Retourne (couleur, confiance).
fn screen_space_reflection(p: vec3<f32>, r: vec3<f32>, dist: f32, jitter: f32) -> vec4<f32> {
    let cam = view.world_position;
    var t_prev = 0.0;
    var t = (0.3 + 0.015 * dist) * (0.75 + 0.5 * jitter);
    for (var i = 0; i < 18; i++) {
        let q = p + r * t;
        let c = view.clip_from_world * vec4(q, 1.0);
        if c.w <= 0.0 {
            break;
        }
        let ndc = c.xyz / c.w;
        let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        // rien ne dépasse ~30 m (cimes des cocotiers) : au-delà, le rayon ne voit que le ciel
        if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) || q.y > sea_level() + 30.0 {
            break;
        }
        let ds = scene_depth(uv);
        if ds > 0.0 {
            let sp = uv_depth_to_world(uv, ds);
            let ray_d = distance(q, cam);
            let scene_d = distance(sp, cam);
            let thickness = 0.6 + 0.12 * t;
            if ray_d > scene_d && ray_d - scene_d < thickness && sp.y > sea_level() - 0.15 && scene_d < 5000.0 {
                // affinage par dichotomie
                var a = t_prev;
                var b = t;
                for (var k = 0; k < 5; k++) {
                    let m = 0.5 * (a + b);
                    let qm = p + r * m;
                    let uvm = world_to_uv(qm);
                    let spm = uv_depth_to_world(uvm, scene_depth(uvm));
                    if distance(qm, cam) > distance(spm, cam) {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                let uvh = world_to_uv(p + r * b);
                let edge = min(min(uvh.x, 1.0 - uvh.x), min(uvh.y, 1.0 - uvh.y));
                let conf = smoothstep(0.0, 0.07, edge) * (1.0 - smoothstep(150.0, 400.0, b));
                return vec4(scene_color(uvh) / view.exposure, conf);
            }
        }
        t_prev = t;
        t = t * 1.32 + 0.2;
    }
    return vec4(0.0);
}

// ------------------------------------------------------------------------------------------
// Fond de sable "infini" (au-delà du maillage du fond marin)
// ------------------------------------------------------------------------------------------

fn analytic_floor_radiance() -> vec3<f32> {
    let albedo = vec3(0.86, 0.82, 0.73) * 0.62;
    return albedo / PI * downwelling_irradiance(floor_depth());
}

// ------------------------------------------------------------------------------------------
// Vue depuis l'air
// ------------------------------------------------------------------------------------------

/// Scintillement du soleil sur les vaguelettes trop petites pour être résolues : au lieu
/// d'étaler le reflet (lobe large et terne), chaque cellule (≈ 3 pixels, pour que l'anticrénelage temporel ne les éteigne pas) porte une facette
/// de pente aléatoire qui tourne lentement ; seules celles bien orientées renvoient le soleil,
/// très fort. La moyenne reste celle du lobe large, mais concentrée en points éblouissants
/// que le bloom transforme en étoiles. Renvoie le terme D (distribution des normales).
fn sparkle_at(xz: vec2<f32>, n: vec3<f32>, h: vec3<f32>, sigma: f32, cell: f32) -> f32 {
    let g = xz / cell;
    let c = floor(g);
    let r = hash22(c);
    let r2 = hash22(c + vec2(17.3, 41.9));
    // pente gaussienne (Box-Muller) qui tourne avec le temps
    let mag = sigma * sqrt(-2.0 * log(max(r.x, 1e-4)));
    let ang = TAU * r.y + sim_time() * (0.4 + 0.9 * r2.x);
    let s = mag * vec2(cos(ang), sin(ang));
    let nm = normalize(n + vec3(s.x, 0.0, s.y));
    let a_t = 0.025;
    // point rond au centre de la cellule (rayon aléatoire), énergie conservée
    let rad = 0.25 + 0.2 * r2.y;
    let disc = 1.0 - smoothstep(rad * 0.6, rad, length(g - c - 0.5));
    return d_ggx(max(dot(nm, h), 0.0), a_t) * disc / (PI * rad * rad * 0.8);
}

fn sun_sparkle(xz: vec2<f32>, n: vec3<f32>, h: vec3<f32>, sigma: f32, footprint: f32) -> f32 {
    let lvl = log2(max(footprint * 3.5, 0.04));
    let l0 = floor(lvl);
    let t = lvl - l0;
    let a = sparkle_at(xz, n, h, sigma, exp2(l0));
    let b = sparkle_at(xz + 7.7, n, h, sigma, exp2(l0 + 1.0));
    return mix(a, b, t);
}


fn shade_above(
    p: vec3<f32>,
    v: vec3<f32>,
    dist: f32,
    n: vec3<f32>,
    slope_var: f32,
    detail: f32,
    frag_xy: vec2<f32>,
    sw: ShoreWave,
    footprint: f32,
) -> vec3<f32> {
    let n_dot_v = dot(n, v);
    let f = fresnel_dielectric(max(n_dot_v, 0.0), 1.0 / IOR_WATER);
    let alpha = clamp(sqrt(slope_var), 0.006, 1.0);
    let l = sun_dir();
    let cloud = cloud_shadow(p);
    // ombres portées (palmiers, rochers) sur l'eau : seulement près de l'île
    var shadow = 1.0;
    if sw.depth < 3.0 {
        shadow = sun_shadow(p, vec3(0.0, 1.0, 0.0), frag_xy);
    }
    let sun_vis = cloud * shadow;

    // --- réflexion du ciel, et de la scène là où le rayon la rencontre
    var r = reflect(-v, n);
    r.y = abs(r.y);
    let lod = clamp(log2(alpha * 1304.0 + 1.0) - 0.5, 0.0, 10.0);
    var reflected = sky_radiance(r, lod);
    let jitter = fract(52.9829189 * fract(dot(frag_xy, vec2(0.06711056, 0.00583715))));
    let rs = normalize(vec3(r.x, max(r.y, 0.015), r.z));
    var ssr = vec4(0.0);
    if !debug_flag(1u) {
        ssr = screen_space_reflection(p, rs, dist, jitter);
    }
    reflected = mix(reflected, ssr.rgb, ssr.a * (1.0 - smoothstep(0.15, 0.5, alpha)));

    // --- reflet du soleil (scintillement)
    let a_sun = sqrt(alpha * alpha + 0.5 * sun_angular_radius() * sun_angular_radius());
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let d_smooth = d_ggx(max(dot(n, h), 0.0), a_sun);
    let d_spark = sun_sparkle(p.xz, n, h, alpha, footprint);
    let spec = mix(d_smooth, d_spark * max(ocean.look2.z, 1.0), 0.85) * vis_smith_ggx(max(n_dot_v, 1e-3), n_dot_l, a_sun)
        * fresnel_dielectric(max(dot(v, h), 0.0), 1.0 / IOR_WATER) * n_dot_l;
    let sun_glint = sun_illuminance() * spec * sun_vis;

    // --- réfraction : le fond vu à travers l'eau
    let uv0 = (frag_xy - view.viewport.xy) / view.viewport.zw;
    let d0 = scene_depth(uv0);
    let floor0 = uv_depth_to_world(uv0, d0);
    let has_floor = d0 > 0.0 && distance(floor0, view.world_position) < 2.0e4;
    var water_depth = select(p.y + floor_depth(), p.y - floor0.y, has_floor);
    water_depth = max(water_depth, 0.0);
    let t = refract(-v, n, 1.0 / IOR_WATER);
    let t_down = max(-t.y, 0.12);
    let slant = water_depth / t_down;
    var floor_rad: vec3<f32>;
    var path = slant;
    if has_floor {
        // décalage du fond dû à la réfraction (limité pour rester à l'écran)
        let pr = p + t * min(slant, 40.0);
        var uv = world_to_uv(pr);
        let dr = scene_depth(uv);
        let pw = uv_depth_to_world(uv, dr);
        let valid = all(uv > vec2(0.001)) && all(uv < vec2(0.999)) && dr > 0.0
            && pw.y < p.y - 0.01 && distance(pw, view.world_position) > dist;
        if valid {
            path = distance(p, pw);
        } else {
            uv = uv0;
        }
        floor_rad = scene_color(uv) / view.exposure;
    } else {
        floor_rad = analytic_floor_radiance();
    }

    // --- traversée de l'eau (remontée vers la surface)
    let a = water_absorption();
    let b = water_scattering();
    let bb = water_backscattering();
    let c_up = a + b * 0.6;
    let t_up = exp(-c_up * path);
    // lumière rétrodiffusée par la colonne d'eau
    let e_d = downwelling_irradiance_c(0.0, cloud);
    let r_inf = 0.33 * bb / (a + bb);
    let k_d = (a + bb) * 1.1;
    let column = r_inf * e_d / PI * (1.0 - exp(-(k_d + c_up) * path));
    let below = floor_rad * t_up + column;
    // sortie de l'eau : transmission de Fresnel et loi en n²
    let transmitted = below * (1.0 - f) / (IOR_WATER * IOR_WATER);

    var color = f * reflected + transmitted + sun_glint;

    // --- lumière du soleil qui traverse les crêtes (turquoise lumineux à contre-jour)
    if sw.crest > 0.0 {
        let vh = normalize(vec3(v.x, 0.0, v.z) + vec3(1e-4, 0.0, 0.0));
        let lh = normalize(vec3(l.x, 0.0, l.z) + vec3(1e-4, 0.0, 0.0));
        let lobe = 0.25 + 0.75 * pow(clamp(0.5 - 0.5 * dot(vh, lh), 0.0, 1.0), 3.0);
        let thick = 0.7;
        let tint = b / (a + b) * exp(-(a + b) * thick);
        let facing = clamp(1.0 - n_dot_v, 0.0, 1.0);
        let sss = sun_illuminance() * sun_vis * sun_surface_transmission() * tint
            * (0.045 * lobe) * pow(sw.crest, 2.0) * (0.4 + 0.6 * facing);
        color += sss * (1.0 - f);
    }

    // --- écume : brisants, traînées, bord de l'eau, jet de rive
    // liseré d'écume au contact du sable, selon l'énergie des vagues qui arrivent
    let energy = clamp(sw.foam_trail * 2.0 + sw.foam, 0.12, 1.0);
    let edge = smoothstep(0.08, 0.0, water_depth) * 0.6 * energy;
    let trail = smoothstep(0.2, 1.0, sw.foam_trail) * 0.5;
    var coverage = clamp(max(max(sw.foam, trail), edge) * foam_strength(), 0.0, 1.0);
    coverage = max(coverage, sw.sheet * 0.18);
    let fp = foam_pattern(p.xz, coverage, footprint);
    let foam = fp.x;
    if foam > 0.0 {
        let e = sun_illuminance() * max(l.y, 0.0) * sun_vis + sky_irradiance(vec3(0.0, 1.0, 0.0));
        // mousse épaisse plus claire, fine plus grise ; l'écume laisse passer un peu de l'eau
        let foam_rad = vec3(0.86, 0.88, 0.88) / PI * e * (0.7 + 0.35 * fp.y);
        color = mix(color, foam_rad, foam * 0.94);
    }

    // --- brume atmosphérique vers l'horizon
    let haze = 1.0 - exp(-dist / 18000.0);
    let horizon = sky_radiance(normalize(vec3(-v.x, 0.02, -v.z)), 3.0);
    color = mix(color, horizon, haze * 0.6);
    return color;
}

// ------------------------------------------------------------------------------------------
// Vue depuis l'eau (dessous de la surface)
// ------------------------------------------------------------------------------------------

fn shade_below(p: vec3<f32>, v: vec3<f32>, n_up: vec3<f32>, slope_var: f32, frag_xy: vec2<f32>, sw: ShoreWave) -> vec3<f32> {
    let n = -n_up;
    let cos_i = max(dot(n, v), 0.0);
    let f = fresnel_dielectric(cos_i, IOR_WATER);
    let alpha = clamp(sqrt(slope_var), 0.01, 1.0);
    var color = vec3(0.0);
    // fenêtre de Snell : le ciel et le soleil, comprimés dans un cône de 97°
    if f < 1.0 {
        let t = refract(-v, n, IOR_WATER);
        let lod = clamp(log2(alpha * 2.0 * 1304.0 + 1.0), 0.0, 10.0);
        var sky = sky_radiance(t, lod);
        let a_t = clamp(alpha * 1.8 + 0.004, 0.006, 1.0);
        let cos_sun = dot(t, sun_dir());
        sky += sun_illuminance() * d_ggx(max(cos_sun, 0.0), a_t) * 0.25 * cloud_shadow(p);
        color += sky * (1.0 - f) * IOR_WATER * IOR_WATER;
    }
    // réflexion (totale hors de la fenêtre de Snell) : l'eau et le fond
    let r = reflect(-v, n);
    let m = water_path(0.2, r, 400.0, 1.0);
    let to_floor = (floor_depth()) / max(-r.y, 0.05);
    let mf = water_path(0.2, r, to_floor, 1.0);
    let floor_rad = vec3(0.86, 0.82, 0.73) * 0.62 / PI * downwelling_irradiance(floor_depth());
    let reflected = select(m.inscatter, floor_rad * mf.transmittance + mf.inscatter, r.y < -0.05);
    color += f * reflected;
    // l'écume vue d'en dessous : plaques claires et opaques
    let coverage = clamp(max(sw.foam, smoothstep(0.12, 0.9, sw.foam_trail) * 0.6) * foam_strength(), 0.0, 1.0);
    let foam = foam_pattern(p.xz, coverage, 0.02).x;
    if foam > 0.0 {
        let e = downwelling_irradiance(0.05) * 0.6;
        color = mix(color, vec3(0.8) / PI * e, foam * 0.8);
    }
    return underwater_fog(color, p, frag_xy);
}

@fragment
fn fragment(in: WaterVertexOutput) -> @location(0) vec4<f32> {
    let footprint = max(length(dpdx(in.rest_xz)), length(dpdy(in.rest_xz)));
    if debug_mode() == 3u || debug_mode() == 4u {
        return vec4(0.0, 0.1, 0.12, 1.0);
    }
    let p = in.world_position;
    let to_cam = view.world_position - p;
    let dist = length(to_cam);
    let v = to_cam / dist;
    let sw = shore_wave(in.rest_xz);
    let surf = wave_normal(in.rest_xz, footprint, sw);
    var color: vec3<f32>;
    if camera_underwater() {
        color = shade_below(p, v, surf.normal, surf.slope_variance, in.position.xy, sw);
    } else {
        color = shade_above(p, v, dist, surf.normal, surf.slope_variance, surf.detail_height, in.position.xy, sw, footprint);
    }
    return vec4(color * view.exposure, 1.0);
}
