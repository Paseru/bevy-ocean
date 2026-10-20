//! Fond marin et île : sable (rides, sable mouillé par le jet de rive, laisse de mer),
//! récif (crête, pâtés de corail), sol sous la végétation, rochers et patates de corail.
//! Éclairage sous-marin avec caustiques et atténuation dans l'eau.

import bevy_pbr::render::{forward_io::VertexOutput, mesh_view_bindings::view};
import super::ocean_common::{
    PI, sea_level, sim_time, noise2, fbm2, noise3, fbm3, lit_surface, apply_media, debug_mode,
    beach_wetness, foam_lace,
};

struct SeabedParams {
    base_color: vec4<f32>,
    // x : type (0 sable, 1 roche, 2 corail), y : rugosité, z : échelle du motif
    kind: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(10) var<uniform> params: SeabedParams;
// textures photographiées (Poly Haven, CC0) : sable de plage (vue aérienne, motif de 30 m),
// sol de sable mêlé de débris végétaux (motif de 2 m) ; "nrh" = (normale x, normale y,
// rugosité, relief)
@group(constants::MATERIAL_BIND_GROUP) @binding(11) var sand_albedo_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(12) var tex_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(13) var sand_nrh_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(14) var litter_albedo_tex: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(15) var litter_nrh_tex: texture_2d<f32>;

/// Échantillonnage avec dérivées explicites (utilisable dans des branches).
fn tex_grad(t: texture_2d<f32>, q: vec2<f32>, tile: f32, g: vec4<f32>) -> vec4<f32> {
    return textureSampleGrad(t, tex_sampler, q / tile, g.xy / tile, g.zw / tile);
}

/// Normale d'une carte (xy en [0,1]) appliquée à une surface presque horizontale.
fn map_normal(t: vec4<f32>, strength: f32) -> vec2<f32> {
    return (t.xy * 2.0 - 1.0) * strength;
}

/// Rides de sable : rides de houle sous l'eau, rides éoliennes sur le sable sec.
/// Les bruits (lents à calculer) sont évalués une seule fois au centre ; seule la phase des
/// sinusoïdes est réévaluée pour les différences finies. Retourne (hauteur, pente x, pente z).
fn ripples(q: vec2<f32>, depth: f32, footprint: f32) -> vec3<f32> {
    let underwater = smoothstep(-0.3, 0.6, depth);
    let w_big = (1.0 - smoothstep(0.08, 0.3, footprint)) * underwater;
    let w_small = (1.0 - smoothstep(0.012, 0.045, footprint)) * underwater;
    let w_aeo = (1.0 - smoothstep(0.01, 0.04, footprint)) * smoothstep(0.2, 0.6, -depth);
    if w_big + w_small + w_aeo <= 0.0 {
        return vec3(0.0);
    }
    let e = max(footprint * 0.5, 0.01);
    var h = vec3(0.0);
    if w_big + w_small > 0.0 {
        // mégarides et petites rides de houle, déformées par du bruit
        let warp = fbm2(q * 0.22, 3) * 6.0;
        let dir1 = normalize(vec2(0.8, 0.6));
        let k1 = 6.2831 / 0.85;
        let dir2 = normalize(vec2(0.55, 0.83));
        let k2 = 6.2831 / 0.13;
        let w2 = warp * 1.7 + fbm2(q * 2.1, 2) * 3.0;
        let amp2 = 0.006 * (0.6 + 0.4 * noise2(q * 0.7));
        for (var i = 0; i < 3; i++) {
            let o = select(select(vec2(0.0), vec2(e, 0.0), i == 1), vec2(0.0, e), i == 2);
            let ph1 = dot(q + o, dir1) * k1 + warp;
            let big = (sin(ph1) + 0.25 * sin(2.0 * ph1)) * 0.03;
            let small = sin(dot(q + o, dir2) * k2 + w2) * amp2;
            h[i] += big * w_big + small * w_small;
        }
    }
    if w_aeo > 0.0 {
        // rides éoliennes (alizé d'est : crêtes nord-sud)
        let dir3 = normalize(vec2(1.0, 0.25));
        let k3 = 6.2831 / 0.11;
        let w3 = fbm2(q * 0.9, 3) * 5.0;
        let amp3 = 0.0035 * smoothstep(0.3, 0.7, noise2(q * 0.08)) * w_aeo;
        for (var i = 0; i < 3; i++) {
            let o = select(select(vec2(0.0), vec2(e, 0.0), i == 1), vec2(0.0, e), i == 2);
            let ph3 = dot(q + o, dir3) * k3 + w3;
            h[i] += (sin(ph3) + 0.35 * sin(2.0 * ph3 + 1.0)) * amp3;
        }
    }
    return vec3(h.x, (h.y - h.x) / e, (h.z - h.x) / e);
}

struct SandSample {
    normal: vec3<f32>,
    albedo: vec3<f32>,
    gloss: f32,
    ao: f32,
}

fn sand(p: vec3<f32>, n_geo: vec3<f32>, footprint: f32, masks: vec4<f32>, g: vec4<f32>) -> SandSample {
    let depth = sea_level() - p.y;
    let rel = -depth;
    let q = p.xz;
    // normale perturbée par les rides
    let rip = ripples(q, depth, footprint);
    let h0 = rip.x;
    var slope = rip.yz;

    // sable photographié : grande échelle (30 m, vue aérienne) et détail (même photo
    // réduite à 3,7 m : ses rides deviennent des rides éoliennes de ~15 cm)
    let macro_a = tex_grad(sand_albedo_tex, q, 30.0, g);
    let macro_n = tex_grad(sand_nrh_tex, q, 30.0, g);
    let det_a = tex_grad(sand_albedo_tex, q + vec2(11.3, 4.7), 3.7, g);
    let det_n = tex_grad(sand_nrh_tex, q + vec2(11.3, 4.7), 3.7, g);
    let dry = smoothstep(0.15, 0.6, -depth);
    let det_w = (1.0 - smoothstep(0.01, 0.06, footprint)) * (0.35 + 0.65 * dry);
    slope -= map_normal(macro_n, 0.35) + map_normal(det_n, 0.55 * det_w);
    var n = normalize(n_geo + vec3(-slope.x, 0.0, -slope.y));
    // la photo est un sable brun-gris : on n'en garde que les variations de luminance (et un
    // soupçon de teinte), la couleur de base reste celle du sable corallien
    let lum_w = vec3(0.2126, 0.7152, 0.0722);
    let macro_v = mix(vec3(dot(macro_a.rgb, lum_w)), macro_a.rgb, 0.25) / 0.5;
    let det_v = mix(vec3(dot(det_a.rgb, lum_w)), det_a.rgb, 0.25) / 0.5;
    let tex_var = macro_v * mix(vec3(1.0), det_v, 0.6 * det_w);
    let cavity = mix(1.0, 0.8 + 0.4 * det_n.w, det_w * 0.6);

    // couleur : sable corallien blanc, avec les variations de la photo
    var albedo = params.base_color.rgb * mix(vec3(1.0), tex_var, 0.7) * cavity;
    let large = fbm2(q * 0.03, 3);
    let mid = fbm2(q * 0.35, 3);
    albedo *= 0.92 + 0.12 * large + 0.06 * (mid - 0.5);
    albedo = mix(albedo, albedo * vec3(1.03, 0.98, 0.9), smoothstep(0.4, 0.8, large) * 0.5);
    // débris de coquillages et de corail : points plus sombres ou plus clairs
    let fine_ok = 1.0 - smoothstep(0.01, 0.05, footprint);
    let speck = noise2(q * 23.0);
    albedo *= 1.0 - 0.22 * smoothstep(0.86, 0.95, speck) * fine_ok;
    albedo = mix(albedo, vec3(0.95, 0.93, 0.88), smoothstep(0.9, 0.97, noise2(q * 17.0 + 5.0)) * 0.6 * fine_ok);
    // les creux des rides retiennent des débris plus sombres
    albedo *= 1.0 - clamp(-h0 * 6.0, 0.0, 0.12);
    var ao = 1.0;

    // --- récif : débris de corail, pâtés vivants, algues (sous l'eau)
    let reef = masks.r * smoothstep(-0.1, 0.4, depth);
    if reef > 0.001 {
        let v1 = fbm2(q * 0.6, 4);
        let v2 = noise2(q * 4.0);
        // débris de corail clairs, pâtés vivants bruns/verts, quelques taches roses
        var rc = mix(vec3(0.66, 0.60, 0.48), vec3(0.42, 0.40, 0.28), smoothstep(0.45, 0.7, v1));
        rc = mix(rc, vec3(0.50, 0.36, 0.38), smoothstep(0.64, 0.8, fbm2(q * 0.25 + 7.0, 3)) * 0.6);
        rc *= 0.8 + 0.4 * v2;
        let cover = smoothstep(0.3, 0.7, reef + 0.4 * (v1 - 0.5));
        albedo = mix(albedo, rc, cover);
        // relief irrégulier des débris
        let b = vec2(noise2(q * 3.0 + 1.3) - 0.5, noise2(q * 3.0 + 7.1) - 0.5) * 0.6 * cover;
        n = normalize(n + vec3(b.x, 0.0, b.y));
        ao = mix(1.0, 0.7 + 0.3 * v2, cover);
    }

    var gloss = 0.0;
    if depth > 0.0 {
        albedo *= 0.88;
    } else {
        // --- sol sous la végétation : sable mêlé de feuilles et de débris végétaux (photo)
        let veg = masks.g;
        if veg > 0.001 {
            // deux échelles, la seconde tournée, pour casser la répétition du motif
            let q2 = vec2(q.x * 0.6 - q.y * 0.8, q.x * 0.8 + q.y * 0.6) + vec2(3.3, 7.1);
            let g2 = vec4(g.x * 0.6 - g.y * 0.8, g.x * 0.8 + g.y * 0.6, g.z * 0.6 - g.w * 0.8, g.z * 0.8 + g.w * 0.6);
            let mixw = smoothstep(0.3, 0.7, fbm2(q * 0.15, 3));
            let la = mix(tex_grad(litter_albedo_tex, q, 2.0, g), tex_grad(litter_albedo_tex, q2, 3.1, g2), mixw);
            let ln = mix(tex_grad(litter_nrh_tex, q, 2.0, g), tex_grad(litter_nrh_tex, q2, 3.1, g2), mixw);
            let litter = fbm2(q * 0.4, 3);
            let w = veg * smoothstep(0.25, 0.55, litter + 0.45 * veg);
            // photo normalisée (luminance moyenne 0,5) : on la ramène à un sol sombre et brun ;
            // de loin, la répétition du motif se verrait : on passe à sa couleur moyenne
            let far = smoothstep(0.03, 0.15, footprint);
            let la_v = mix(mix(vec3(dot(la.rgb, vec3(0.2126, 0.7152, 0.0722))), la.rgb, 0.6) / 0.5, vec3(1.0), far);
            var soil = la_v * vec3(0.21, 0.19, 0.15);
            // tapis de lianes rampantes (patate de bord de mer) : plaques vertes
            let vines = smoothstep(0.48, 0.62, fbm2(q * 0.18 + 2.0, 4)) * smoothstep(0.3, 0.7, veg);
            let leafy = 0.75 + 0.5 * noise2(q * 9.0) * (1.0 - far);
            soil = mix(soil, vec3(0.06, 0.10, 0.025) * leafy, vines);
            albedo = mix(albedo, soil, w);
            let ns = map_normal(ln, 0.8) * w;
            n = normalize(n + vec3(-ns.x, 0.0, ns.y));
            ao = min(ao, mix(1.0, 0.7 + 0.3 * ln.w, w));
        }
        // --- jet de rive : sable humide, pellicule brillante, dentelle d'écume
        var wet = vec4(0.0, 0.0, 0.0, 0.3);
        if rel < 1.4 {
            wet = beach_wetness(q, rel);
        }
        let film = wet.x;
        let damp = wet.y;
        albedo *= mix(1.0, 0.58, damp);
        gloss = max(film * 0.95, damp * 0.22);
        n = normalize(mix(n, n_geo, film * 0.7));
        if wet.z > 0.0 {
            let lace = foam_lace(q, 0.6, footprint);
            albedo = mix(albedo, vec3(0.9, 0.91, 0.9), lace * wet.z * 0.8);
        }
        // --- laisse de mer : algues séchées et débris au-dessus des plus hauts jets de rive
        let reach = wet.w;
        let band = smoothstep(reach - 0.02, reach + 0.06, rel) * smoothstep(reach + 0.35, reach + 0.12, rel);
        if band > 0.0 {
            let clumps = smoothstep(0.58, 0.72, fbm2(q * 1.3 + 3.0, 3)) * smoothstep(0.3, 0.6, noise2(q * 0.2));
            albedo = mix(albedo, vec3(0.13, 0.10, 0.06), clumps * band);
            ao = min(ao, 1.0 - 0.3 * clumps * band);
        }
    }
    var s: SandSample;
    s.normal = n;
    s.albedo = albedo;
    s.gloss = gloss;
    s.ao = ao;
    return s;
}

/// Normale perturbée par un bruit 3D (relief de la roche ou du corail).
fn bump3(p: vec3<f32>, n: vec3<f32>, scale: f32, strength: f32) -> vec3<f32> {
    let e = 0.15 / scale;
    let h = fbm3(p * scale, 4);
    let hx = fbm3((p + vec3(e, 0.0, 0.0)) * scale, 4);
    let hy = fbm3((p + vec3(0.0, e, 0.0)) * scale, 4);
    let hz = fbm3((p + vec3(0.0, 0.0, e)) * scale, 4);
    let g = vec3(hx - h, hy - h, hz - h) / e;
    let gt = g - n * dot(g, n);
    return normalize(n - gt * strength);
}

/// Crêtes sinueuses (corail cerveau) : bruit "ridged" dans [0, 1], 1 sur les crêtes.
fn ridges(p: vec3<f32>) -> f32 {
    let a = 1.0 - abs(2.0 * fbm3(p, 3) - 1.0);
    return a * a;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let p = in.world_position.xyz;
    let footprint = max(length(dpdx(p)), length(dpdy(p)));
    if debug_mode() == 2u || debug_mode() == 4u {
        return vec4(vec3(0.5, 0.45, 0.4) * max(in.world_normal.y, 0.2) * 0.3, 1.0);
    }
    var n_geo = normalize(in.world_normal);
    let kind = u32(params.kind.x + 0.5);
    var vcol = vec4(1.0, 0.0, 0.5, 1.0);
    @if(VERTEX_COLORS)
    vcol = in.color;

    var n = n_geo;
    var albedo = params.base_color.rgb;
    var gloss = 0.0;
    var ao = 1.0;
    if kind == 0u {
        let g = vec4(dpdx(p.xz), dpdy(p.xz));
        let s = sand(p, n_geo, footprint, vcol, g);
        n = s.normal;
        gloss = s.gloss;
        albedo = s.albedo;
        ao = s.ao;
    } else if kind == 1u {
        // rocher calcaire : gris-beige, algues sombres sous l'eau
        ao = vcol.r;
        n = bump3(p, n_geo, 1.6, 0.35);
        let v = fbm3(p * 0.9, 4);
        let fine = fbm3(p * 6.0, 3);
        albedo = params.base_color.rgb * (0.7 + 0.5 * v) * (0.85 + 0.3 * fine);
        let wet = smoothstep(0.6, -0.2, p.y - sea_level());
        albedo = mix(albedo, albedo * vec3(0.55, 0.62, 0.45), wet * smoothstep(0.45, 0.7, fbm3(p * 2.0, 3)));
        albedo *= mix(1.0, 0.7, wet);
        ao = mix(1.0, ao, 0.8) * (0.75 + 0.25 * v);
    } else {
        // patate de corail : massive, crêtes sinueuses (corail cerveau), teintes variées
        ao = vcol.r;
        let fine_ok = 1.0 - smoothstep(0.02, 0.08, footprint);
        let r = mix(0.5, ridges(p * 3.1), fine_ok);
        n = bump3(p, n_geo, 2.4, 0.45);
        // relief des crêtes (dérivée du motif)
        let e = 0.03;
        let gr = vec3(
            ridges((p + vec3(e, 0.0, 0.0)) * 3.1) - ridges(p * 3.1),
            ridges((p + vec3(0.0, e, 0.0)) * 3.1) - ridges(p * 3.1),
            ridges((p + vec3(0.0, 0.0, e)) * 3.1) - ridges(p * 3.1)
        ) / e;
        let grt = gr - n * dot(gr, n);
        n = normalize(n - grt * 0.06 * fine_ok);
        let tint_n = fbm3(p * 0.15 + vec3(3.0), 2);
        var base = params.base_color.rgb;
        base = mix(base, vec3(0.42, 0.45, 0.30), smoothstep(0.45, 0.65, tint_n));
        base = mix(base, vec3(0.50, 0.33, 0.40), smoothstep(0.62, 0.75, fbm3(p * 0.3 + vec3(9.0), 2)) * 0.8);
        let speck = noise3(p * 9.0);
        albedo = base * (0.72 + 0.35 * r) * (0.9 + 0.2 * speck);
        ao *= 0.7 + 0.3 * r;
    }

    let color = lit_surface(p, n, n_geo, albedo, ao, gloss, footprint, in.position.xy);
    return vec4(apply_media(color, p, in.position.xy) * view.exposure, 1.0);
}
