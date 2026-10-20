// The tone mapping (`aces.rs`): ACES 2.0's output transform, SDR Rec.709 at
// 100 nits (the Academy's `Lib.Academy.OutputTransform.ctl`, its forward
// way, written again; what only depends on the display handed over in
// `p`), or Unreal's Filmic one (`film`, as `PostProcessCombineLUTs.usf`
// and `TonemapCommon.ush` have it), as the map's look says; the look
// (Unreal's vignette, gamut expansion and color correction) and the
// grading before. In: the camera's
// scene linear Rec.709; out: the display's linear Rec.709, in [0, 1],
// dithered as Bevy's tone mapping does.
//
// As Unreal's `PostProcessCombineLUTs.usf` does, all of it but the
// vignette is worked out once a frame for a 32³ table (`build`, a compute
// pass, `BUILD`), its inputs the scene's, 0 then a log scale (`LUT_LOWEST`
// to `LUT_HIGHEST` stops), its outputs the display's gamma encoded; each
// pixel then reads it (`fragment`, `APPLY`), between its eight nearest.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct Params {
    rec709_to_ap1: mat3x3<f32>,
    ap1_to_cam: mat3x3<f32>,
    input_to_aab: mat3x3<f32>,
    limit_from_aab: mat3x3<f32>,
    limit_cam_to_rgb: mat3x3<f32>,
    f_l_n: f32,
    cz: f32,
    inv_cz: f32,
    a_w_j: f32,
    inv_a_w_j: f32,
    forward_limit: f32,
    ts_n_r: f32,
    ts_g: f32,
    ts_t_1: f32,
    ts_s_2: f32,
    ts_m_2: f32,
    limit_j_max: f32,
    model_gamma_inv: f32,
    sat: f32,
    sat_thr: f32,
    compr: f32,
    chroma_compress_scale: f32,
    mid_j: f32,
    focus_dist: f32,
    lower_hull_gamma_inv: f32,
    hue_search: vec2<i32>,
    // Hue, cusp J, cusp M, upper hull gamma's inverse.
    table: array<vec4<f32>, 362>,
    reach_m: array<vec4<f32>, 91>,
}

@group(0) @binding(1) var<storage, read> p: Params;

// The grading (`camera::Grading`) and the map's look (`aces::Look`).
struct Grade {
    exposure: f32,
    contrast: f32,
    saturation: f32,
    gamma: f32,
    temperature: f32,
    vignette: f32,
    shadows_max: f32,
    highlights_min: f32,
    highlights_max: f32,
    // 1 for Unreal's Filmic tone mapping, 0 for ACES 2.0; Unreal's
    // `ExpandGamut` and `BlueCorrection`.
    filmic: f32,
    expand_gamut: f32,
    blue_correction: f32,
    // Saturation, contrast, gamma, gain and offset: the whole picture's,
    // then the shadows', the mid tones' and the highlights'.
    correction: array<vec4<f32>, 20>,
}
@group(0) @binding(0) var<uniform> grade: Grade;

#ifdef BUILD
@group(0) @binding(2) var lut_out: texture_storage_3d<rgba16float, write>;
#endif
#ifdef APPLY
@group(0) @binding(2) var picture: texture_2d<f32>;
@group(0) @binding(3) var lut: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;
#endif

// The table's side, and its inputs: its first slice 0, the others from
// 2^LUT_LOWEST to 2^LUT_HIGHEST on a log scale (half a stop a slice).
const LUT_SIZE: f32 = 32.0;
const LUT_LOWEST: f32 = -10.0;
const LUT_HIGHEST: f32 = 6.0;

// Where a scene's color is in the table (slices, 0 to `LUT_SIZE - 1`):
// under the second slice, as far along as it is from 0 to it.
fn lut_slice(color: vec3<f32>) -> vec3<f32> {
    let logs = (log2(max(color, vec3(1e-10))) - LUT_LOWEST) / (LUT_HIGHEST - LUT_LOWEST)
        * (LUT_SIZE - 2.0) + 1.0;
    let near = color / exp2(LUT_LOWEST);
    return clamp(select(logs, near, color < vec3(exp2(LUT_LOWEST))), vec3(0.0), vec3(LUT_SIZE - 1.0));
}

// A slice's scene color, `lut_slice`'s other way.
fn lut_input(slice: vec3<f32>) -> vec3<f32> {
    let color = exp2(LUT_LOWEST + (slice - 1.0) / (LUT_SIZE - 2.0) * (LUT_HIGHEST - LUT_LOWEST));
    return select(color, slice * exp2(LUT_LOWEST), slice < vec3(1.0));
}

// Before the transform, on the scene linear picture: exposure, white
// balance, saturation, contrast around middle gray.
fn graded(rgb: vec3<f32>) -> vec3<f32> {
    var c = max(rgb, vec3(0.0)) * exp2(grade.exposure);
    let t = grade.temperature * 0.2;
    c *= vec3(1.0 + t, 1.0, 1.0 - t);
    let luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3(luma), c, grade.saturation), vec3(0.0));
    return 0.18 * pow(c / 0.18, vec3(grade.contrast));
}

// Unreal's (`ACESCommon.ush`'s, and its products), its rows as columns: `v * M`
// is Unreal's `mul(M, v)`. sRGB is Unreal's working color space, D65
// adapted to AP1's D60 by Bradford.
const SRGB_TO_AP1 = mat3x3<f32>(
    vec3(0.6130974024, 0.3395231461, 0.0473794514),
    vec3(0.0701937225, 0.9163538791, 0.0134523986),
    vec3(0.0206155929, 0.1095697729, 0.8698146341),
);
const AP1_TO_SRGB = mat3x3<f32>(
    vec3(1.7050509926, -0.6217921205, -0.0832588722),
    vec3(-0.1302564175, 1.1408047365, -0.0105483190),
    vec3(-0.0240033568, -0.1289689761, 1.1529723328),
);
const AP1_RGB2Y = vec3(0.2722287168, 0.6740817658, 0.0536895174);

// Unreal's vignette (its cosine fourth law), at that pixel of a picture of
// that size: a circle, the corners as dark whatever the picture's shape.
fn vignette(position: vec2<f32>, size: vec2<f32>) -> f32 {
    if grade.vignette <= 0.0 {
        return 1.0;
    }
    let screen = vec2(2.0 * position.x / size.x - 1.0, 1.0 - 2.0 * position.y / size.y);
    let aspect = size.y / size.x;
    let circle = screen * vec2(1.0, aspect) * sqrt(2.0) / sqrt(1.0 + aspect * aspect);
    let tangent = circle * grade.vignette;
    let cos2 = 1.0 / (dot(tangent, tangent) + 1.0);
    return cos2 * cos2;
}

// Unreal's `ColorCorrect`, in AP1.
fn color_correct(
    color: vec3<f32>,
    saturation: vec4<f32>,
    contrast: vec4<f32>,
    gamma: vec4<f32>,
    gain: vec4<f32>,
    offset: vec4<f32>,
) -> vec3<f32> {
    let luma = dot(color, AP1_RGB2Y);
    var c = max(vec3(0.0), mix(vec3(luma), color, saturation.xyz * saturation.w));
    c = pow(c / 0.18, contrast.xyz * contrast.w) * 0.18;
    c = pow(c, 1.0 / (gamma.xyz * gamma.w));
    return c * (gain.xyz * gain.w) + (offset.xyz + offset.w);
}

// A region's (1 the shadows, 2 the mid tones, 3 the highlights) on top of
// the whole picture's.
fn corrected(color: vec3<f32>, region: u32) -> vec3<f32> {
    let r = region * 5u;
    return color_correct(
        color,
        grade.correction[r] * grade.correction[0],
        grade.correction[r + 1u] * grade.correction[1],
        grade.correction[r + 2u] * grade.correction[2],
        grade.correction[r + 3u] * grade.correction[3],
        grade.correction[r + 4u] + grade.correction[4],
    );
}

// Unreal's `ColorCorrectAll`, in AP1.
fn color_correct_all(color: vec3<f32>) -> vec3<f32> {
    let luma = dot(color, AP1_RGB2Y);
    let shadows = 1.0 - smoothstep(0.0, grade.shadows_max, luma);
    let highlights = smoothstep(grade.highlights_min, grade.highlights_max, luma);
    return corrected(color, 1u) * shadows
        + corrected(color, 2u) * (1.0 - shadows - highlights)
        + corrected(color, 3u) * highlights;
}

// Unreal's gamut expansion (`CombineLUTsCommon`): bright saturated colors
// pushed out, between P3 and AP1, as a wide gamut would show them. In AP1.
const EXPAND_GAMUT = mat3x3<f32>(
    vec3(1.3704123718, -0.3292921877, -0.0636831194),
    vec3(-0.0834334917, 1.0970927480, -0.0108613795),
    vec3(-0.0257933209, -0.0986257988, 1.2036949526),
);

fn expand_gamut(ap1: vec3<f32>) -> vec3<f32> {
    let luma = dot(ap1, AP1_RGB2Y);
    let chroma = ap1 / luma - 1.0;
    let amount = (1.0 - exp2(-4.0 * dot(chroma, chroma)))
        * (1.0 - exp2(-4.0 * grade.expand_gamut * luma * luma));
    return select(mix(ap1, ap1 * EXPAND_GAMUT, amount), ap1, luma <= 0.0);
}

// Unreal's blue correction and its inverse, in AP1 (`ComputeFilmColorNoGamma`).
const BLUE_CORRECT = mat3x3<f32>(
    vec3(0.9386393778, 0.0, 0.0613606221),
    vec3(0.0, 0.8307941329, 0.1692058671),
    vec3(0.0, 0.0, 1.0),
);
const BLUE_CORRECT_INV = mat3x3<f32>(
    vec3(1.0653748754, 0.0000014468, -0.0653710053),
    vec3(-0.0000003455, 1.2036635244, -0.2036677199),
    vec3(0.0000000199, 0.0000000212, 0.9999996001),
);
const AP1_TO_AP0 = mat3x3<f32>(
    vec3(0.6954522413, 0.1406786965, 0.1638690622),
    vec3(0.0447945634, 0.8596711184, 0.0955343182),
    vec3(-0.0055258826, 0.0040252103, 1.0015006722),
);
const AP0_TO_AP1 = mat3x3<f32>(
    vec3(1.4514393161, -0.2365107469, -0.2149285693),
    vec3(-0.0765537733, 1.1762296998, -0.0996759264),
    vec3(0.0083161484, -0.0060324498, 0.9977163014),
);
// Unreal's film curve, its defaults.
const FILM_SLOPE: f32 = 0.88;
const FILM_TOE: f32 = 0.55;
const FILM_SHOULDER: f32 = 0.26;
const FILM_BLACK_CLIP: f32 = 0.0;
const FILM_WHITE_CLIP: f32 = 0.04;

fn rgb_2_saturation(rgb: vec3<f32>) -> f32 {
    let lo = min(min(rgb.r, rgb.g), rgb.b);
    let hi = max(max(rgb.r, rgb.g), rgb.b);
    return (max(hi, 1e-10) - max(lo, 1e-10)) / max(hi, 1e-2);
}

fn rgb_2_yc(rgb: vec3<f32>) -> f32 {
    let chroma = sqrt(max(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) + rgb.r * (rgb.r - rgb.b), 0.0));
    return (rgb.b + rgb.g + rgb.r + 1.75 * chroma) / 3.0;
}

fn sigmoid_shaper(x: f32) -> f32 {
    let t = max(1.0 - abs(0.5 * x), 0.0);
    return 0.5 * (1.0 + sign(x) * (1.0 - t * t));
}

fn glow_fwd(yc: f32, gain: f32, mid: f32) -> f32 {
    if yc <= 2.0 / 3.0 * mid {
        return gain;
    }
    if yc >= 2.0 * mid {
        return 0.0;
    }
    return gain * (mid / yc - 0.5);
}

fn rgb_2_hue(rgb: vec3<f32>) -> f32 {
    if rgb.r == rgb.g && rgb.g == rgb.b {
        return 0.0;
    }
    var hue = degrees(atan2(sqrt(3.0) * (rgb.g - rgb.b), 2.0 * rgb.r - rgb.g - rgb.b));
    if hue < 0.0 {
        hue += 360.0;
    }
    return clamp(hue, 0.0, 360.0);
}

fn log10(x: vec3<f32>) -> vec3<f32> {
    return log2(x) * 0.30102999566;
}

// Unreal's `FilmToneMap`: ACES 1's glow and red modifier, then its own
// curve (toe, straight, shoulder) on the log of the color. AP1 in and out.
fn film_tone_map(ap1: vec3<f32>) -> vec3<f32> {
    var ap0 = ap1 * AP1_TO_AP0;
    let saturation = rgb_2_saturation(ap0);
    let s = sigmoid_shaper((saturation - 0.4) / 0.2);
    ap0 *= 1.0 + glow_fwd(rgb_2_yc(ap0), 0.05 * s, 0.08);
    var centered = rgb_2_hue(ap0);
    if centered > 180.0 {
        centered -= 360.0;
    }
    let weight = smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centered / 135.0));
    ap0.r += weight * weight * saturation * (0.03 - ap0.r) * (1.0 - 0.82);

    var working = max(ap0 * AP0_TO_AP1, vec3(0.0));
    working = mix(vec3(dot(working, AP1_RGB2Y)), working, 0.96);

    let toe_scale = 1.0 + FILM_BLACK_CLIP - FILM_TOE;
    let shoulder_scale = 1.0 + FILM_WHITE_CLIP - FILM_SHOULDER;
    let in_match = 0.18;
    let out_match = 0.18;
    var toe_match: f32;
    if FILM_TOE > 0.8 {
        toe_match = (1.0 - FILM_TOE - out_match) / FILM_SLOPE + log10(vec3(in_match)).x;
    } else {
        let bt = (out_match + FILM_BLACK_CLIP) / toe_scale - 1.0;
        toe_match = log10(vec3(in_match)).x - 0.5 * log((1.0 + bt) / (1.0 - bt)) * (toe_scale / FILM_SLOPE);
    }
    let straight_match = (1.0 - FILM_TOE) / FILM_SLOPE - toe_match;
    let shoulder_match = FILM_SHOULDER / FILM_SLOPE - straight_match;

    let log_color = log10(max(working, vec3(1e-10)));
    let straight = FILM_SLOPE * (log_color + straight_match);
    var toe = -FILM_BLACK_CLIP
        + (2.0 * toe_scale) / (1.0 + exp((-2.0 * FILM_SLOPE / toe_scale) * (log_color - toe_match)));
    var shoulder = (1.0 + FILM_WHITE_CLIP)
        - (2.0 * shoulder_scale) / (1.0 + exp((2.0 * FILM_SLOPE / shoulder_scale) * (log_color - shoulder_match)));
    toe = select(straight, toe, log_color < vec3(toe_match));
    shoulder = select(straight, shoulder, log_color > vec3(shoulder_match));
    var t = clamp((log_color - toe_match) / (shoulder_match - toe_match), vec3(0.0), vec3(1.0));
    if shoulder_match < toe_match {
        t = 1.0 - t;
    }
    t = (3.0 - 2.0 * t) * t * t;
    var tone = mix(toe, shoulder, t);
    tone = mix(vec3(dot(tone, AP1_RGB2Y)), tone, 0.93);
    return max(tone, vec3(0.0));
}

// Unreal's Filmic tone mapping, with its blue correction around the curve:
// AP1 in, linear sRGB out, in [0, 1].
fn film(ap1_in: vec3<f32>) -> vec3<f32> {
    var ap1 = mix(ap1_in, ap1_in * BLUE_CORRECT, grade.blue_correction);
    ap1 = film_tone_map(ap1);
    ap1 = mix(ap1, ap1 * BLUE_CORRECT_INV, grade.blue_correction);
    return clamp(ap1 * AP1_TO_SRGB, vec3(0.0), vec3(1.0));
}

const TABLE_SIZE: i32 = 360;
const TOTAL_TABLE: i32 = 362;
const CAM_NL_OFFSET: f32 = 27.13;
const J_SCALE: f32 = 100.0;
const REF_LUMINANCE: f32 = 100.0;
const SMOOTH_CUSPS: f32 = 0.12;
const CUSP_MID_BLEND: f32 = 1.3;
const FOCUS_GAIN_BLEND: f32 = 0.3;
const COMPRESSION_THRESHOLD: f32 = 0.75;
const PI: f32 = 3.14159265358979;

fn wrap_to_360(hue: f32) -> f32 {
    return hue - 360.0 * floor(hue / 360.0);
}

fn hue_position(hue: f32, size: i32) -> i32 {
    return i32(wrap_to_360(hue) / 360.0 * f32(size));
}

fn reach_m(i: i32) -> f32 {
    return p.reach_m[i / 4][i % 4];
}

fn copysign(magnitude: f32, sign: f32) -> f32 {
    return select(abs(magnitude), -abs(magnitude), sign < 0.0);
}

fn compress(rc: f32) -> f32 {
    let f = pow(rc, 0.42);
    return f / (CAM_NL_OFFSET + f);
}

fn compress_inv(ra: f32) -> f32 {
    let r = min(ra, 0.99);
    return pow(CAM_NL_OFFSET * r / (1.0 - r), 1.0 / 0.42);
}

fn j_to_achromatic(j: f32) -> f32 {
    return pow(max(j, 0.0) / J_SCALE, p.inv_cz);
}

fn j_to_y(j: f32) -> f32 {
    return compress_inv(p.a_w_j * j_to_achromatic(abs(j))) / p.f_l_n;
}

fn y_to_j(y: f32) -> f32 {
    let ra = compress(abs(y) * p.f_l_n);
    return copysign(J_SCALE * pow(ra * p.inv_a_w_j, p.cz), y);
}

// AP1, clamped, to J, M and h.
fn ap1_to_jmh(ap1: vec3<f32>) -> vec3<f32> {
    let m = ap1 * p.ap1_to_cam;
    let compressed = vec3(
        copysign(compress(abs(m.x)), m.x),
        copysign(compress(abs(m.y)), m.y),
        copysign(compress(abs(m.z)), m.z),
    );
    let aab = compressed * p.input_to_aab;
    if aab.x <= 0.0 {
        return vec3(0.0);
    }
    let h = wrap_to_360(degrees(atan2(aab.z, aab.y)));
    return vec3(J_SCALE * pow(aab.x, p.cz), length(aab.yz), h);
}

fn jmh_to_display(jmh: vec3<f32>) -> vec3<f32> {
    let h = radians(jmh.z);
    let aab = vec3(j_to_achromatic(jmh.x), jmh.y * cos(h), jmh.y * sin(h));
    let cones = aab * p.limit_from_aab;
    let rgb_m = vec3(
        copysign(compress_inv(abs(cones.x)), cones.x),
        copysign(compress_inv(abs(cones.y)), cones.y),
        copysign(compress_inv(abs(cones.z)), cones.z),
    );
    return rgb_m * p.limit_cam_to_rgb;
}

fn tonescale(x: f32) -> f32 {
    let f = p.ts_m_2 * pow(max(0.0, x) / (x + p.ts_s_2), p.ts_g);
    return max(0.0, f * f / (f + p.ts_t_1)) * p.ts_n_r;
}

fn reach_m_from_table(h: f32) -> f32 {
    let base = hue_position(h, TABLE_SIZE);
    let t = h - f32(base);
    let lo = base + 1;
    return mix(reach_m(lo), reach_m(lo + 1), t);
}

fn toe(x: f32, limit: f32, k1_in: f32, k2_in: f32) -> f32 {
    if x > limit {
        return x;
    }
    let k2 = max(k2_in, 0.001);
    let k1 = sqrt(k1_in * k1_in + k2 * k2);
    let k3 = (limit + k1) / (limit + k2);
    let minus_b = k3 * x - k1;
    let minus_c = k2 * k3 * x;
    return 0.5 * (minus_b + sqrt(minus_b * minus_b + 4.0 * minus_c));
}

fn chroma_compress_norm(h: f32) -> f32 {
    let hr = radians(h);
    let a = cos(hr);
    let b = sin(hr);
    let cos_hr2 = a * a - b * b;
    let sin_hr2 = 2.0 * a * b;
    let cos_hr3 = 4.0 * a * a * a - 3.0 * a;
    let sin_hr3 = 3.0 * b - 4.0 * b * b * b;
    let m = 11.34072 * a + 16.46899 * cos_hr2 + 7.88380 * cos_hr3 + 14.66441 * b
        - 6.37224 * sin_hr2 + 9.19364 * sin_hr3 + 77.12896;
    return m * p.chroma_compress_scale;
}

fn tonemap_and_compress(jmh: vec3<f32>) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;
    let j_ts = y_to_j(tonescale(j_to_y(j) / REF_LUMINANCE));
    var m_compr = m;
    if m != 0.0 {
        let nj = j_ts / p.limit_j_max;
        let snj = max(0.0, 1.0 - nj);
        let m_norm = chroma_compress_norm(h);
        let limit = pow(nj, p.model_gamma_inv) * reach_m_from_table(h) / m_norm;
        m_compr = m * pow(j_ts / j, p.model_gamma_inv) / m_norm;
        m_compr = limit - toe(limit - m_compr, limit - 0.001, snj * p.sat, sqrt(nj * nj + p.sat_thr));
        m_compr = toe(m_compr, limit, nj * p.compr, snj) * m_norm;
    }
    return vec3(j_ts, m_compr, h);
}

fn solve_j_intersect(j: f32, m: f32, focus_j: f32, max_j: f32, slope_gain: f32) -> f32 {
    let m_scaled = m / slope_gain;
    let a = m_scaled / focus_j;
    if j < focus_j {
        let b = 1.0 - m_scaled;
        let c = -j;
        let root = sqrt(b * b - 4.0 * a * c);
        return -2.0 * c / (b + root);
    }
    let b = -(1.0 + m_scaled + max_j * a);
    let c = max_j * m_scaled + j;
    let root = sqrt(b * b - 4.0 * a * c);
    return -2.0 * c / (b - root);
}

fn compression_slope(intersect_j: f32, focus_j: f32, slope_gain: f32) -> f32 {
    let direction = select(p.limit_j_max - intersect_j, intersect_j, intersect_j < focus_j);
    return direction * (intersect_j - focus_j) / (focus_j * slope_gain);
}

fn smin_scaled(a: f32, b: f32, reference: f32) -> f32 {
    let s = SMOOTH_CUSPS * reference;
    let h = max(s - abs(a - b), 0.0) / s;
    return min(a, b) - h * h * h * s / 6.0;
}

fn line_boundary_m(intersect: f32, slope: f32, inv_gamma: f32, j_max: f32, m_max: f32, reference: f32) -> f32 {
    let shifted = reference * pow(intersect / reference, inv_gamma);
    return shifted * m_max / (j_max - slope * m_max);
}

fn focus_gain(j: f32, threshold: f32) -> f32 {
    var gain = p.limit_j_max * p.focus_dist;
    if j > threshold {
        var adjustment = log2((p.limit_j_max - threshold) / max(0.0001, p.limit_j_max - j)) / log2(10.0);
        adjustment = adjustment * adjustment + 1.0;
        gain *= adjustment;
    }
    return gain;
}

// The table's interval holding the hue: its upper index.
fn hue_interval(h: f32) -> i32 {
    var i = 1 + hue_position(h, TOTAL_TABLE);
    var lo = max(1, i + p.hue_search.x);
    var hi = min(1 + TABLE_SIZE, i + p.hue_search.y);
    while lo + 1 < hi {
        if h > p.table[i].x {
            lo = i;
        } else {
            hi = i;
        }
        i = (lo + hi) / 2;
    }
    return max(1, hi);
}

fn cusp_from_table(h: f32) -> vec2<f32> {
    var lo = 0;
    var hi = 1 + TABLE_SIZE;
    var i = hue_position(h, TABLE_SIZE) + 1;
    while lo + 1 < hi {
        if h > p.table[i].x {
            lo = i;
        } else {
            hi = i;
        }
        i = (lo + hi) / 2;
    }
    let below = p.table[hi - 1];
    let above = p.table[hi];
    let t = (h - below.x) / (above.x - below.x);
    return mix(below.yz, above.yz, t);
}

fn gamut_compress(jmh: vec3<f32>) -> vec3<f32> {
    let j = jmh.x;
    let m = jmh.y;
    let h = jmh.z;
    if j <= 0.0 {
        return vec3(0.0, 0.0, h);
    }
    if m < 0.0 || j > p.limit_j_max {
        return vec3(j, 0.0, h);
    }

    // The hue's.
    let i_hi = hue_interval(h);
    let t = h - p.table[i_hi - 1].x;
    let cusp = cusp_from_table(h);
    let gamma_top_inv = mix(p.table[i_hi - 1].w, p.table[i_hi].w, t);
    let focus_j = mix(cusp.x, p.mid_j, min(1.0, CUSP_MID_BLEND - cusp.x / p.limit_j_max));
    let threshold = mix(cusp.x, p.limit_j_max, FOCUS_GAIN_BLEND);

    let slope_gain = focus_gain(j, threshold);
    let intersect_source = solve_j_intersect(j, m, focus_j, p.limit_j_max, slope_gain);
    let slope = compression_slope(intersect_source, focus_j, slope_gain);
    let intersect_cusp = solve_j_intersect(cusp.x, cusp.y, focus_j, p.limit_j_max, slope_gain);

    let lower = line_boundary_m(intersect_source, slope, p.lower_hull_gamma_inv, cusp.x, cusp.y, intersect_cusp);
    let upper = line_boundary_m(
        p.limit_j_max - intersect_source,
        -slope,
        gamma_top_inv,
        p.limit_j_max - cusp.x,
        cusp.y,
        p.limit_j_max - intersect_cusp,
    );
    let gamut_m = smin_scaled(lower, upper, cusp.y);
    if gamut_m <= 0.0 {
        return vec3(j, 0.0, h);
    }
    let reach_boundary_m = line_boundary_m(
        intersect_source,
        slope,
        p.model_gamma_inv,
        p.limit_j_max,
        reach_m_from_table(h),
        p.limit_j_max,
    );

    // M remapped between the gamut's boundary and the reach's.
    var remapped = m;
    let proportion = max(gamut_m / reach_boundary_m, COMPRESSION_THRESHOLD);
    let start = proportion * gamut_m;
    if m > start && proportion < 1.0 {
        let gamut_offset = gamut_m - start;
        let reach_offset = reach_boundary_m - start;
        let scale = reach_offset / ((reach_offset / gamut_offset) - 1.0);
        let nd = (m - start) / scale;
        remapped = start + scale * nd / (1.0 + nd);
    }
    return vec3(intersect_source + remapped * slope, remapped, h);
}

fn output_transform(rec709: vec3<f32>) -> vec3<f32> {
    let ap1 = clamp(rec709 * p.rec709_to_ap1, vec3(0.0), vec3(p.forward_limit));
    let jmh = ap1_to_jmh(ap1);
    let compressed = gamut_compress(tonemap_and_compress(jmh));
    return clamp(jmh_to_display(compressed), vec3(0.0), vec3(1.0));
}

// Bevy's own, as its tone mapping dithers.
fn screen_space_dither(frag_coord: vec2<f32>) -> vec3<f32> {
    var dither = vec3<f32>(dot(vec2<f32>(171.0, 231.0), frag_coord)).xxx;
    dither = fract(dither.rgb / vec3<f32>(103.0, 71.0, 97.0));
    return (dither - 0.5) / 255.0;
}

// The display's color of a scene's (after its vignette), gamma encoded
// (2.2), not dithered.
fn display(scene_color: vec3<f32>) -> vec3<f32> {
    let scene = graded(scene_color);
    let corrected = color_correct_all(expand_gamut(scene * SRGB_TO_AP1));
    var rgb: vec3<f32>;
    if grade.filmic > 0.5 {
        rgb = film(corrected);
    } else {
        rgb = output_transform(corrected * AP1_TO_SRGB);
    }
    rgb = pow(rgb, vec3(1.0 / grade.gamma));
    return pow(rgb, vec3(1.0 / 2.2));
}

#ifdef BUILD
@compute @workgroup_size(4, 4, 4)
fn build(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= vec3(u32(LUT_SIZE))) {
        return;
    }
    textureStore(lut_out, id, vec4(display(lut_input(vec3<f32>(id))), 1.0));
}
#endif

#ifdef APPLY
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureLoad(picture, vec2<i32>(in.position.xy), 0);
    let size = vec2<f32>(textureDimensions(picture));
    let scene = color.rgb * vignette(in.position.xy, size);
    let at = (lut_slice(scene) + 0.5) / LUT_SIZE;
    let rgb = textureSampleLevel(lut, lut_sampler, at, 0.0).rgb
        + screen_space_dither(in.position.xy);
    return vec4(pow(max(rgb, vec3(0.0)), vec3(2.2)), color.a);
}
#endif
