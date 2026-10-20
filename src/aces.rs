//! The tone mapping, in place of Bevy's: the player's camera has none of
//! Bevy's (`Tonemapping::None`, `Aces` on it, `camera.rs`), and a pass of
//! `assets/shaders/aces.wgsl` brings its picture, scene linear Rec.709, to
//! the display's, by ACES 2.0's output transform.
//!
//! It is the Academy's reference (`aces-core`'s
//! `Lib.Academy.OutputTransform.ctl`, written again), as its
//! `Rec709-D65_100nit_in_Rec709-D65_sRGB` does: to ACES (AP0, a Bradford
//! adaptation from D65 to ACES's white), into the Hellwig 2022 color
//! appearance model's JMh, the tone scale on J (Daniele's, 100 nits at the
//! peak), the chroma compressed in the gamut and the gamut compressed to
//! Rec.709, back to RGB, clamped to [0, 1]. What only depends on the
//! display (the model's matrices, the tone scale's numbers, and the tables
//! by hue: the reach's colorfulness, the gamut's cusps and its upper hull's
//! gamma) is worked out here once, as the reference does in its
//! `init_ODTParams`, and handed to the shader in one storage buffer
//! (`Params`).
//!
//! Or, where the map's look says so (`Look::filmic`: UDS's, under the
//! sky), Unreal's own Filmic tone mapping, its default: ACES 1's glow and
//! red modifier, its film curve in AP1 and its blue correction around it
//! (`PostProcessCombineLUTs.usf`, `TonemapCommon.ush`, written again).
//!
//! Before it, the map's `Look` (`world::Map::look`: its Unreal post
//! process volume's): its vignette, its gamut expansion and its color
//! correction in AP1, as Unreal does them; and the
//! debug menu's `Grading`. The sRGB encoding is left to the window's
//! picture, as Bevy's own. It runs where Bevy's would (before the anti
//! aliasing and the upscaler, `Core3dSystems::PostProcess`),
//! and dithers as Bevy's does.
//!
//! As Unreal's `PostProcessCombineLUTs` does, all of it but the vignette is
//! worked out for a table, 32³ (`LUT_SIZE`), each frame (a compute pass,
//! the shader's `build`: the grading and the look may change), and each
//! pixel only reads it (`fragment`): ACES 2.0's transform a pixel cost an
//! integrated GPU 2 ms a frame, the table and its reading a sixth of it
//! (measured by `platform/bench.rs`).

use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::core_pipeline::tonemapping::tonemapping;
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only, texture_2d, texture_3d, texture_storage_3d, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor,
    ComputePipelineDescriptor, Extent3d, FilterMode, FragmentState, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, StorageBuffer,
    StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{RenderApp, RenderStartup};

const SHADER: &str = "shaders/aces.wgsl";
/// The table's side (the shader's `LUT_SIZE`), and its workgroups' (its
/// `build`'s).
const LUT_SIZE: u32 = 32;
const LUT_GROUP: u32 = 4;

pub struct AcesPlugin;

impl Plugin for AcesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<Aces>::default());
        let Some(renderer) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        renderer.add_systems(RenderStartup, init).add_systems(
            Core3d,
            // Where Bevy's tone mapping would be, none on the camera.
            aces.before(tonemapping).in_set(Core3dSystems::PostProcess),
        );
    }
}

/// How the picture is graded before (and, the gamma, after) its tone
/// mapping, as `aces.wgsl` does it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Grading {
    /// Brightness, in stops (EV) added to the exposure.
    pub exposure: f32,
    /// Around middle gray, 1 as it is.
    pub contrast: f32,
    /// 0 gray, 1 as it is.
    pub saturation: f32,
    /// On the display's picture: over 1 brighter mid tones.
    pub gamma: f32,
    /// White balance: under 0 bluer, over 0 warmer.
    pub temperature: f32,
}

impl Grading {
    pub const NEUTRAL: Self = Self {
        exposure: 0.0,
        contrast: 1.0,
        saturation: 1.0,
        gamma: 1.0,
        temperature: 0.0,
    };
}

/// On a camera with no tone mapping of Bevy's (`Tonemapping::None`): its
/// picture brought to the display's by ACES 2.0, the map's `Look` and the
/// debug menu's grading (`camera::Grading`) with it.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct Aces {
    pub grading: Grading,
    pub look: Look,
}

/// What a map's post process volume (Unreal's) makes of its picture, the
/// numbers it overrides (its `bOverride_`s) and Unreal 5.8's defaults for
/// the others (`FPostProcessSettings`'s): its color correction (Unreal's
/// Nuke-like `ColorCorrectAll`, in AP1: the whole picture's, then the
/// shadows', the mid tones' and the highlights' on top, the shadows under
/// `shadows_max` of luminance, the highlights from `highlights_min` to
/// `highlights_max`, blended in between; as Unreal does it under its
/// Standard ACES, which has no film curve), its vignette (Unreal's cosine fourth law: the corners at `cos^4` of an
/// angle whose tangent is `vignette` times theirs; none at 0). `NONE` for a map with none of it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Look {
    pub global: Correction,
    pub shadows: Correction,
    pub midtones: Correction,
    pub highlights: Correction,
    pub shadows_max: f32,
    pub highlights_min: f32,
    pub highlights_max: f32,
    pub vignette: f32,
    /// Unreal's Filmic tone mapping (its default, `film` in the shader) in
    /// place of ACES 2.0, with its `BlueCorrection` around its curve.
    pub filmic: bool,
    pub blue_correction: f32,
    /// Unreal's `ExpandGamut`: bright saturated colors pushed out as a wide
    /// gamut would show them, before the color correction (1 is Unreal's
    /// default; none at 0).
    pub expand_gamut: f32,
}

impl Look {
    /// Unreal's defaults, without its vignette (0.4 there): nothing changed.
    pub const NONE: Self = Self {
        global: Correction::NONE,
        shadows: Correction::NONE,
        midtones: Correction::NONE,
        highlights: Correction::NONE,
        shadows_max: 0.09,
        highlights_min: 0.5,
        highlights_max: 1.0,
        vignette: 0.0,
        filmic: false,
        blue_correction: 0.0,
        expand_gamut: 0.0,
    };
}

/// One of Unreal's color corrections, red, green and blue each and the
/// three at once (W): around mid gray, `saturation` from its luminance,
/// `contrast` a power, then `gamma` the inverse power, `gain` a factor and
/// `offset` added.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Correction {
    pub saturation: Vec4,
    pub contrast: Vec4,
    pub gamma: Vec4,
    pub gain: Vec4,
    pub offset: Vec4,
}

impl Correction {
    pub const NONE: Self = Self {
        saturation: Vec4::ONE,
        contrast: Vec4::ONE,
        gamma: Vec4::ONE,
        gain: Vec4::ONE,
        offset: Vec4::ZERO,
    };

    fn rows(self) -> [Vec4; 5] {
        [
            self.saturation,
            self.contrast,
            self.gamma,
            self.gain,
            self.offset,
        ]
    }
}

/// The grading and the look, as the shader reads them: Unreal's
/// corrections `correction`, five rows (`Correction::rows`) each for the
/// whole picture, the shadows, the mid tones and the highlights.
#[derive(ShaderType, Clone, Copy)]
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
    filmic: f32,
    expand_gamut: f32,
    blue_correction: f32,
    correction: [Vec4; 20],
}

impl Grade {
    fn new(aces: &Aces) -> Self {
        let Aces { grading: g, look } = *aces;
        let regions = [look.global, look.shadows, look.midtones, look.highlights];
        let correction = std::array::from_fn(|i| regions[i / 5].rows()[i % 5]);
        Self {
            exposure: g.exposure,
            contrast: g.contrast,
            saturation: g.saturation,
            gamma: g.gamma,
            temperature: g.temperature,
            vignette: look.vignette,
            shadows_max: look.shadows_max,
            highlights_min: look.highlights_min,
            highlights_max: look.highlights_max,
            filmic: if look.filmic { 1.0 } else { 0.0 },
            expand_gamut: look.expand_gamut,
            blue_correction: look.blue_correction,
            correction,
        }
    }
}

#[derive(Resource)]
struct Pipeline {
    layout: BindGroupLayoutDescriptor,
    id: CachedRenderPipelineId,
    build_layout: BindGroupLayoutDescriptor,
    build_id: CachedComputePipelineId,
    params: StorageBuffer<Params>,
    /// The table, written by `build` and read by `fragment`, and how.
    lut: TextureView,
    linear: Sampler,
}

fn init(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    fullscreen: Res<FullscreenShader>,
    assets: Res<AssetServer>,
) {
    let shader = assets.load(SHADER);
    let layout = BindGroupLayoutDescriptor::new(
        "aces",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, uniform_buffer::<Grade>(false)),
                (
                    2,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (3, texture_3d(TextureSampleType::Float { filterable: true })),
                (4, sampler(SamplerBindingType::Filtering)),
            ),
        ),
    );
    let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("aces".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: vec!["APPLY".into()],
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    let build_layout = BindGroupLayoutDescriptor::new(
        "aces_build",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<Grade>(false),
                storage_buffer_read_only::<Params>(false),
                texture_storage_3d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let build_id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("aces_build".into()),
        layout: vec![build_layout.clone()],
        shader,
        shader_defs: vec!["BUILD".into()],
        entry_point: Some("build".into()),
        ..default()
    });
    let lut = device
        .create_texture(&TextureDescriptor {
            label: Some("aces_lut"),
            size: Extent3d {
                width: LUT_SIZE,
                height: LUT_SIZE,
                depth_or_array_layers: LUT_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default());
    let linear = device.create_sampler(&SamplerDescriptor {
        label: Some("aces_lut"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let mut params = StorageBuffer::from(Odt::new(PEAK_LUMINANCE, REC709).params());
    params.set_label(Some("aces_params"));
    params.write_buffer(&device, &queue);
    commands.insert_resource(Pipeline {
        layout,
        id,
        build_layout,
        build_id,
        params,
        lut,
        linear,
    });
}

fn aces(
    view: ViewQuery<(&ViewTarget, &Aces)>,
    pipeline: Res<Pipeline>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (target, aces) = view.into_inner();
    // A picture with no more than the display's range has nothing to map.
    if target.main_texture_format() != TextureFormat::Rgba16Float {
        return;
    }
    let (Some(render), Some(build)) = (
        cache.get_render_pipeline(pipeline.id),
        cache.get_compute_pipeline(pipeline.build_id),
    ) else {
        return;
    };
    let Some(params) = pipeline.params.binding() else {
        return;
    };
    let mut grade = UniformBuffer::from(Grade::new(aces));
    grade.write_buffer(ctx.render_device(), &queue);
    let Some(grade) = grade.binding() else {
        return;
    };
    let build_group = ctx.render_device().create_bind_group(
        "aces_build",
        &cache.get_bind_group_layout(&pipeline.build_layout),
        &BindGroupEntries::sequential((grade.clone(), params, &pipeline.lut)),
    );
    let post = target.post_process_write();
    let bind_group = ctx.render_device().create_bind_group(
        "aces",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::with_indices((
            (0, grade),
            (2, post.source),
            (3, &pipeline.lut),
            (4, &pipeline.linear),
        )),
    );
    // Timed where Bevy's diagnostics are on (`bench.rs`), nothing otherwise.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    // The table, for this frame's grading and look.
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("aces_build"),
                timestamp_writes: None,
            });
        let span = diagnostics.pass_span(&mut pass, "aces_build");
        pass.set_pipeline(build);
        pass.set_bind_group(0, &build_group, &[]);
        let groups = LUT_SIZE / LUT_GROUP;
        pass.dispatch_workgroups(groups, groups, groups);
        span.end(&mut pass);
    }
    let mut pass = ctx
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("aces"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    let span = diagnostics.pass_span(&mut pass, "aces");
    pass.set_pipeline(render);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
    span.end(&mut pass);
}

/// What the shader is handed: the reference's `ODTParams`, as much of it
/// as a pixel reads. Its matrices are the reference's (a row vector times
/// the matrix: `v * m` in the shader).
#[derive(ShaderType, Clone)]
struct Params {
    /// Linear Rec.709 (D65) to AP1, for the clamp ahead of the model.
    rec709_to_ap1: Mat3,
    /// AP1 to the model's cone responses of ACES's (AP0's) input.
    ap1_to_cam: Mat3,
    /// The cone responses (compressed) to A, a, b.
    input_to_aab: Mat3,
    /// A, a, b to the cone responses, and those to Rec.709, of the
    /// display's (the limiting primaries').
    limit_from_aab: Mat3,
    limit_cam_to_rgb: Mat3,
    f_l_n: f32,
    cz: f32,
    inv_cz: f32,
    a_w_j: f32,
    inv_a_w_j: f32,
    // The tone scale.
    forward_limit: f32,
    ts_n_r: f32,
    ts_g: f32,
    ts_t_1: f32,
    ts_s_2: f32,
    ts_m_2: f32,
    // The compressions'.
    limit_j_max: f32,
    model_gamma_inv: f32,
    sat: f32,
    sat_thr: f32,
    compr: f32,
    chroma_compress_scale: f32,
    mid_j: f32,
    focus_dist: f32,
    lower_hull_gamma_inv: f32,
    hue_search: IVec2,
    /// By hue (`TOTAL_TABLE`, the first and last wrapping it): the hue,
    /// the gamut's cusp's J and M, and its upper hull's gamma's inverse.
    table: [Vec4; TOTAL_TABLE],
    /// The reach's M at the limit's J, a hue a degree.
    reach_m: [Vec4; TOTAL_TABLE.div_ceil(4)],
}

// The reference's constants.
const TABLE_SIZE: usize = 360;
const TOTAL_TABLE: usize = TABLE_SIZE + 2;
const BASE_INDEX: usize = 1;
const HUE_LIMIT: f64 = 360.0;
const CUSP_CORNERS: usize = 6;
const TOTAL_CORNERS: usize = CUSP_CORNERS + 2;
const MAX_SORTED_CORNERS: usize = 2 * CUSP_CORNERS;
const REACH_CUSP_TOLERANCE: f64 = 1e-3;
const DISPLAY_CUSP_TOLERANCE: f64 = 1e-7;
const GAMMA_MINIMUM: f64 = 0.0;
const GAMMA_MAXIMUM: f64 = 5.0;
const GAMMA_SEARCH_STEP: f64 = 0.4;
const GAMMA_ACCURACY: f64 = 1e-5;
const REF_LUMINANCE: f64 = 100.0;
const L_A: f64 = 100.0;
const Y_B: f64 = 20.0;
const SURROUND: [f64; 3] = [0.9, 0.59, 0.9];
const J_SCALE: f64 = 100.0;
const CAM_NL_OFFSET: f64 = 0.2713 * 100.0;
const CAM_NL_SCALE: f64 = 4.0 * 100.0;
const CHROMA_COMPRESS: f64 = 2.4;
const CHROMA_COMPRESS_FACT: f64 = 3.3;
const CHROMA_EXPAND: f64 = 1.3;
const CHROMA_EXPAND_FACT: f64 = 0.69;
const CHROMA_EXPAND_THR: f64 = 0.5;
const SMOOTH_CUSPS: f64 = 0.12;
const SMOOTH_M: f64 = 0.27;
const CUSP_MID_BLEND: f64 = 1.3;
const FOCUS_GAIN_BLEND: f64 = 0.3;
const FOCUS_DISTANCE: f64 = 1.35;
const FOCUS_DISTANCE_SCALING: f64 = 1.75;
const TEST_POSITIONS: [f64; 5] = [0.01, 0.1, 0.5, 0.8, 0.99];

fn model_gamma() -> f64 {
    SURROUND[1] * (1.48 + (Y_B / REF_LUMINANCE).sqrt())
}

/// The display's: SDR, 100 nits at its peak.
const PEAK_LUMINANCE: f64 = 100.0;

/// Primaries and white, as `x, y`: red, green, blue, white.
type Chromaticities = [[f64; 2]; 4];

const REC709: Chromaticities = [[0.64, 0.33], [0.3, 0.6], [0.15, 0.06], [0.3127, 0.329]];
const AP0: Chromaticities = [
    [0.7347, 0.2653],
    [0.0, 1.0],
    [0.0001, -0.077],
    [0.32168, 0.33767],
];
const AP1: Chromaticities = [
    [0.713, 0.293],
    [0.165, 0.83],
    [0.128, 0.044],
    [0.32168, 0.33767],
];
const CAM16: Chromaticities = [
    [0.8336, 0.1735],
    [2.3854, -1.4659],
    [0.087, -0.125],
    [0.333, 0.333],
];
const BRADFORD: M3 = [
    [0.8951, -0.7502, 0.0389],
    [0.2664, 1.7135, -0.0685],
    [-0.1614, 0.0367, 1.0296],
];

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A row vector times a matrix, as the reference's `mult_f3_f33`.
fn mul(v: V3, m: &M3) -> V3 {
    std::array::from_fn(|j| v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j])
}

fn mul_m(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn scale(s: f64, v: V3) -> V3 {
    v.map(|x| s * x)
}

fn invert(m: &M3) -> M3 {
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    std::array::from_fn(|i| std::array::from_fn(|j| c(j, i) / det))
}

/// The reference's `RGBtoXYZ_f33`, for Y = 1.
fn rgb_to_xyz(c: &Chromaticities) -> M3 {
    let [r, g, b, w] = *c;
    let y = 1.0;
    let x = w[0] * y / w[1];
    let z = (1.0 - w[0] - w[1]) * y / w[1];
    let d = r[0] * (b[1] - g[1]) + b[0] * (g[1] - r[1]) + g[0] * (r[1] - b[1]);
    let sr = (x * (b[1] - g[1]) - g[0] * (y * (b[1] - 1.0) + b[1] * (x + z))
        + b[0] * (y * (g[1] - 1.0) + g[1] * (x + z)))
        / d;
    let sg = (x * (r[1] - b[1]) + r[0] * (y * (b[1] - 1.0) + b[1] * (x + z))
        - b[0] * (y * (r[1] - 1.0) + r[1] * (x + z)))
        / d;
    let sb = (x * (g[1] - r[1]) - r[0] * (y * (g[1] - 1.0) + g[1] * (x + z))
        + g[0] * (y * (r[1] - 1.0) + r[1] * (x + z)))
        / d;
    [
        [sr * r[0], sr * r[1], sr * (1.0 - r[0] - r[1])],
        [sg * g[0], sg * g[1], sg * (1.0 - g[0] - g[1])],
        [sb * b[0], sb * b[1], sb * (1.0 - b[0] - b[1])],
    ]
}

fn xyz_to_rgb(c: &Chromaticities) -> M3 {
    invert(&rgb_to_xyz(c))
}

/// From a white to another (Bradford's, the reference's
/// `calculate_cat_matrix`).
fn adaptation(from: [f64; 2], to: [f64; 2]) -> M3 {
    let xyz = |w: [f64; 2]| [w[0] / w[1], 1.0, (1.0 - w[0] - w[1]) / w[1]];
    let source = mul(xyz(from), &BRADFORD);
    let destination = mul(xyz(to), &BRADFORD);
    let mut von_kries = IDENTITY;
    for i in 0..3 {
        von_kries[i][i] = destination[i] / source[i];
    }
    mul_m(&BRADFORD, &mul_m(&von_kries, &invert(&BRADFORD)))
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

fn wrap_to_360(hue: f64) -> f64 {
    hue.rem_euclid(360.0)
}

fn hue_position(hue: f64, size: usize) -> usize {
    (wrap_to_360(hue) / HUE_LIMIT * size as f64) as usize
}

fn compress(rc: f64) -> f64 {
    let f = rc.powf(0.42);
    f / (CAM_NL_OFFSET + f)
}

fn compress_inv(ra: f64) -> f64 {
    let ra = ra.min(0.99);
    (CAM_NL_OFFSET * ra / (1.0 - ra)).powf(1.0 / 0.42)
}

fn signed(f: fn(f64) -> f64, v: f64) -> f64 {
    f(v.abs()).copysign(v)
}

/// The color appearance model, for some primaries (`JMhParams`).
#[derive(Clone)]
struct Cam {
    rgb_to_cam: M3,
    cam_to_rgb: M3,
    to_aab: M3,
    from_aab: M3,
    f_l_n: f64,
    cz: f64,
    inv_cz: f64,
    a_w_j: f64,
    inv_a_w_j: f64,
}

impl Cam {
    fn new(primaries: &Chromaticities) -> Self {
        let matrix_16 = xyz_to_rgb(&CAM16);
        let base_to_aab: M3 = [
            [2.0, 1.0, 1.0 / 9.0],
            [1.0, -12.0 / 11.0, 1.0 / 9.0],
            [1.0 / 20.0, 1.0 / 11.0, -2.0 / 9.0],
        ];
        let rgb_to_xyz = rgb_to_xyz(primaries);
        let xyz_w = mul([REF_LUMINANCE; 3], &rgb_to_xyz);
        let y_w = xyz_w[1];
        let rgb_w = mul(xyz_w, &matrix_16);
        let k = 1.0 / (5.0 * L_A + 1.0);
        let k4 = k * k * k * k;
        let f_l = 0.2 * k4 * (5.0 * L_A) + 0.1 * (1.0 - k4).powi(2) * (5.0 * L_A).powf(1.0 / 3.0);
        let f_l_n = f_l / REF_LUMINANCE;
        let cz = model_gamma();
        let d_rgb: V3 = std::array::from_fn(|i| f_l_n * y_w / rgb_w[i]);
        let rgb_aw: V3 = std::array::from_fn(|i| compress(d_rgb[i] * rgb_w[i]));
        let to_aab = base_to_aab.map(|row| row.map(|x| x * CAM_NL_SCALE));
        let a_w = to_aab[0][0] * rgb_aw[0] + to_aab[1][0] * rgb_aw[1] + to_aab[2][0] * rgb_aw[2];
        let a_w_j = compress(f_l);
        let mut rgb_to_cam = mul_m(
            &mul_m(&rgb_to_xyz, &matrix_16),
            &IDENTITY.map(|row| row.map(|x| x * REF_LUMINANCE)),
        );
        for row in &mut rgb_to_cam {
            for (j, x) in row.iter_mut().enumerate() {
                *x *= d_rgb[j];
            }
        }
        let to_aab = to_aab.map(|row| {
            [
                row[0] / a_w,
                row[1] * 43.0 * SURROUND[2],
                row[2] * 43.0 * SURROUND[2],
            ]
        });
        Self {
            cam_to_rgb: invert(&rgb_to_cam),
            rgb_to_cam,
            from_aab: invert(&to_aab),
            to_aab,
            f_l_n,
            cz,
            inv_cz: 1.0 / cz,
            a_w_j,
            inv_a_w_j: 1.0 / a_w_j,
        }
    }

    fn j_to_achromatic(&self, j: f64) -> f64 {
        (j / J_SCALE).powf(self.inv_cz)
    }

    fn y_to_j(&self, y: f64) -> f64 {
        let ra = compress(y.abs() * self.f_l_n);
        (J_SCALE * (ra * self.inv_a_w_j).powf(self.cz)).copysign(y)
    }

    fn rgb_to_aab(&self, rgb: V3) -> V3 {
        let m = mul(rgb, &self.rgb_to_cam);
        mul(m.map(|v| signed(compress, v)), &self.to_aab)
    }

    fn rgb_to_jmh(&self, rgb: V3) -> V3 {
        let [a, aa, bb] = self.rgb_to_aab(rgb);
        if a <= 0.0 {
            return [0.0; 3];
        }
        [
            J_SCALE * a.powf(self.cz),
            (aa * aa + bb * bb).sqrt(),
            wrap_to_360(bb.atan2(aa).to_degrees()),
        ]
    }

    fn jmh_to_rgb(&self, [j, m, h]: V3) -> V3 {
        let h = h.to_radians();
        let aab = [self.j_to_achromatic(j), m * h.cos(), m * h.sin()];
        let cones = mul(aab, &self.from_aab).map(|v| signed(compress_inv, v));
        mul(cones, &self.cam_to_rgb)
    }
}

/// The tone scale's numbers (`TSParams`).
struct ToneScale {
    n_r: f64,
    g: f64,
    t_1: f64,
    c_t: f64,
    s_2: f64,
    m_2: f64,
    forward_limit: f64,
    log_peak: f64,
}

impl ToneScale {
    fn new(peak: f64) -> Self {
        let n = peak;
        let n_r = 100.0;
        let g = 1.15;
        let c = 0.18;
        let c_d = 10.013;
        let w_g = 0.14;
        let t_1 = 0.04;
        let r_hit_min = 128.0;
        let r_hit_max = 896.0;
        let r_hit =
            r_hit_min + (r_hit_max - r_hit_min) * ((n / n_r).ln() / (10000.0f64 / 100.0).ln());
        let m_0 = n / n_r;
        let m_1 = 0.5 * (m_0 + (m_0 * (m_0 + 4.0 * t_1)).sqrt());
        let w_i = (n / 100.0).log2();
        let c_t = c_d / n_r * (1.0 + w_i * w_g);
        let u = ((r_hit / m_1) / ((r_hit / m_1) + 1.0)).powf(g);
        let m = m_1 / u;
        let g_ip = 0.5 * (c_t + (c_t * (c_t + 4.0 * t_1)).sqrt());
        let g_ipp2 = -(m_1 * (g_ip / m).powf(1.0 / g)) / ((g_ip / m).powf(1.0 / g) - 1.0);
        let w_2 = c / g_ipp2;
        let s_2 = w_2 * m_1;
        let u_2 = ((r_hit / m_1) / ((r_hit / m_1) + w_2)).powf(g);
        let m_2 = m_1 / u_2;
        Self {
            n_r,
            g,
            t_1,
            c_t,
            s_2,
            m_2,
            forward_limit: 8.0 * r_hit,
            log_peak: (n / n_r).log10(),
        }
    }
}

fn compute_focus_j(cusp_j: f64, mid_j: f64, limit_j_max: f64) -> f64 {
    lerp(
        cusp_j,
        mid_j,
        (CUSP_MID_BLEND - cusp_j / limit_j_max).min(1.0),
    )
}

fn get_focus_gain(j: f64, threshold: f64, limit_j_max: f64, focus_dist: f64) -> f64 {
    let mut gain = limit_j_max * focus_dist;
    if j > threshold {
        let adjustment = ((limit_j_max - threshold) / (limit_j_max - j).max(0.0001)).log10();
        gain *= adjustment * adjustment + 1.0;
    }
    gain
}

fn solve_j_intersect(j: f64, m: f64, focus_j: f64, max_j: f64, slope_gain: f64) -> f64 {
    let m_scaled = m / slope_gain;
    let a = m_scaled / focus_j;
    if j < focus_j {
        let b = 1.0 - m_scaled;
        let c = -j;
        let root = (b * b - 4.0 * a * c).sqrt();
        -2.0 * c / (b + root)
    } else {
        let b = -(1.0 + m_scaled + max_j * a);
        let c = max_j * m_scaled + j;
        let root = (b * b - 4.0 * a * c).sqrt();
        -2.0 * c / (b - root)
    }
}

fn compression_slope(intersect_j: f64, focus_j: f64, limit_j_max: f64, slope_gain: f64) -> f64 {
    let direction = if intersect_j < focus_j {
        intersect_j
    } else {
        limit_j_max - intersect_j
    };
    direction * (intersect_j - focus_j) / (focus_j * slope_gain)
}

fn smin_scaled(a: f64, b: f64, reference: f64) -> f64 {
    let s = SMOOTH_CUSPS * reference;
    let h = (s - (a - b).abs()).max(0.0) / s;
    a.min(b) - h * h * h * s / 6.0
}

fn line_boundary_m(
    intersect: f64,
    slope: f64,
    inv_gamma: f64,
    j_max: f64,
    m_max: f64,
    reference: f64,
) -> f64 {
    let shifted = reference * (intersect / reference).powf(inv_gamma);
    shifted * m_max / (j_max - slope * m_max)
}

fn gamut_boundary_m(
    cusp: [f64; 2],
    j_max: f64,
    gamma_top_inv: f64,
    gamma_bottom_inv: f64,
    intersect_source: f64,
    slope: f64,
    intersect_cusp: f64,
) -> f64 {
    let lower = line_boundary_m(
        intersect_source,
        slope,
        gamma_bottom_inv,
        cusp[0],
        cusp[1],
        intersect_cusp,
    );
    let upper = line_boundary_m(
        j_max - intersect_source,
        -slope,
        gamma_top_inv,
        j_max - cusp[0],
        cusp[1],
        j_max - intersect_cusp,
    );
    smin_scaled(lower, upper, cusp[1])
}

/// The unit cube's corners, red, yellow, green, cyan, blue, magenta.
fn cube_corner(corner: usize) -> V3 {
    let on = |offset: usize| {
        if (corner + offset) % CUSP_CORNERS < 3 {
            1.0
        } else {
            0.0
        }
    };
    [on(1), on(5), on(3)]
}

/// The output transform's parameters for a display (`ODTParams`).
struct Odt {
    peak: f64,
    input: Cam,
    reach: Cam,
    limit: Cam,
    ts: ToneScale,
    limit_j_max: f64,
    model_gamma_inv: f64,
    reach_m: [f64; TOTAL_TABLE],
    sat: f64,
    sat_thr: f64,
    compr: f64,
    chroma_compress_scale: f64,
    mid_j: f64,
    focus_dist: f64,
    lower_hull_gamma_inv: f64,
    /// Hue, cusp J, cusp M.
    cusps: [V3; TOTAL_TABLE],
    upper_hull_gamma: [f64; TOTAL_TABLE],
    hue_search: [i32; 2],
}

impl Odt {
    fn new(peak: f64, limiting: Chromaticities) -> Self {
        let input = Cam::new(&AP0);
        let reach = Cam::new(&AP1);
        let limit = Cam::new(&limiting);
        let ts = ToneScale::new(peak);
        let limit_j_max = input.y_to_j(peak);
        let mid_j = input.y_to_j(ts.c_t * REF_LUMINANCE);
        let mut odt = Self {
            peak,
            reach_m: reach_m_table(&reach, limit_j_max),
            sat: (CHROMA_EXPAND - CHROMA_EXPAND * CHROMA_EXPAND_FACT * ts.log_peak).max(0.2),
            sat_thr: CHROMA_EXPAND_THR / peak,
            compr: CHROMA_COMPRESS + CHROMA_COMPRESS * CHROMA_COMPRESS_FACT * ts.log_peak,
            chroma_compress_scale: (0.03379 * peak).powf(0.30596) - 0.45135,
            mid_j,
            focus_dist: FOCUS_DISTANCE + FOCUS_DISTANCE * FOCUS_DISTANCE_SCALING * ts.log_peak,
            lower_hull_gamma_inv: 1.0 / (1.14 + 0.07 * ts.log_peak),
            input,
            reach,
            limit,
            ts,
            limit_j_max,
            model_gamma_inv: 1.0 / model_gamma(),
            cusps: [[0.0; 3]; TOTAL_TABLE],
            upper_hull_gamma: [0.0; TOTAL_TABLE],
            hue_search: [0, 1],
        };
        odt.cusps = odt.gamut_table();
        odt.upper_hull_gamma = odt.upper_hull_gamma_table();
        for i in BASE_INDEX..BASE_INDEX + TABLE_SIZE {
            let delta = i as i32 - hue_position(odt.cusps[i][0], TOTAL_TABLE) as i32;
            odt.hue_search[0] = odt.hue_search[0].min(delta);
            odt.hue_search[1] = odt.hue_search[1].max(delta + 1);
        }
        odt
    }

    /// The display's corners, lowest hue at 1, wrapped at 0 and 7 (RGB,
    /// JMh).
    fn limiting_corners(&self) -> ([V3; TOTAL_CORNERS], [V3; TOTAL_CORNERS]) {
        let rgb: [V3; CUSP_CORNERS] =
            std::array::from_fn(|i| scale(self.peak / REF_LUMINANCE, cube_corner(i)));
        let jmh = rgb.map(|rgb| self.limit.rgb_to_jmh(rgb));
        let lowest =
            (0..CUSP_CORNERS).fold(0, |min, i| if jmh[i][2] < jmh[min][2] { i } else { min });
        let mut rgb_corners = [[0.0; 3]; TOTAL_CORNERS];
        let mut jmh_corners = [[0.0; 3]; TOTAL_CORNERS];
        for i in 0..CUSP_CORNERS {
            rgb_corners[i + 1] = rgb[(i + lowest) % CUSP_CORNERS];
            jmh_corners[i + 1] = jmh[(i + lowest) % CUSP_CORNERS];
        }
        rgb_corners[0] = rgb_corners[CUSP_CORNERS];
        rgb_corners[CUSP_CORNERS + 1] = rgb_corners[1];
        jmh_corners[0] = jmh_corners[CUSP_CORNERS];
        jmh_corners[CUSP_CORNERS + 1] = jmh_corners[1];
        jmh_corners[0][2] -= HUE_LIMIT;
        jmh_corners[CUSP_CORNERS + 1][2] += HUE_LIMIT;
        (rgb_corners, jmh_corners)
    }

    /// The reach's corners at the limit's J (`find_reach_corners_table`).
    fn reach_corners(&self) -> [V3; TOTAL_CORNERS] {
        let limit_a = self.reach.j_to_achromatic(self.limit_j_max);
        let jmh: [V3; CUSP_CORNERS] = std::array::from_fn(|i| {
            let corner = cube_corner(i);
            let (mut lower, mut upper) = (0.0, self.ts.forward_limit);
            while upper - lower > REACH_CUSP_TOLERANCE {
                let test = (lower + upper) / 2.0;
                if self.reach.rgb_to_aab(scale(test, corner))[0] < limit_a {
                    lower = test;
                } else {
                    upper = test;
                }
            }
            self.reach.rgb_to_jmh(scale(upper, corner))
        });
        let lowest =
            (0..CUSP_CORNERS).fold(0, |min, i| if jmh[i][2] < jmh[min][2] { i } else { min });
        let mut corners = [[0.0; 3]; TOTAL_CORNERS];
        for i in 0..CUSP_CORNERS {
            corners[i + 1] = jmh[(i + lowest) % CUSP_CORNERS];
        }
        corners[0] = corners[CUSP_CORNERS];
        corners[CUSP_CORNERS + 1] = corners[1];
        corners[0][2] -= HUE_LIMIT;
        corners[CUSP_CORNERS + 1][2] += HUE_LIMIT;
        corners
    }

    /// The display's cusps, by hues as even as can be that fall on the
    /// corners of both gamuts (`make_uniform_hue_gamut_table`).
    fn gamut_table(&self) -> [V3; TOTAL_TABLE] {
        let reach = self.reach_corners();
        let (rgb_corners, jmh_corners) = self.limiting_corners();
        let hues = hue_table(sorted_hues(&reach, &jmh_corners));
        let mut table = [[0.0; 3]; TOTAL_TABLE];
        for i in BASE_INDEX..TOTAL_TABLE {
            let [j, m] = self.display_cusp(hues[i], &rgb_corners, &jmh_corners);
            table[i] = [hues[i], j, m * (1.0 + SMOOTH_M * SMOOTH_CUSPS)];
        }
        table[0] = [hues[0], table[TABLE_SIZE][1], table[TABLE_SIZE][2]];
        table[BASE_INDEX + TABLE_SIZE] = [
            hues[BASE_INDEX + TABLE_SIZE],
            table[BASE_INDEX][1],
            table[BASE_INDEX][2],
        ];
        table
    }

    fn display_cusp(
        &self,
        hue: f64,
        rgb: &[V3; TOTAL_CORNERS],
        jmh: &[V3; TOTAL_CORNERS],
    ) -> [f64; 2] {
        let upper = (1..TOTAL_CORNERS).find(|&i| jmh[i][2] > hue).unwrap_or(1);
        let lower = upper - 1;
        if jmh[lower][2] == hue {
            return [jmh[lower][0], jmh[lower][1]];
        }
        let at = |t: f64| -> V3 {
            let sample = std::array::from_fn(|k| lerp(rgb[lower][k], rgb[upper][k], t));
            self.limit.rgb_to_jmh(sample)
        };
        let (mut lower_t, mut upper_t) = (0.0, 1.0);
        while upper_t - lower_t > DISPLAY_CUSP_TOLERANCE {
            let t = (lower_t + upper_t) / 2.0;
            let h = at(t)[2];
            if h < jmh[lower][2] {
                upper_t = t;
            } else if h >= jmh[upper][2] {
                lower_t = t;
            } else if h > hue {
                upper_t = t;
            } else {
                lower_t = t;
            }
        }
        let found = at((lower_t + upper_t) / 2.0);
        [found[0], found[1]]
    }

    /// The upper hull's gamma's inverse by hue
    /// (`make_upper_hull_gamma_table`).
    #[allow(clippy::needless_range_loop)]
    fn upper_hull_gamma_table(&self) -> [f64; TOTAL_TABLE] {
        let mut table = [0.0; TOTAL_TABLE];
        for i in BASE_INDEX..BASE_INDEX + TABLE_SIZE {
            let [hue, cusp_j, cusp_m] = self.cusps[i];
            let cusp = [cusp_j, cusp_m];
            let threshold = lerp(cusp_j, self.limit_j_max, FOCUS_GAIN_BLEND);
            let focus_j = compute_focus_j(cusp_j, self.mid_j, self.limit_j_max);
            let tests = TEST_POSITIONS.map(|position| {
                let j = lerp(cusp_j, self.limit_j_max, position);
                let gain = get_focus_gain(j, threshold, self.limit_j_max, self.focus_dist);
                let intersect = solve_j_intersect(j, cusp_m, focus_j, self.limit_j_max, gain);
                let slope = compression_slope(intersect, focus_j, self.limit_j_max, gain);
                let intersect_cusp =
                    solve_j_intersect(cusp_j, cusp_m, focus_j, self.limit_j_max, gain);
                (intersect, slope, intersect_cusp)
            });
            let fits = |gamma_top_inv: f64| {
                tests.iter().all(|&(intersect, slope, intersect_cusp)| {
                    let m = gamut_boundary_m(
                        cusp,
                        self.limit_j_max,
                        gamma_top_inv,
                        self.lower_hull_gamma_inv,
                        intersect,
                        slope,
                        intersect_cusp,
                    );
                    let rgb = self.limit.jmh_to_rgb([intersect + slope * m, m, hue]);
                    let limit = self.peak / REF_LUMINANCE;
                    rgb.iter().any(|&c| c > limit)
                })
            };
            let mut low = GAMMA_MINIMUM;
            let mut high = low + GAMMA_SEARCH_STEP;
            while high < GAMMA_MAXIMUM && !fits(1.0 / high) {
                low = high;
                high += GAMMA_SEARCH_STEP;
            }
            while high - low > GAMMA_ACCURACY {
                let gamma = (high + low) / 2.0;
                if fits(1.0 / gamma) {
                    high = gamma;
                } else {
                    low = gamma;
                }
            }
            table[i] = 1.0 / high;
        }
        table[0] = table[TABLE_SIZE];
        table[TABLE_SIZE + BASE_INDEX] = table[BASE_INDEX];
        table
    }

    fn params(&self) -> Params {
        let gpu = |m: &M3| {
            Mat3::from_cols(
                Vec3::new(m[0][0] as f32, m[1][0] as f32, m[2][0] as f32),
                Vec3::new(m[0][1] as f32, m[1][1] as f32, m[2][1] as f32),
                Vec3::new(m[0][2] as f32, m[1][2] as f32, m[2][2] as f32),
            )
        };
        let ap0_to_ap1 = mul_m(&rgb_to_xyz(&AP0), &xyz_to_rgb(&AP1));
        let ap1_to_ap0 = mul_m(&rgb_to_xyz(&AP1), &xyz_to_rgb(&AP0));
        let rec709_to_ap0 = mul_m(
            &rgb_to_xyz(&REC709),
            &mul_m(&adaptation(REC709[3], AP0[3]), &xyz_to_rgb(&AP0)),
        );
        Params {
            rec709_to_ap1: gpu(&mul_m(&rec709_to_ap0, &ap0_to_ap1)),
            ap1_to_cam: gpu(&mul_m(&ap1_to_ap0, &self.input.rgb_to_cam)),
            input_to_aab: gpu(&self.input.to_aab),
            limit_from_aab: gpu(&self.limit.from_aab),
            limit_cam_to_rgb: gpu(&self.limit.cam_to_rgb),
            f_l_n: self.input.f_l_n as f32,
            cz: self.input.cz as f32,
            inv_cz: self.input.inv_cz as f32,
            a_w_j: self.input.a_w_j as f32,
            inv_a_w_j: self.input.inv_a_w_j as f32,
            forward_limit: self.ts.forward_limit as f32,
            ts_n_r: self.ts.n_r as f32,
            ts_g: self.ts.g as f32,
            ts_t_1: self.ts.t_1 as f32,
            ts_s_2: self.ts.s_2 as f32,
            ts_m_2: self.ts.m_2 as f32,
            limit_j_max: self.limit_j_max as f32,
            model_gamma_inv: self.model_gamma_inv as f32,
            sat: self.sat as f32,
            sat_thr: self.sat_thr as f32,
            compr: self.compr as f32,
            chroma_compress_scale: self.chroma_compress_scale as f32,
            mid_j: self.mid_j as f32,
            focus_dist: self.focus_dist as f32,
            lower_hull_gamma_inv: self.lower_hull_gamma_inv as f32,
            hue_search: IVec2::from_array(self.hue_search),
            table: std::array::from_fn(|i| {
                let [hue, j, m] = self.cusps[i];
                Vec4::new(
                    hue as f32,
                    j as f32,
                    m as f32,
                    self.upper_hull_gamma[i] as f32,
                )
            }),
            reach_m: std::array::from_fn(|i| {
                Vec4::from_array(std::array::from_fn(|k| {
                    self.reach_m.get(4 * i + k).copied().unwrap_or(0.0) as f32
                }))
            }),
        }
    }
}

/// The reach's M at the limit's J, a hue a degree (`make_reach_m_table`).
fn reach_m_table(reach: &Cam, limit_j_max: f64) -> [f64; TOTAL_TABLE] {
    let mut table = [0.0; TOTAL_TABLE];
    let outside = |m: f64, hue: f64| {
        reach
            .jmh_to_rgb([limit_j_max, m, hue])
            .iter()
            .any(|&c| c < 0.0)
    };
    for i in 0..TABLE_SIZE {
        let hue = i as f64 * HUE_LIMIT / TABLE_SIZE as f64;
        let (search_range, search_maximum) = (50.0, 1300.0);
        let mut low = 0.0;
        let mut high = low + search_range;
        while high < search_maximum && !outside(high, hue) {
            low = high;
            high += search_range;
        }
        while high - low > 1e-2 {
            let m = (high + low) / 2.0;
            if outside(m, hue) {
                high = m;
            } else {
                low = m;
            }
        }
        table[i + BASE_INDEX] = high;
    }
    table[0] = table[TABLE_SIZE];
    table[BASE_INDEX + TABLE_SIZE] = table[BASE_INDEX];
    table
}

/// The two gamuts' corners' hues, merged in order
/// (`extract_sorted_cube_hues`).
fn sorted_hues(
    reach: &[V3; TOTAL_CORNERS],
    limit: &[V3; TOTAL_CORNERS],
) -> [f64; MAX_SORTED_CORNERS] {
    let mut sorted = [0.0; MAX_SORTED_CORNERS];
    let (mut r, mut l, mut idx) = (1, 1, 0);
    while (r < CUSP_CORNERS + 1 || l < CUSP_CORNERS + 1) && idx < MAX_SORTED_CORNERS {
        let (reach_hue, limit_hue) = (reach[r][2], limit[l][2]);
        if reach_hue == limit_hue {
            sorted[idx] = reach_hue;
            r += 1;
            l += 1;
        } else if reach_hue < limit_hue {
            sorted[idx] = reach_hue;
            r += 1;
        } else {
            sorted[idx] = limit_hue;
            l += 1;
        }
        idx += 1;
    }
    sorted
}

/// Hues as evenly spread as can be, the corners' among them
/// (`build_hue_table`).
fn hue_table(sorted: [f64; MAX_SORTED_CORNERS]) -> [f64; TOTAL_TABLE] {
    let mut table = [0.0; TOTAL_TABLE];
    let spacing = TABLE_SIZE as f64 / HUE_LIMIT;
    let mut samples = [0i64; MAX_SORTED_CORNERS];
    let mut last = 0i64;
    let mut min_index = if sorted[0] == 0.0 { 0 } else { 1 };
    for idx in 0..MAX_SORTED_CORNERS {
        let mut nominal = ((sorted[idx] * spacing).round() as i64)
            .max(min_index)
            .min(TABLE_SIZE as i64 - 1);
        if last == nominal {
            if idx > 1 && samples[idx - 2] != samples[idx - 1] - 1 {
                samples[idx - 1] -= 1;
            } else {
                nominal += 1;
            }
        }
        samples[idx] = nominal.min(TABLE_SIZE as i64 - 1);
        min_index = nominal;
        last = min_index;
    }
    let mut fill = |count: i64, lower: f64, upper: f64, base: i64| {
        let delta = (upper - lower) / count as f64;
        for i in 0..count {
            table[(base + i) as usize] = lower + i as f64 * delta;
        }
    };
    let mut total = 0;
    fill(samples[0], 0.0, sorted[0], total + 1);
    total += samples[0];
    for i in 1..MAX_SORTED_CORNERS {
        let count = samples[i] - samples[i - 1];
        fill(count, sorted[i - 1], sorted[i], total + 1);
        total += count;
    }
    fill(
        TABLE_SIZE as i64 - total,
        sorted[MAX_SORTED_CORNERS - 1],
        HUE_LIMIT,
        total + 1,
    );
    table[0] = table[BASE_INDEX + TABLE_SIZE - 1] - HUE_LIMIT;
    table[BASE_INDEX + TABLE_SIZE] = table[BASE_INDEX] + HUE_LIMIT;
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables are whole and in order, and a few of their numbers as the
    /// reference has them for SDR Rec.709.
    #[test]
    fn tables() {
        let odt = Odt::new(PEAK_LUMINANCE, REC709);
        assert!((odt.limit_j_max - 100.0).abs() < 1.0, "{}", odt.limit_j_max);
        for i in 1..TOTAL_TABLE {
            assert!(
                odt.cusps[i][0] > odt.cusps[i - 1][0],
                "hues out of order at {i}"
            );
        }
        for i in 0..TOTAL_TABLE {
            assert!(
                odt.reach_m[i] > 0.0 && odt.upper_hull_gamma[i] > 0.0,
                "empty at {i}"
            );
            assert!(
                odt.cusps[i][1] > 0.0 && odt.cusps[i][2] > 0.0,
                "no cusp at {i}"
            );
        }
    }
}
