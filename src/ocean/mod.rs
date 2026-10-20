//! Plugin de l'océan : champ de vagues, maillage de la surface qui suit la caméra,
//! matériau de l'eau et paramètres partagés avec les autres shaders (fond marin, ciel).

pub mod fft;
pub mod shore;
pub mod waves;

use std::f32::consts::TAU;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor},
    light::NotShadowCaster,
    mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, ColorWrites, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

use crate::sky_data;
use fft::{FftCascade, FftWaves, WindSpectrum};
pub use waves::{FFT_SPLIT_WAVELENGTH, MAX_WAVES, SeaState, WaveField};

/// Niveau moyen de la mer (m).
pub const SEA_LEVEL: f32 = 0.0;
/// Profondeur du fond de sable au large (m).
pub const FLOOR_DEPTH: f32 = 15.0;

pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((MaterialPlugin::<WaterMaterial>::default(), MaterialPlugin::<WaterMotionMaterial>::default()))
            .insert_resource(Ocean::new(SeaState::default()))
            .init_resource::<SimulationTime>()
            .add_systems(PreStartup, init_ocean_textures)
            .add_systems(Startup, spawn_water)
            .add_systems(
                PostUpdate,
                (update_ocean, sync_ocean_uniforms::<WaterMaterial>, sync_ocean_uniforms::<WaterMotionMaterial>)
                    .chain()
                    .in_set(OceanSystems::Sync),
            )
            .configure_sets(PostUpdate, OceanSystems::Sync);
    }
}

/// Ensemble de systèmes qui synchronisent les paramètres de l'océan vers les matériaux.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum OceanSystems {
    Sync,
}

/// Horloge de la simulation (peut être figée pour des captures reproductibles).
#[derive(Resource, Default)]
pub struct SimulationTime {
    pub seconds: f64,
    pub frozen: Option<f64>,
}

/// Paramètres optiques de l'eau (coefficients par canal RGB, en 1/m).
#[derive(Clone, Copy, Debug)]
pub struct WaterOptics {
    /// Absorption de l'eau claire tropicale.
    pub absorption: Vec3,
    /// Diffusion (particules, molécules).
    pub scattering: Vec3,
    /// Part rétrodiffusée de la diffusion.
    pub backscatter_ratio: f32,
}

impl Default for WaterOptics {
    fn default() -> Self {
        Self {
            absorption: Vec3::new(0.30, 0.030, 0.024),
            scattering: Vec3::new(0.020, 0.026, 0.040),
            backscatter_ratio: 0.03,
        }
    }
}

/// Textures partagées par les matériaux de la scène.
#[derive(Resource, Clone)]
pub struct OceanTextures {
    /// Cubemap HDR du ciel (sans le disque solaire), avec mipmaps, en kcd/m².
    pub sky: Handle<Image>,
    /// Pentes et dérivées secondes des petites vagues (FFT), tableau de 2 cascades.
    pub fft_slopes: Handle<Image>,
    pub fft_extra: Handle<Image>,
    /// Vagues côtières précalculées (voir `shore.rs`).
    pub shore0: Handle<Image>,
    pub shore1: Handle<Image>,
    /// Écume photographiée (ambientCG Foam001/003, CC0) : opacités et reliefs.
    pub foam: Handle<Image>,
    /// Sable de plage (Poly Haven aerial_beach_01, CC0) : albédo, et (normale xy, rugosité, relief).
    pub sand_albedo: Handle<Image>,
    pub sand_nrh: Handle<Image>,
    /// Sable sous la végétation, avec débris végétaux (Poly Haven forrest_sand_01, CC0).
    pub litter_albedo: Handle<Image>,
    pub litter_nrh: Handle<Image>,
}

/// État global de l'océan.
#[derive(Resource)]
pub struct Ocean {
    pub field: WaveField,
    pub optics: WaterOptics,
    pub sea_state: SeaState,
    /// La caméra est-elle sous la surface ?
    pub camera_underwater: bool,
    /// Hauteur de l'eau sous la caméra.
    pub surface_at_camera: f32,
    /// Variance des pentes non résolues (vaguelettes capillaires < 2 cm).
    pub capillary_slope_variance: f32,
    pub caustics_strength: f32,
    pub foam_strength: f32,
    /// (taille, variance des pentes, octave haute, étendue en octaves) des cascades FFT
    pub fft_info: [Vec4; 2],
    /// Mode de débogage des shaders (variable d'environnement `OCEAN_DEBUG`).
    pub debug_mode: f32,
    /// Effets désactivés (réglages graphiques, ou `OCEAN_FLAGS` pour les mesures) :
    /// 1 réflexions en espace écran, 2 rayons sous l'eau, 4 caustiques des vagues côtières.
    pub debug_flags: f32,
    /// Ambiance (voir `OceanUniforms::look`).
    pub look: Vec4,
    /// Ambiance (voir `OceanUniforms::look2`).
    pub look2: Vec4,
    pub uniforms: OceanUniforms,
    /// Vagues côtières précalculées (au démarrage).
    pub shore: Option<shore::ShoreField>,
    /// Temps de la simulation (s), tel qu'envoyé aux shaders.
    pub time: f32,
}

impl Ocean {
    pub fn new(state: SeaState) -> Self {
        let field = WaveField::new(&state);
        let mut ocean = Self {
            field,
            optics: WaterOptics::default(),
            sea_state: state,
            camera_underwater: false,
            surface_at_camera: 0.0,
            capillary_slope_variance: 0.004,
            caustics_strength: 1.0,
            foam_strength: 1.0,
            fft_info: [Vec4::ZERO; 2],
            debug_mode: std::env::var("OCEAN_DEBUG").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            debug_flags: std::env::var("OCEAN_FLAGS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            look: Vec4::new(1.0, 0.72, 0.5, 0.0),
            look2: Vec4::ZERO,
            uniforms: OceanUniforms::default(),
            shore: None,
            time: 0.0,
        };
        ocean.rebuild_uniforms(0.0);
        ocean
    }

    /// Hauteur absolue de la surface de l'eau en (x, z) : houle et mer du vent (atténuées
    /// près de l'île), vagues côtières, et nappe du jet de rive sur la plage.
    pub fn surface_height(&self, x: f32, z: f32) -> f32 {
        let Some(field) = &self.shore else {
            return SEA_LEVEL + self.field.height_at(x, z);
        };
        let st = shore::shore_state(field, x, z, self.time);
        if let Some(y) = st.beach_y {
            return y;
        }
        SEA_LEVEL + self.field.height_at_weighted(x, z, st.swell_w, st.wind_w) + st.eta
    }

    fn rebuild_uniforms(&mut self, time: f32) {
        let u = &mut self.uniforms;
        u.params0 = Vec4::new(
            time,
            if self.camera_underwater { 1.0 } else { 0.0 },
            SEA_LEVEL,
            self.field.waves.len() as f32,
        );
        u.params1 = Vec4::new(self.debug_flags, self.caustics_strength, self.foam_strength, sky_data::SKY_CUBE_SCALE);
        u.params2 = Vec4::new(self.capillary_slope_variance, FLOOR_DEPTH, self.surface_at_camera, self.debug_mode);
        let sd = Vec3::from_array(sky_data::SUN_DIRECTION).normalize();
        u.sun_dir = sd.extend(sky_data::SUN_ILLUMINANCE);
        // rayon angulaire du soleil : 0.2666°
        // ambiance : soleil légèrement doré (look2.y), luminance conservée
        let warm = Vec3::new(1.0 + 0.10 * self.look2.y, 1.0, 1.0 - 0.22 * self.look2.y);
        let warm = warm / warm.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        u.sun_color = (Vec3::from_array(sky_data::SUN_COLOR) * warm).extend(0.004_653);
        u.absorption = self.optics.absorption.extend(0.0);
        u.scattering = self.optics.scattering.extend(self.optics.backscatter_ratio);
        u.fft_a = self.fft_info[0];
        u.fft_b = self.fft_info[1];
        u.look = self.look;
        u.look2 = self.look2;
        for (i, c) in sky_data::SKY_SH_IRRADIANCE.iter().enumerate() {
            u.sh[i] = Vec4::new(c[0], c[1], c[2], 0.0);
        }
        for (i, w) in self.field.waves.iter().enumerate().take(MAX_WAVES) {
            u.waves[2 * i] = Vec4::new(w.dir.x, w.dir.y, w.k, w.amplitude);
            u.waves[2 * i + 1] = Vec4::new(w.omega, self.field.phases[i], w.steepness, TAU / w.k);
        }
    }
}

/// Paramètres partagés par tous les shaders de la scène (même structure côté WGSL :
/// `OceanUniforms` dans `shaders/ocean_common.wesl`).
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct OceanUniforms {
    /// x : temps (s), y : caméra sous l'eau (0/1), z : niveau de la mer, w : nombre de vagues
    pub params0: Vec4,
    /// x : effets désactivés (débogage), y : force des caustiques, z : force de l'écume,
    /// w : échelle de la cubemap du ciel
    pub params1: Vec4,
    /// x : variance des pentes capillaires, y : profondeur du fond, z : hauteur de l'eau sous la caméra
    pub params2: Vec4,
    /// xyz : direction vers le soleil, w : éclairement (lux)
    pub sun_dir: Vec4,
    /// rgb : couleur du soleil, w : rayon angulaire (rad)
    pub sun_color: Vec4,
    /// rgb : absorption (1/m)
    pub absorption: Vec4,
    /// rgb : diffusion (1/m), w : part rétrodiffusée
    pub scattering: Vec4,
    /// cascade FFT A : (taille, variance des pentes, octave haute, étendue en octaves)
    pub fft_a: Vec4,
    /// cascade FFT B
    pub fft_b: Vec4,
    /// Ambiance : x saturation du ciel, y force des ombres des nuages, z couverture nuageuse,
    /// w libre
    pub look: Vec4,
    /// Ambiance : x « dé-brume » de l'horizon, y soleil doré, z éclat des scintillements, w libre
    pub look2: Vec4,
    /// Irradiance du ciel en harmoniques sphériques (lux)
    pub sh: [Vec4; 9],
    /// Vagues : (dir.x, dir.z, k, amplitude), (pulsation, phase, raideur, longueur d'onde)
    pub waves: [Vec4; MAX_WAVES * 2],
}

impl Default for OceanUniforms {
    fn default() -> Self {
        Self {
            params0: Vec4::ZERO,
            params1: Vec4::ZERO,
            params2: Vec4::ZERO,
            sun_dir: Vec4::Y,
            sun_color: Vec4::ONE,
            absorption: Vec4::ZERO,
            scattering: Vec4::ZERO,
            fft_a: Vec4::ONE,
            fft_b: Vec4::ONE,
            look: Vec4::new(1.0, 0.72, 0.5, 0.0),
            look2: Vec4::ZERO,
            sh: [Vec4::ZERO; 9],
            waves: [Vec4::ZERO; MAX_WAVES * 2],
        }
    }
}

/// Les matériaux qui reçoivent les paramètres de l'océan.
pub trait OceanMaterial: Material {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms;
}

/// Matériau de la surface de l'eau.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterMaterial {
    #[uniform(0)]
    pub ocean: OceanUniforms,
    #[texture(1, dimension = "cube")]
    #[sampler(2)]
    pub sky: Handle<Image>,
    #[texture(3, dimension = "2d_array")]
    #[sampler(4)]
    pub fft_slopes: Handle<Image>,
    #[texture(5, dimension = "2d_array")]
    pub fft_extra: Handle<Image>,
    #[texture(6, filterable = false)]
    pub shore0: Handle<Image>,
    #[texture(7, filterable = false)]
    pub shore1: Handle<Image>,
    #[texture(8)]
    #[sampler(9)]
    pub foam: Handle<Image>,
}

impl OceanMaterial for WaterMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for WaterMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/water.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/water.wesl".into()
    }

    // L'eau lit l'image de la scène opaque (le fond) pour la réfraction.
    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // visible des deux côtés (au-dessus et en dessous de la surface)
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

#[derive(Component)]
pub struct WaterSurface;

/// Double invisible de la surface de l'eau : il n'écrit que les vecteurs de mouvement de
/// l'eau dans la passe préalable (l'eau, transparente, n'y passe pas). Sans lui,
/// l'anti-crénelage temporel et le flou de mouvement prendraient le mouvement du fond.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterMotionMaterial {
    #[uniform(0)]
    pub ocean: OceanUniforms,
    #[texture(6, filterable = false)]
    pub shore0: Handle<Image>,
    #[texture(7, filterable = false)]
    pub shore1: Handle<Image>,
}

impl OceanMaterial for WaterMotionMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for WaterMotionMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/water_motion.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/water_motion.wesl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/water_motion_prepass.wesl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/water_motion_prepass.wesl".into()
    }

    // dessiné après les objets opaques dans la passe préalable : il est testé contre leur
    // profondeur (le fond sous l'eau est derrière, l'île devant)
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(0.5)
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        if descriptor.label.as_deref() == Some("prepass_pipeline") {
            // ni profondeur ni normale : seulement les vecteurs de mouvement
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = Some(false);
            }
            if let Some(fragment) = descriptor.fragment.as_mut() {
                if let Some(Some(target)) = fragment.targets.get_mut(0) {
                    target.write_mask = ColorWrites::empty();
                }
            }
        }
        Ok(())
    }
}

/// Charge le ciel et crée les cascades FFT.
fn init_ocean_textures(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut ocean: ResMut<Ocean>,
) {
    let sky = assets
        .load_builder()
        .with_settings(|s: &mut ImageLoaderSettings| {
            s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::ClampToEdge,
                address_mode_v: ImageAddressMode::ClampToEdge,
                address_mode_w: ImageAddressMode::ClampToEdge,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                ..default()
            });
        })
        .load("sky/sky_cube.ktx2");

    let spec = WindSpectrum {
        wind_dir: ocean.field.wind_dir,
        peak_omega: ocean.field.peak_omega,
        alpha: ocean.field.alpha,
        boost: ocean.sea_state.detail_boost,
    };
    let n = 128usize;
    let split = TAU / FFT_SPLIT_WAVELENGTH;
    let size_a = 9.7f32;
    let size_b = 1.45f32;
    let k_mid = TAU / 0.22;
    let k_nyq_b = std::f32::consts::PI * n as f32 / size_b;
    let cascades = vec![
        FftCascade::new(size_a, n, (split, k_mid), &spec, 1001),
        FftCascade::new(size_b, n, (k_mid, 0.92 * k_nyq_b), &spec, 2002),
    ];
    for (i, c) in cascades.iter().enumerate() {
        let f_hi = c.k_band.1 * c.size / TAU;
        let f_lo = c.k_band.0 * c.size / TAU;
        let oct_hi = f_hi.log2() - (n as f32 * 0.5).log2();
        let span = (f_hi / f_lo).log2();
        ocean.fft_info[i] = Vec4::new(c.size, c.slope_variance, oct_hi, span);
        info!(
            "Cascade FFT {} : motif de {:.2} m, variance des pentes {:.4}",
            i, c.size, c.slope_variance
        );
    }
    let mut fft = FftWaves::new(cascades, &mut images);
    fft.update(0.0, &mut images);
    // vagues côtières : propagation de la houle autour de l'île
    let field = shore::ShoreField::bake(ocean.sea_state.swell_dir, ocean.sea_state.swell_hs, &mut images);
    // textures photographiées, répétées, filtrées (anisotrope)
    let tiled = |path: &str| -> Handle<Image> {
        assets
            .load_builder()
            .with_settings(|s: &mut ImageLoaderSettings| {
                s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                    address_mode_u: ImageAddressMode::Repeat,
                    address_mode_v: ImageAddressMode::Repeat,
                    mag_filter: ImageFilterMode::Linear,
                    min_filter: ImageFilterMode::Linear,
                    mipmap_filter: ImageFilterMode::Linear,
                    anisotropy_clamp: 16,
                    ..default()
                });
            })
            .load(path.to_string())
    };
    commands.insert_resource(OceanTextures {
        sky,
        fft_slopes: fft.slopes.clone(),
        fft_extra: fft.extra.clone(),
        shore0: field.image0.clone(),
        shore1: field.image1.clone(),
        foam: tiled("textures/foam.ktx2"),
        sand_albedo: tiled("textures/sand_albedo.ktx2"),
        sand_nrh: tiled("textures/sand_nrh.ktx2"),
        litter_albedo: tiled("textures/litter_albedo.ktx2"),
        litter_nrh: tiled("textures/litter_nrh.ktx2"),
    });
    ocean.shore = Some(field);
    commands.insert_resource(fft);
}

/// Maillage radial centré sur la caméra : très dense près de la caméra, de plus en plus
/// lâche vers l'horizon (le shader le déplace pour qu'il suive la caméra).
fn build_water_mesh() -> Mesh {
    let segments = 320usize;
    let r0 = 0.35f32;
    let growth = 1.0 + TAU / segments as f32;
    let r_max = 30_000.0f32;
    let mut radii = Vec::new();
    let mut r = r0;
    while r < r_max {
        radii.push(r);
        r *= growth;
    }
    radii.push(r_max);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(1 + radii.len() * segments);
    positions.push([0.0, 0.0, 0.0]);
    for &r in &radii {
        for j in 0..segments {
            let a = j as f32 / segments as f32 * TAU;
            positions.push([r * a.cos(), 0.0, r * a.sin()]);
        }
    }
    let ring = |i: usize, j: usize| -> u32 { (1 + i * segments + (j % segments)) as u32 };
    let mut indices: Vec<u32> = Vec::with_capacity(segments * 3 + (radii.len() - 1) * segments * 6);
    for j in 0..segments {
        indices.extend_from_slice(&[0, ring(0, j + 1), ring(0, j)]);
    }
    for i in 0..radii.len() - 1 {
        for j in 0..segments {
            let a = ring(i, j);
            let b = ring(i, j + 1);
            let c = ring(i + 1, j);
            let d = ring(i + 1, j + 1);
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let normals = vec![[0.0f32, 1.0, 0.0]; positions.len()];
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_indices(Indices::U32(indices))
}

fn spawn_water(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut motion_materials: ResMut<Assets<WaterMotionMaterial>>,
    ocean: Res<Ocean>,
    textures: Res<OceanTextures>,
) {
    info!(
        "Océan : {} grandes vagues, écart-type de hauteur {:.3} m, variance des pentes {:.4}",
        ocean.field.waves.len(),
        ocean.field.height_std(),
        ocean.field.slope_variance()
    );
    let mesh = meshes.add(build_water_mesh());
    commands.spawn((
        Name::new("Mouvement de l'eau"),
        Mesh3d(mesh.clone()),
        MeshMaterial3d(motion_materials.add(WaterMotionMaterial {
            ocean: ocean.uniforms,
            shore0: textures.shore0.clone(),
            shore1: textures.shore1.clone(),
        })),
        Transform::default(),
        NoFrustumCulling,
        NotShadowCaster,
    ));
    commands.spawn((
        Name::new("Surface de l'eau"),
        WaterSurface,
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(WaterMaterial {
            ocean: ocean.uniforms,
            sky: textures.sky.clone(),
            fft_slopes: textures.fft_slopes.clone(),
            fft_extra: textures.fft_extra.clone(),
            shore0: textures.shore0.clone(),
            shore1: textures.shore1.clone(),
            foam: textures.foam.clone(),
        })),
        Transform::default(),
        NoFrustumCulling,
        NotShadowCaster,
    ));
}

/// Avance le temps, simule les vagues et met à jour l'état "caméra sous l'eau".
fn update_ocean(
    time: Res<Time>,
    mut sim: ResMut<SimulationTime>,
    mut ocean: ResMut<Ocean>,
    mut fft: ResMut<FftWaves>,
    mut images: ResMut<Assets<Image>>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
) {
    sim.seconds = match sim.frozen {
        Some(t) => t,
        None => sim.seconds + time.delta_secs_f64(),
    };
    let t = sim.seconds;
    ocean.field.update_phases(t);
    let t0 = std::time::Instant::now();
    fft.update(t, &mut images);
    if std::env::var("OCEAN_FPS_LOG").is_ok() && (time.elapsed_secs() % 2.0) < time.delta_secs() {
        info!("FFT : {:.2} ms", t0.elapsed().as_secs_f32() * 1000.0);
    }
    if let Some(cam) = camera.iter().next() {
        let p = cam.translation();
        let h = ocean.surface_height(p.x, p.z);
        ocean.surface_at_camera = h;
        ocean.camera_underwater = p.y < h;
    }
    let now = (t % 3600.0) as f32;
    ocean.time = now;
    ocean.rebuild_uniforms(now);
}

/// Copie les paramètres de l'océan dans tous les matériaux de type `M`.
pub fn sync_ocean_uniforms<M: OceanMaterial>(ocean: Res<Ocean>, mut materials: ResMut<Assets<M>>) {
    for (_, mat) in materials.iter_mut() {
        *mat.ocean_uniforms_mut() = ocean.uniforms;
    }
}
