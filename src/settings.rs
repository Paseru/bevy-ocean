//! Réglages graphiques : menu (touche Échap), préréglages Bas / Moyen / Élevé / Ultra,
//! ambiance des couleurs, sauvegarde automatique dans `reglages.txt`.

use bevy::{
    anti_alias::{
        contrast_adaptive_sharpening::ContrastAdaptiveSharpening,
        fxaa::Fxaa,
        smaa::{Smaa, SmaaPreset},
        taa::TemporalAntiAliasing,
    },
    core_pipeline::tonemapping::Tonemapping,
    curve::cubic_splines::LinearSpline,
    light::{DirectionalLightShadowMap, ShadowFilteringMethod},
    pbr::{ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel},
    post_process::{
        auto_exposure::{AutoExposure, AutoExposureCompensationCurve, AutoExposurePlugin},
        bloom::{Bloom, BloomCompositeMode, BloomPrefilter},
        dof::{DepthOfField, DepthOfFieldMode},
        effect_stack::{ChromaticAberration, Vignette},
        motion_blur::MotionBlur,
    },
    prelude::*,
    picking::hover::Hovered,
    render::{
        camera::{MipBias, TemporalJitter},
        view::{ColorGrading, ColorGradingGlobal, ColorGradingSection},
    },
    ui::Pressed,
    ui_widgets::{Activate, Button},
    window::{CursorGrabMode, CursorOptions, PresentMode, PrimaryWindow},
};

use crate::camera::FlyCamera;
use crate::ocean::Ocean;
use crate::post::{FilmGrain, LensFlare};
use crate::render_scale::RenderScale;

const SETTINGS_FILE: &str = "reglages.txt";

/// Systèmes du menu et de l'application des réglages.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SettingsSystems;

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        let capture = std::env::var("OCEAN_SHOTS").is_ok() || std::env::var("OCEAN_WINDOW_SHOT").is_ok();
        let mut settings = if capture { GraphicsSettings::default() } else { GraphicsSettings::load() };
        if let Ok(p) = std::env::var("OCEAN_PRESET") {
            if let Some(p) = Preset::parse(&p) {
                settings.apply_preset(p);
            }
        }
        if let Ok(v) = std::env::var("OCEAN_AA") {
            settings.aa = match v.as_str() {
                "smaa" => AntiAliasing::Smaa,
                "fxaa" => AntiAliasing::Fxaa,
                "aucun" => AntiAliasing::Aucun,
                _ => AntiAliasing::Taa,
            };
        }
        if let Ok(v) = std::env::var("OCEAN_LOOK") {
            settings.look = match v.as_str() {
                "naturel" => Look::Naturel,
                "tropical" => Look::Tropical,
                _ => Look::Cinema,
            };
        }
        app.add_plugins(AutoExposurePlugin)
            .insert_resource(settings)
            .insert_resource(Persist(!capture))
            .init_resource::<Menu>()
            .add_systems(Startup, (setup_exposure_curve, spawn_menu))
            .add_systems(
                Update,
                (toggle_menu, menu_keyboard, menu_buttons, refresh_menu, apply_settings)
                    .chain()
                    .in_set(SettingsSystems),
            );
    }
}

// ---------------------------------------------------------------------------------------------
// Réglages
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Bas,
    Moyen,
    Eleve,
    Ultra,
    Perso,
}

impl Preset {
    fn name(self) -> &'static str {
        match self {
            Preset::Bas => "Bas",
            Preset::Moyen => "Moyen",
            Preset::Eleve => "Élevé",
            Preset::Ultra => "Ultra",
            Preset::Perso => "Personnalisé",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bas" => Some(Preset::Bas),
            "moyen" => Some(Preset::Moyen),
            "eleve" | "élevé" => Some(Preset::Eleve),
            "ultra" => Some(Preset::Ultra),
            "perso" | "personnalisé" => Some(Preset::Perso),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AntiAliasing {
    /// anti-crénelage temporel : le plus propre (feuilles, herbe, reflets qui scintillent)
    Taa,
    Smaa,
    Fxaa,
    Aucun,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Look {
    /// façon Unreal Engine 5 : lumière éclatante, gros halo lumineux (bloom), soleil
    /// éblouissant, contraste et couleurs poussés
    Cinema,
    /// lumière franche, ciel bleu profond, couleurs vives (ambiance carte postale)
    Tropical,
    /// rendu neutre, proche d'une photo non retouchée
    Naturel,
}

#[derive(Resource, Clone, PartialEq, Debug)]
pub struct GraphicsSettings {
    pub preset: Preset,
    /// part de la résolution de la fenêtre utilisée pour le rendu 3D
    pub render_scale: f32,
    pub aa: AntiAliasing,
    /// occlusion ambiante et ombres de contact : 0 non, 1 moyenne, 2 haute, 3 ultra
    pub ao: u8,
    /// ombres du soleil : 0 basses (1024), 1 moyennes (2048), 2 hautes (4096) ; douces dès 1
    pub shadows: u8,
    pub reflections: bool,
    pub light_shafts: bool,
    pub motion_blur: bool,
    pub depth_of_field: bool,
    /// vignettage, aberration chromatique, reflets d'objectif, grain
    pub lens: bool,
    pub look: Look,
    /// correction d'exposition (IL) : + = plus lumineux
    pub brightness: f32,
    pub auto_exposure: bool,
    pub fov: f32,
    pub vsync: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        let mut s = Self {
            preset: Preset::Eleve,
            render_scale: 0.75,
            aa: AntiAliasing::Taa,
            ao: 1,
            shadows: 1,
            reflections: true,
            light_shafts: true,
            motion_blur: true,
            depth_of_field: true,
            lens: true,
            look: Look::Cinema,
            brightness: 0.0,
            auto_exposure: false,
            fov: 45.0,
            vsync: true,
        };
        s.apply_preset(Preset::Eleve);
        s
    }
}

impl GraphicsSettings {
    pub fn apply_preset(&mut self, p: Preset) {
        self.preset = p;
        let (scale, aa, ao, shadows, refl, shafts, mb, dof, lens) = match p {
            Preset::Bas => (0.55, AntiAliasing::Fxaa, 0, 0, false, false, false, false, false),
            Preset::Moyen => (0.65, AntiAliasing::Taa, 0, 1, true, true, false, false, true),
            Preset::Eleve => (0.75, AntiAliasing::Taa, 1, 1, true, true, true, true, true),
            Preset::Ultra => (1.0, AntiAliasing::Taa, 3, 2, true, true, true, true, true),
            Preset::Perso => return,
        };
        self.render_scale = scale;
        self.aa = aa;
        self.ao = ao;
        self.shadows = shadows;
        self.reflections = refl;
        self.light_shafts = shafts;
        self.motion_blur = mb;
        self.depth_of_field = dof;
        self.lens = lens;
    }

    fn load() -> Self {
        let mut s = Self::default();
        let Ok(text) = std::fs::read_to_string(SETTINGS_FILE) else {
            return s;
        };
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            let (k, v) = (k.trim(), v.trim());
            let b = v == "oui";
            match k {
                "preset" => {
                    if let Some(p) = Preset::parse(v) {
                        s.preset = p;
                    }
                }
                "resolution" => s.render_scale = v.parse().unwrap_or(s.render_scale),
                "anticrenelage" => {
                    s.aa = match v {
                        "taa" => AntiAliasing::Taa,
                        "smaa" => AntiAliasing::Smaa,
                        "fxaa" => AntiAliasing::Fxaa,
                        _ => AntiAliasing::Aucun,
                    }
                }
                "occlusion" => s.ao = v.parse().unwrap_or(s.ao),
                "ombres" => s.shadows = v.parse().unwrap_or(s.shadows),
                "reflets" => s.reflections = b,
                "rayons" => s.light_shafts = b,
                "flou_mouvement" => s.motion_blur = b,
                "profondeur_champ" => s.depth_of_field = b,
                "effets_objectif" => s.lens = b,
                "ambiance" => {
                    s.look = match v {
                        "naturel" => Look::Naturel,
                        "tropical" => Look::Tropical,
                        _ => Look::Cinema,
                    }
                }
                "luminosite" => s.brightness = v.parse().unwrap_or(0.0),
                "exposition_auto" => s.auto_exposure = b,
                "champ_vision" => s.fov = v.parse().unwrap_or(45.0),
                "vsync" => s.vsync = b,
                _ => {}
            }
        }
        s.render_scale = s.render_scale.clamp(0.4, 1.0);
        s.ao = s.ao.min(3);
        s.shadows = s.shadows.min(2);
        s.fov = s.fov.clamp(30.0, 90.0);
        s.brightness = s.brightness.clamp(-2.0, 2.0);
        s
    }

    fn save(&self) {
        let yn = |b: bool| if b { "oui" } else { "non" };
        let text = format!(
            "# Réglages graphiques (modifiables aussi dans le jeu : touche Échap)\n\
             preset = {}\nresolution = {:.2}\nanticrenelage = {}\nocclusion = {}\nombres = {}\n\
             reflets = {}\nrayons = {}\nflou_mouvement = {}\nprofondeur_champ = {}\n\
             effets_objectif = {}\nambiance = {}\nluminosite = {:.2}\nexposition_auto = {}\n\
             champ_vision = {:.0}\nvsync = {}\n",
            match self.preset {
                Preset::Bas => "bas",
                Preset::Moyen => "moyen",
                Preset::Eleve => "eleve",
                Preset::Ultra => "ultra",
                Preset::Perso => "perso",
            },
            self.render_scale,
            match self.aa {
                AntiAliasing::Taa => "taa",
                AntiAliasing::Smaa => "smaa",
                AntiAliasing::Fxaa => "fxaa",
                AntiAliasing::Aucun => "aucun",
            },
            self.ao,
            self.shadows,
            yn(self.reflections),
            yn(self.light_shafts),
            yn(self.motion_blur),
            yn(self.depth_of_field),
            yn(self.lens),
            match self.look {
                Look::Cinema => "cinema",
                Look::Tropical => "tropical",
                Look::Naturel => "naturel",
            },
            self.brightness,
            yn(self.auto_exposure),
            self.fov,
            yn(self.vsync),
        );
        let _ = std::fs::write(SETTINGS_FILE, text);
    }
}

#[derive(Resource)]
struct Persist(bool);

/// Paramètres d'une ambiance de couleurs.
pub struct LookParams {
    pub tonemapping: Tonemapping,
    /// décalage d'exposition (IL) par rapport à l'exposition de base
    pub ev_offset: f32,
    pub bloom: f32,
    pub saturation: f32,
    pub temperature: f32,
    pub contrast: f32,
    pub highlights_gain: f32,
    /// bloom : seuil (0 = tout diffuse), douceur du seuil, mode additif, étalement
    pub bloom_threshold: f32,
    pub bloom_softness: f32,
    pub bloom_additive: bool,
    pub bloom_low_freq: f32,
    /// halo autour du soleil dans le ciel (diffusion de Mie)
    pub sun_halo: f32,
    /// saturation du ciel (effet de filtre polarisant)
    pub sky_saturation: f32,
    /// force et couverture des ombres des nuages
    pub cloud_strength: f32,
    pub cloud_cover: f32,
    /// horizon moins laiteux (0 = HDRI brut)
    pub dehaze: f32,
    /// soleil doré (0 = blanc)
    pub sun_warm: f32,
    /// éclat des scintillements du soleil sur l'eau (1 = physique)
    pub sparkle: f32,
}

impl LookParams {
    pub fn of(look: Look) -> Self {
        let mut p = match look {
            Look::Cinema => Self {
                tonemapping: Tonemapping::AcesFitted,
                ev_offset: -0.8,
                bloom: 0.4,
                saturation: 1.22,
                temperature: 0.0,
                contrast: 1.25,
                highlights_gain: 1.0,
                // seuil assez bas pour que le ciel près du soleil, les scintillements de l'eau,
                // l'écume et le sable en plein soleil diffusent ; étalement modéré (pas de voile)
                bloom_threshold: 0.85,
                bloom_softness: 0.7,
                bloom_additive: true,
                bloom_low_freq: 0.55,
                sun_halo: 0.5,
                sky_saturation: 1.7,
                cloud_strength: 0.5,
                cloud_cover: 0.35,
                dehaze: 0.8,
                sun_warm: 0.6,
                sparkle: 2.5,
            },
            Look::Tropical => Self {
                tonemapping: Tonemapping::AcesFitted,
                ev_offset: -0.95,
                bloom: 0.35,
                saturation: 1.2,
                temperature: 0.0,
                contrast: 1.15,
                highlights_gain: 1.0,
                bloom_threshold: 0.8,
                bloom_softness: 0.7,
                bloom_additive: true,
                bloom_low_freq: 0.6,
                sun_halo: 0.45,
                sky_saturation: 1.6,
                cloud_strength: 0.55,
                cloud_cover: 0.38,
                dehaze: 0.7,
                sun_warm: 0.35,
                sparkle: 2.0,
            },
            Look::Naturel => Self {
                tonemapping: Tonemapping::TonyMcMapface,
                ev_offset: 0.0,
                bloom: 0.2,
                saturation: 1.0,
                temperature: 0.0,
                contrast: 1.0,
                highlights_gain: 1.0,
                bloom_threshold: 1.0,
                bloom_softness: 0.6,
                bloom_additive: true,
                bloom_low_freq: 0.5,
                sun_halo: 0.0,
                sky_saturation: 1.0,
                cloud_strength: 0.72,
                cloud_cover: 0.5,
                dehaze: 0.0,
                sun_warm: 0.0,
                sparkle: 1.0,
            },
        };
        // réglage fin pour les essais : OCEAN_TUNE="ev=-0.5,sat=1.1,..."
        if let Ok(t) = std::env::var("OCEAN_TUNE") {
            for kv in t.split(',') {
                let Some((k, v)) = kv.split_once('=') else { continue };
                let f: f32 = v.parse().unwrap_or(0.0);
                match k.trim() {
                    "ev" => p.ev_offset = f,
                    "bloom" => p.bloom = f,
                    "sat" => p.saturation = f,
                    "temp" => p.temperature = f,
                    "contrast" => p.contrast = f,
                    "gain" => p.highlights_gain = f,
                    "sky" => p.sky_saturation = f,
                    "clouds" => p.cloud_strength = f,
                    "cover" => p.cloud_cover = f,
                    "thr" => p.bloom_threshold = f,
                    "soft" => p.bloom_softness = f,
                    "add" => p.bloom_additive = f > 0.5,
                    "lf" => p.bloom_low_freq = f,
                    "halo" => p.sun_halo = f,
                    "dehaze" => p.dehaze = f,
                    "warm" => p.sun_warm = f,
                    "sparkle" => p.sparkle = f,
                    "tm" => {
                        p.tonemapping = match v {
                            "aces" => Tonemapping::AcesFitted,
                            "agx" => Tonemapping::AgX,
                            "tony" => Tonemapping::TonyMcMapface,
                            "neutral" => Tonemapping::KhronosPbrNeutral,
                            "blender" => Tonemapping::BlenderFilmic,
                            _ => p.tonemapping,
                        }
                    }
                    _ => {}
                }
            }
        }
        p
    }
}

/// Exposition de base (EV100) avec l'ambiance et la luminosité choisies.
pub fn target_ev100(settings: &GraphicsSettings) -> f32 {
    crate::camera::base_ev100() + LookParams::of(settings.look).ev_offset - settings.brightness
}

// ---------------------------------------------------------------------------------------------
// Application des réglages
// ---------------------------------------------------------------------------------------------

#[derive(Resource)]
struct ExposureCurve(Handle<AutoExposureCompensationCurve>);

fn setup_exposure_curve(mut commands: Commands, mut curves: ResMut<Assets<AutoExposureCompensationCurve>>) {
    // x : luminance moyenne mesurée (log2), y : compensation (IL). Zone morte entre -1,75 et
    // +1,75 ; au-delà on corrige, au plus de 1,25 IL. (valeurs exactement représentables)
    let curve = AutoExposureCompensationCurve::from_curve(LinearSpline::new([
        vec2(-10.0, -8.75),
        vec2(-3.0, -1.75),
        vec2(-1.75, -1.75),
        vec2(1.75, 1.75),
        vec2(3.0, 1.75),
        vec2(10.0, 8.75),
    ]))
    .expect("courbe d'exposition");
    commands.insert_resource(ExposureCurve(curves.add(curve)));
}

fn apply_settings(
    mut commands: Commands,
    settings: Res<GraphicsSettings>,
    persist: Res<Persist>,
    curve: Option<Res<ExposureCurve>>,
    mut ocean: ResMut<Ocean>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    render_scale: Option<ResMut<RenderScale>>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut lights: Query<&mut DirectionalLight>,
    mut cameras: Query<(Entity, &mut Projection, &mut Tonemapping), With<FlyCamera>>,
    mut first: Local<bool>,
) {
    let Some(curve) = curve else { return };
    if !settings.is_changed() && *first {
        return;
    }
    *first = true;
    let s = &*settings;
    if persist.0 {
        s.save();
    }
    let look = LookParams::of(s.look);
    if let Some(mut rs) = render_scale {
        if std::env::var("OCEAN_RENDER_SCALE").is_err() {
            rs.factor = s.render_scale;
        }
    }
    window.present_mode = if s.vsync && std::env::var("OCEAN_NO_VSYNC").is_err() {
        PresentMode::AutoVsync
    } else {
        PresentMode::AutoNoVsync
    };
    // ombres
    shadow_map.size = match s.shadows {
        0 => 1024,
        1 => 2048,
        _ => 4096,
    };
    for mut l in &mut lights {
        l.soft_shadow_size = if s.shadows >= 1 { Some(2.0) } else { None };
        l.contact_shadows_enabled = s.ao >= 1;
    }
    // effets calculés dans nos shaders
    let env_flags = std::env::var("OCEAN_FLAGS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    let mut flags = env_flags;
    if !s.reflections {
        flags |= 1;
    }
    if !s.light_shafts {
        flags |= 2;
    }
    ocean.debug_flags = flags as f32;
    ocean.look = Vec4::new(look.sky_saturation, look.cloud_strength, look.cloud_cover, look.sun_halo);
    ocean.look2 = Vec4::new(look.dehaze, look.sun_warm, look.sparkle, 0.0);

    for (e, mut proj, mut tm) in &mut cameras {
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = s.fov.to_radians();
        }
        *tm = look.tonemapping;
        let mut ec = commands.entity(e);
        ec.insert((
            Bloom {
                intensity: look.bloom,
                low_frequency_boost: look.bloom_low_freq,
                prefilter: BloomPrefilter {
                    threshold: look.bloom_threshold,
                    threshold_softness: look.bloom_softness,
                },
                composite_mode: if look.bloom_additive {
                    BloomCompositeMode::Additive
                } else {
                    BloomCompositeMode::EnergyConserving
                },
                ..Bloom::NATURAL
            },
            ColorGrading {
                global: ColorGradingGlobal {
                    temperature: look.temperature,
                    post_saturation: look.saturation,
                    ..default()
                },
                shadows: ColorGradingSection { contrast: look.contrast, ..default() },
                midtones: ColorGradingSection { contrast: look.contrast, ..default() },
                highlights: ColorGradingSection {
                    contrast: look.contrast,
                    gain: look.highlights_gain,
                    ..default()
                },
            },
        ));
        // anti-crénelage (l'occlusion ambiante exige Msaa::Off : pas de MSAA)
        ec.remove::<(TemporalAntiAliasing, TemporalJitter, MipBias, Smaa, Fxaa)>();
        match s.aa {
            AntiAliasing::Taa => {
                ec.insert((TemporalAntiAliasing::default(), ShadowFilteringMethod::Temporal));
            }
            AntiAliasing::Smaa => {
                ec.insert((Smaa { preset: SmaaPreset::High }, ShadowFilteringMethod::Gaussian));
            }
            AntiAliasing::Fxaa => {
                ec.insert((Fxaa::default(), ShadowFilteringMethod::Gaussian));
            }
            AntiAliasing::Aucun => {
                ec.insert(ShadowFilteringMethod::Gaussian);
            }
        }
        // accentuation : compense le flou du TAA et du rendu à résolution réduite
        ec.insert(ContrastAdaptiveSharpening {
            enabled: s.aa != AntiAliasing::Aucun,
            sharpening_strength: if s.aa == AntiAliasing::Taa { 0.5 } else { 0.3 },
            denoise: false,
        });
        // occlusion ambiante, ombres de contact
        if s.ao > 0 {
            ec.insert((
                ScreenSpaceAmbientOcclusion {
                    quality_level: match s.ao {
                        1 => ScreenSpaceAmbientOcclusionQualityLevel::Medium,
                        2 => ScreenSpaceAmbientOcclusionQualityLevel::High,
                        _ => ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
                    },
                    radius: 1.2,
                    constant_object_thickness: 0.4,
                },
                ContactShadows {
                    linear_steps: if s.ao >= 2 { 16 } else { 10 },
                    thickness: 0.12,
                    length: 0.6,
                },
            ));
        } else {
            ec.remove::<(ScreenSpaceAmbientOcclusion, ContactShadows)>();
        }
        // effets de caméra
        if s.motion_blur {
            ec.insert(MotionBlur {
                shutter_angle: 0.5,
                samples: 4,
            });
        } else {
            ec.remove::<MotionBlur>();
        }
        if s.depth_of_field {
            ec.insert(DepthOfField {
                mode: DepthOfFieldMode::Gaussian,
                focal_distance: 30.0,
                aperture_f_stops: 4.0,
                // capteur plein format : 24 mm
                sensor_height: 0.024,
                max_circle_of_confusion_diameter: 24.0,
                max_depth: f32::INFINITY,
            });
        } else {
            ec.remove::<DepthOfField>();
        }
        if s.lens {
            ec.insert((
                Vignette {
                    intensity: 0.18,
                    radius: 0.9,
                    smoothness: 3.0,
                    ..default()
                },
                ChromaticAberration {
                    intensity: 0.0025,
                    max_samples: 6,
                    ..default()
                },
                LensFlare::default(),
                FilmGrain::default(),
            ));
        } else {
            ec.remove::<(Vignette, ChromaticAberration, LensFlare, FilmGrain)>();
        }
        if s.auto_exposure {
            ec.insert(AutoExposure {
                range: -10.0..=6.0,
                filter: 0.15..=0.92,
                speed_brighten: 1.2,
                speed_darken: 2.0,
                exponential_transition_distance: 1.0,
                compensation_curve: curve.0.clone(),
                ..default()
            });
        } else {
            ec.remove::<AutoExposure>();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct Menu {
    pub open: bool,
    selected: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Row {
    Preset,
    Resolution,
    Aa,
    Ao,
    Shadows,
    Reflections,
    Shafts,
    MotionBlur,
    Dof,
    Lens,
    Look,
    Brightness,
    AutoExposure,
    Fov,
    Vsync,
}

const ROWS: [Row; 15] = [
    Row::Preset,
    Row::Look,
    Row::Brightness,
    Row::Resolution,
    Row::Aa,
    Row::Ao,
    Row::Shadows,
    Row::Reflections,
    Row::Shafts,
    Row::MotionBlur,
    Row::Dof,
    Row::Lens,
    Row::AutoExposure,
    Row::Fov,
    Row::Vsync,
];

impl Row {
    fn label(self) -> &'static str {
        match self {
            Row::Preset => "Qualité",
            Row::Resolution => "Résolution du rendu",
            Row::Aa => "Anti-crénelage",
            Row::Ao => "Occlusion ambiante",
            Row::Shadows => "Ombres",
            Row::Reflections => "Reflets dans l'eau",
            Row::Shafts => "Rayons sous l'eau",
            Row::MotionBlur => "Flou de mouvement",
            Row::Dof => "Profondeur de champ",
            Row::Lens => "Effets d'objectif",
            Row::Look => "Ambiance",
            Row::Brightness => "Luminosité",
            Row::AutoExposure => "Exposition automatique",
            Row::Fov => "Champ de vision",
            Row::Vsync => "Synchro verticale",
        }
    }

    fn value(self, s: &GraphicsSettings) -> String {
        let yn = |b: bool| if b { "Oui" } else { "Non" }.to_string();
        match self {
            Row::Preset => s.preset.name().to_string(),
            Row::Resolution => format!("{:.0} %", s.render_scale * 100.0),
            Row::Aa => match s.aa {
                AntiAliasing::Taa => "Temporel (TAA)",
                AntiAliasing::Smaa => "SMAA",
                AntiAliasing::Fxaa => "FXAA",
                AntiAliasing::Aucun => "Aucun",
            }
            .to_string(),
            Row::Ao => ["Non", "Moyenne", "Haute", "Ultra"][s.ao as usize].to_string(),
            Row::Shadows => ["Basses", "Moyennes", "Hautes"][s.shadows as usize].to_string(),
            Row::Reflections => yn(s.reflections),
            Row::Shafts => yn(s.light_shafts),
            Row::MotionBlur => yn(s.motion_blur),
            Row::Dof => yn(s.depth_of_field),
            Row::Lens => yn(s.lens),
            Row::Look => match s.look {
                Look::Cinema => "Cinéma (façon UE5)",
                Look::Tropical => "Tropicale",
                Look::Naturel => "Naturelle",
            }
            .to_string(),
            Row::Brightness => format!("{:+.2} IL", s.brightness),
            Row::AutoExposure => yn(s.auto_exposure),
            Row::Fov => format!("{:.0}°", s.fov),
            Row::Vsync => yn(s.vsync),
        }
    }

    /// Change la valeur (delta = -1 ou +1).
    fn change(self, s: &mut GraphicsSettings, delta: i32) {
        let cycle = |v: i32, n: i32| ((v + delta) % n + n) % n;
        let quality = !matches!(self, Row::Preset | Row::Look | Row::Brightness | Row::AutoExposure | Row::Fov | Row::Vsync);
        match self {
            Row::Preset => {
                let all = [Preset::Bas, Preset::Moyen, Preset::Eleve, Preset::Ultra];
                let i = all.iter().position(|p| *p == s.preset).unwrap_or(2) as i32;
                let j = (i + delta).clamp(0, 3) as usize;
                s.apply_preset(all[j]);
            }
            Row::Resolution => s.render_scale = (s.render_scale + 0.05 * delta as f32).clamp(0.4, 1.0),
            Row::Aa => {
                let all = [AntiAliasing::Taa, AntiAliasing::Smaa, AntiAliasing::Fxaa, AntiAliasing::Aucun];
                let i = all.iter().position(|a| *a == s.aa).unwrap_or(0) as i32;
                s.aa = all[cycle(i, 4) as usize];
            }
            Row::Ao => s.ao = cycle(s.ao as i32, 4) as u8,
            Row::Shadows => s.shadows = cycle(s.shadows as i32, 3) as u8,
            Row::Reflections => s.reflections = !s.reflections,
            Row::Shafts => s.light_shafts = !s.light_shafts,
            Row::MotionBlur => s.motion_blur = !s.motion_blur,
            Row::Dof => s.depth_of_field = !s.depth_of_field,
            Row::Lens => s.lens = !s.lens,
            Row::Look => {
                let all = [Look::Cinema, Look::Tropical, Look::Naturel];
                let i = all.iter().position(|l| *l == s.look).unwrap_or(0) as i32;
                s.look = all[cycle(i, 3) as usize];
            }
            Row::Brightness => s.brightness = (s.brightness + 0.25 * delta as f32).clamp(-2.0, 2.0),
            Row::AutoExposure => s.auto_exposure = !s.auto_exposure,
            Row::Fov => s.fov = (s.fov + 5.0 * delta as f32).clamp(30.0, 90.0),
            Row::Vsync => s.vsync = !s.vsync,
        }
        if quality {
            s.preset = Preset::Perso;
        }
    }
}

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct RowLine(usize);

#[derive(Component)]
struct RowValue(Row);

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum Action {
    Resume,
    Quit,
}

const PANEL: Color = Color::srgba(0.04, 0.07, 0.09, 0.92);
const ROW_SELECTED: Color = Color::srgba(0.2, 0.55, 0.6, 0.35);
const BUTTON: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);
const BUTTON_HOVER: Color = Color::srgba(0.3, 0.75, 0.8, 0.45);

fn spawn_menu(mut commands: Commands, assets: Res<AssetServer>, settings: Res<GraphicsSettings>) {
    let font: Handle<Font> = assets.load("fonts/FiraSans-Regular.ttf");
    let bold: Handle<Font> = assets.load("fonts/FiraSans-Medium.ttf");
    let text = |s: &str, size: f32, f: &Handle<Font>, alpha: f32| {
        (
            Text::new(s),
            TextFont {
                font: f.clone().into(),
                font_size: FontSize::Px(size),
                ..default()
            },
            TextColor(Color::srgba(1.0, 1.0, 1.0, alpha)),
        )
    };
    commands
        .spawn((
            MenuRoot,
            Node {
                position_type: PositionType::Absolute,
                width: percent(100),
                height: percent(100),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.35)),
            GlobalZIndex(10),
            Visibility::Hidden,
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(px(22)),
                    row_gap: px(4),
                    min_width: px(560),
                    ..default()
                },
                BackgroundColor(PANEL),
            ))
            .with_children(|panel| {
                panel.spawn((text("Réglages graphiques", 24.0, &bold, 1.0), Node { margin: UiRect::bottom(px(10)), ..default() }));
                for (i, row) in ROWS.iter().enumerate() {
                    panel
                        .spawn((
                            RowLine(i),
                            Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::SpaceBetween,
                                padding: UiRect::axes(px(8), px(3)),
                                ..default()
                            },
                            BackgroundColor(Color::NONE),
                        ))
                        .with_children(|line| {
                            line.spawn(text(row.label(), 16.0, &font, 0.9));
                            line.spawn(Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: px(8),
                                ..default()
                            })
                            .with_children(|ctl| {
                                for (delta, glyph) in [(-1, "<"), (1, ">")] {
                                    if delta == 1 {
                                        ctl.spawn((
                                            RowValue(*row),
                                            Node { min_width: px(150), justify_content: JustifyContent::Center, ..default() },
                                        ))
                                        .with_children(|v| {
                                            v.spawn(text(&row.value(&settings), 16.0, &bold, 1.0));
                                        });
                                    }
                                    let r = *row;
                                    ctl.spawn((
                                        Button,
                                        Hovered::default(),
                                        Node {
                                            width: px(28),
                                            height: px(24),
                                            justify_content: JustifyContent::Center,
                                            align_items: AlignItems::Center,
                                            ..default()
                                        },
                                        BackgroundColor(BUTTON),
                                    ))
                                    .observe(
                                        move |_: On<Activate>, mut settings: ResMut<GraphicsSettings>, mut menu: ResMut<Menu>| {
                                            r.change(&mut settings, delta);
                                            if let Some(k) = ROWS.iter().position(|x| *x == r) {
                                                menu.selected = k;
                                            }
                                        },
                                    )
                                    .with_children(|b| {
                                        b.spawn(text(glyph, 16.0, &bold, 1.0));
                                    });
                                }
                            });
                        });
                }
                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        margin: UiRect::top(px(14)),
                        ..default()
                    })
                    .with_children(|bottom| {
                        bottom.spawn(text("Échap : fermer   ·   ↑ ↓ : choisir   ·   ← → : modifier", 13.0, &font, 0.6));
                        bottom
                            .spawn(Node { flex_direction: FlexDirection::Row, column_gap: px(10), ..default() })
                            .with_children(|btns| {
                                for (action, label) in [(Action::Resume, "Reprendre"), (Action::Quit, "Quitter")] {
                                    btns.spawn((
                                        Button,
                                        Hovered::default(),
                                        action,
                                        Node { padding: UiRect::axes(px(14), px(6)), ..default() },
                                        BackgroundColor(BUTTON),
                                    ))
                                    .observe(
                                        move |_: On<Activate>,
                                              mut menu: ResMut<Menu>,
                                              mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
                                              mut exit: MessageWriter<AppExit>| match action {
                                            Action::Resume => set_open(&mut menu, false, &mut cursor),
                                            Action::Quit => {
                                                exit.write(AppExit::Success);
                                            }
                                        },
                                    )
                                    .with_children(|b| {
                                        b.spawn(text(label, 16.0, &bold, 1.0));
                                    });
                                }
                            });
                    });
            });
        });
}

fn set_open(menu: &mut Menu, open: bool, cursor: &mut CursorOptions) {
    menu.open = open;
    if open {
        cursor.visible = true;
        cursor.grab_mode = CursorGrabMode::None;
    }
}

fn toggle_menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut menu: ResMut<Menu>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut root: Query<&mut Visibility, With<MenuRoot>>,
    mut opened_once: Local<bool>,
) {
    // OCEAN_MENU_OPEN : menu ouvert au démarrage (captures d'écran du menu)
    if !*opened_once && std::env::var("OCEAN_MENU_OPEN").is_ok() {
        *opened_once = true;
        set_open(&mut menu, true, &mut cursor);
    }
    if keys.just_pressed(KeyCode::Escape) {
        let open = !menu.open;
        set_open(&mut menu, open, &mut cursor);
    }
    if menu.is_changed() {
        for mut v in &mut root {
            *v = if menu.open { Visibility::Inherited } else { Visibility::Hidden };
        }
    }
}

fn menu_keyboard(keys: Res<ButtonInput<KeyCode>>, mut menu: ResMut<Menu>, mut settings: ResMut<GraphicsSettings>) {
    if !menu.open {
        return;
    }
    let n = ROWS.len();
    if keys.just_pressed(KeyCode::ArrowDown) {
        menu.selected = (menu.selected + 1) % n;
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        menu.selected = (menu.selected + n - 1) % n;
    }
    let row = ROWS[menu.selected];
    if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::Enter) {
        row.change(&mut settings, 1);
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        row.change(&mut settings, -1);
    }
}

/// Couleur des boutons selon le survol et l'appui.
fn menu_buttons(mut buttons: Query<(&Hovered, Has<Pressed>, &mut BackgroundColor), With<Button>>) {
    for (hovered, pressed, mut c) in &mut buttons {
        let target = if pressed || hovered.get() { BUTTON_HOVER } else { BUTTON };
        if c.0 != target {
            c.0 = target;
        }
    }
}

fn refresh_menu(
    menu: Res<Menu>,
    settings: Res<GraphicsSettings>,
    values: Query<(&RowValue, &Children)>,
    mut texts: Query<&mut Text>,
    mut lines: Query<(&RowLine, &mut BackgroundColor)>,
) {
    if settings.is_changed() {
        for (rv, children) in &values {
            for c in children.iter() {
                if let Ok(mut t) = texts.get_mut(c) {
                    **t = rv.0.value(&settings);
                }
            }
        }
    }
    if menu.is_changed() {
        for (l, mut bg) in &mut lines {
            bg.0 = if l.0 == menu.selected { ROW_SELECTED } else { Color::NONE };
        }
    }
}
