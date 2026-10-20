//! Caméra libre : ZQSD (positions physiques, donc WASD sur un clavier QWERTY), souris pour
//! regarder, Espace/E pour monter, Ctrl/A pour descendre, Maj pour accélérer.
//!
//! Les effets de rendu portés par la caméra (anti-crénelage, occlusion ambiante, flou de
//! mouvement, profondeur de champ...) sont ajoutés selon les réglages (voir `settings.rs`).
//! Ici : déplacement, exposition (adaptation sous l'eau), mise au point automatique.

use bevy::{
    camera::{Exposure, Hdr},
    core_pipeline::{prepass::DepthPrepass, tonemapping::Tonemapping},
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    pbr::ScreenSpaceTransmission,
    post_process::{bloom::Bloom, dof::DepthOfField},
    prelude::*,
    render::view::Msaa,
    window::{CursorGrabMode, CursorOptions, MonitorSelection, PrimaryWindow, WindowMode},
};

use crate::ocean::Ocean;
use crate::settings::{GraphicsSettings, Menu, target_ev100};
use crate::terrain::{self, Obstacles};

pub struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, spawn_camera).add_systems(
            Update,
            (grab_cursor, toggle_fullscreen, fly_camera, adapt_exposure, autofocus)
                .chain()
                .after(crate::settings::SettingsSystems),
        );
    }
}

/// État de la caméra libre.
#[derive(Component)]
pub struct FlyCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub velocity: Vec3,
    pub sensitivity: f32,
}

/// Point de vue de départ : au-dessus des 15 m d'eau, face à l'île.
pub const START_POSITION: Vec3 = Vec3::new(42.0, 2.6, 395.0);
pub const START_YAW: f32 = 0.10;
pub const START_PITCH: f32 = -0.12;

/// Exposition de base (EV100) en plein soleil, avant l'ambiance choisie et la luminosité
/// (voir `settings.rs`).
pub const BASE_EV100: f32 = 14.5;

/// Champ de vision vertical par défaut (degrés), modifiable dans les réglages.
pub const DEFAULT_FOV_DEG: f32 = 45.0;

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        (
            Name::new("Caméra"),
            Camera3d::default(),
            Hdr,
            Projection::Perspective(PerspectiveProjection {
                fov: DEFAULT_FOV_DEG.to_radians(),
                near: 0.05,
                far: 100_000.0,
                ..default()
            }),
            Transform::from_translation(START_POSITION)
                .with_rotation(Quat::from_euler(EulerRot::YXZ, START_YAW, START_PITCH, 0.0)),
            FlyCamera {
                yaw: START_YAW,
                pitch: START_PITCH,
                speed: 6.0,
                velocity: Vec3::ZERO,
                sensitivity: 0.0022,
            },
        ),
        (
            Exposure { ev100: base_ev100() },
            tonemapping_from_env(),
            Bloom::NATURAL,
            // l'occlusion ambiante exige Msaa::Off : l'anti-crénelage est fait autrement
            // (temporel, SMAA ou FXAA selon les réglages)
            Msaa::Off,
            DepthPrepass,
            ScreenSpaceTransmission::default(),
        ),
    ));
}

/// Opérateur de rendu des tons (réglable avec `OCEAN_TONEMAP` pour les essais).
fn tonemapping_from_env() -> Tonemapping {
    match std::env::var("OCEAN_TONEMAP").as_deref() {
        Ok("agx") => Tonemapping::AgX,
        Ok("aces") => Tonemapping::AcesFitted,
        Ok("tony") => Tonemapping::TonyMcMapface,
        Ok("blender") => Tonemapping::BlenderFilmic,
        Ok("neutral") => Tonemapping::KhronosPbrNeutral,
        Ok("reinhard") => Tonemapping::ReinhardLuminance,
        Ok("sbdt") => Tonemapping::SomewhatBoringDisplayTransform,
        _ => Tonemapping::TonyMcMapface,
    }
}

pub fn base_ev100() -> f32 {
    std::env::var("OCEAN_EV").ok().and_then(|v| v.parse().ok()).unwrap_or(BASE_EV100)
}

/// Clic : on capture la souris pour regarder autour (sauf si le menu est ouvert ;
/// Échap ouvre le menu et libère la souris, voir `settings.rs`).
fn grab_cursor(
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mouse: Res<ButtonInput<MouseButton>>,
    menu: Res<Menu>,
) {
    if mouse.just_pressed(MouseButton::Left) && !menu.open {
        cursor.visible = false;
        cursor.grab_mode = CursorGrabMode::Locked;
    }
}

fn toggle_fullscreen(mut window: Single<&mut Window, With<PrimaryWindow>>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::F11) {
        window.mode = match window.mode {
            WindowMode::Windowed => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
            _ => WindowMode::Windowed,
        };
    }
}

fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    obstacles: Res<Obstacles>,
    menu: Res<Menu>,
    mut query: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let dt = time.delta_secs().min(0.1);
    // menu ouvert : la caméra ralentit et s'arrête
    let keys_active = !menu.open;
    for (mut transform, mut cam) in &mut query {
        // orientation
        if cursor.grab_mode != CursorGrabMode::None {
            let d = mouse_motion.delta;
            cam.yaw -= d.x * cam.sensitivity;
            cam.pitch = (cam.pitch - d.y * cam.sensitivity).clamp(-1.54, 1.54);
        }
        transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);

        // vitesse réglable à la molette
        let lines = match scroll.unit {
            MouseScrollUnit::Line => scroll.delta.y,
            MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
        };
        if lines != 0.0 && keys_active {
            cam.speed = (cam.speed * 1.15f32.powf(lines)).clamp(0.5, 400.0);
        }

        // déplacement (touches physiques : W=Z, A=Q en AZERTY)
        let forward = transform.forward().as_vec3();
        let right = transform.right().as_vec3();
        let mut wish = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
            wish += forward;
        }
        if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
            wish -= forward;
        }
        if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
            wish += right;
        }
        if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
            wish -= right;
        }
        if keys.pressed(KeyCode::Space) || keys.pressed(KeyCode::KeyE) {
            wish += Vec3::Y;
        }
        if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::KeyQ) || keys.pressed(KeyCode::KeyC) {
            wish -= Vec3::Y;
        }
        let boost = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            4.0
        } else {
            1.0
        };
        if !keys_active {
            wish = Vec3::ZERO;
        }
        let target = wish.normalize_or_zero() * cam.speed * boost;
        // accélération douce
        let k = 1.0 - (-dt * 6.0).exp();
        cam.velocity = cam.velocity.lerp(target, k);
        let mut pos = transform.translation + cam.velocity * dt;

        // collisions : le sol (île, fond marin), les coraux, les troncs
        let ground = terrain::height_at(pos.x, pos.z) + 0.45;
        if pos.y < ground {
            pos.y = ground;
            cam.velocity.y = cam.velocity.y.max(0.0);
        }
        for &(c, r) in &obstacles.spheres {
            let d = pos - c;
            let min = r + 0.35;
            let len = d.length();
            if len < min && len > 1e-4 {
                pos = c + d / len * min;
            }
        }
        pos.y = pos.y.min(1500.0);
        transform.translation = pos;
    }
}

/// Exposition : l'œil s'adapte quand on plonge (il fait plus sombre sous l'eau).
fn adapt_exposure(
    time: Res<Time>,
    ocean: Res<Ocean>,
    settings: Res<GraphicsSettings>,
    mut last: Local<Option<f32>>,
    mut query: Query<(&Transform, &mut Exposure), With<FlyCamera>>,
) {
    for (t, mut exposure) in &mut query {
        let depth = (ocean.surface_at_camera - t.translation.y).max(0.0);
        let base = target_ev100(&settings);
        // changement de réglage : on applique tout de suite
        if *last != Some(base) {
            exposure.ev100 = base;
            *last = Some(base);
        }
        let target = if ocean.camera_underwater {
            base - 0.4 - (depth / 10.0).min(0.6)
        } else {
            base
        };
        let k = 1.0 - (-time.delta_secs() * 2.5).exp();
        exposure.ev100 += (target - exposure.ev100) * k;
    }
}

/// Distance du premier obstacle dans l'axe de la caméra (relief, surface de l'eau, coraux).
fn focus_distance(origin: Vec3, dir: Vec3, ocean: &Ocean, obstacles: &Obstacles) -> f32 {
    let underwater = ocean.camera_underwater;
    let mut t = 0.2f32;
    let mut hit = 2000.0f32;
    for _ in 0..90 {
        let p = origin + dir * t;
        let ground = terrain::height_at(p.x, p.z);
        let water = ocean.surface_height(p.x, p.z);
        let blocked = p.y < ground || (!underwater && p.y < water) || (underwater && p.y > water);
        if blocked {
            hit = t;
            break;
        }
        // pas croissant : précis près de la caméra, rapide au loin
        t += 0.12 + t * 0.06;
        if t > 2000.0 {
            break;
        }
    }
    // coraux et rochers (sphères englobantes)
    for &(c, r) in &obstacles.spheres {
        let oc = origin - c;
        let b = oc.dot(dir);
        let disc = b * b - (oc.length_squared() - r * r * 0.8);
        if disc > 0.0 {
            let s = -b - disc.sqrt();
            if s > 0.05 && s < hit {
                hit = s;
            }
        }
    }
    hit
}

/// Mise au point automatique : on vise ce qui est au centre de l'image.
fn autofocus(
    time: Res<Time>,
    ocean: Res<Ocean>,
    obstacles: Res<Obstacles>,
    mut last: Local<Vec3>,
    mut query: Query<(&Transform, &mut DepthOfField), With<FlyCamera>>,
) {
    for (t, mut dof) in &mut query {
        let target = focus_distance(t.translation, t.forward().as_vec3(), &ocean, &obstacles).clamp(0.3, 2000.0);
        // la mise au point glisse (en échelle logarithmique) comme un autofocus ;
        // elle se fait d'un coup si la caméra a été téléportée
        let jumped = t.translation.distance(*last) > 5.0;
        *last = t.translation;
        let k = if jumped { 1.0 } else { 1.0 - (-time.delta_secs() * 4.0).exp() };
        let f = dof.focal_distance.max(0.3).ln();
        dof.focal_distance = (f + (target.ln() - f) * k).exp();
    }
}
