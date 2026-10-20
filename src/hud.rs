//! Aide à l'écran et compteur d'images par seconde.

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};

use crate::ocean::Ocean;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, spawn_hud)
            .add_systems(Update, (toggle_hud, update_hud, log_fps));
    }
}

#[derive(Component)]
struct HudRoot;

#[derive(Component)]
struct HudStats;

const HELP: &str = "ZQSD : se déplacer   ·   Souris : regarder (clic pour capturer la souris)\n\
Espace / E : monter   ·   Ctrl / A : descendre   ·   Maj : plus vite   ·   Molette : vitesse\n\
Échap : réglages graphiques   ·   F11 : plein écran   ·   F2 / F3 : résolution du rendu   ·   H : masquer l'aide";

fn spawn_hud(mut commands: Commands, assets: Res<AssetServer>) {
    let hidden = std::env::var("OCEAN_HIDE_HUD").is_ok();
    let font: Handle<Font> = assets.load("fonts/FiraSans-Regular.ttf");
    commands
        .spawn((
            HudRoot,
            Node {
                position_type: PositionType::Absolute,
                left: px(14),
                bottom: px(12),
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
                ..default()
            },
            if hidden { Visibility::Hidden } else { Visibility::Inherited },
        ))
        .with_children(|p| {
            p.spawn((
                HudStats,
                Text::new(""),
                TextFont {
                    font: font.clone().into(),
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.9)),
                TextShadow::default(),
            ));
            p.spawn((
                Text::new(HELP),
                TextFont {
                    font: font.clone().into(),
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
                TextShadow::default(),
            ));
        });
}

fn toggle_hud(keys: Res<ButtonInput<KeyCode>>, mut q: Query<&mut Visibility, With<HudRoot>>) {
    if keys.just_pressed(KeyCode::KeyH) || keys.just_pressed(KeyCode::F1) {
        for mut v in &mut q {
            *v = match *v {
                Visibility::Hidden => Visibility::Inherited,
                _ => Visibility::Hidden,
            };
        }
    }
}

fn update_hud(
    diagnostics: Res<DiagnosticsStore>,
    ocean: Res<Ocean>,
    camera: Query<&Transform, With<Camera3d>>,
    mut text: Query<&mut Text, With<HudStats>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);
    let Some(cam) = camera.iter().next() else {
        return;
    };
    let depth = ocean.surface_at_camera - cam.translation.y;
    let where_ = if ocean.camera_underwater {
        format!("sous l'eau, profondeur {:.1} m", depth)
    } else {
        format!("altitude {:.1} m", -depth)
    };
    for mut t in &mut text {
        **t = format!("{fps:.0} img/s   ·   {where_}");
    }
}

/// Journalise les images par seconde toutes les 2 s si `OCEAN_FPS_LOG` est défini.
fn log_fps(diagnostics: Res<DiagnosticsStore>, time: Res<Time<Real>>, mut next: Local<f32>) {
    if std::env::var("OCEAN_FPS_LOG").is_err() {
        return;
    }
    let t = time.elapsed_secs();
    if t < *next {
        return;
    }
    *next = t + 2.0;
    if let Some(d) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FRAME_TIME) {
        if let Some(ms) = d.smoothed() {
            info!("temps d'image : {:.2} ms ({:.0} img/s)", ms, 1000.0 / ms);
        }
    }
}
