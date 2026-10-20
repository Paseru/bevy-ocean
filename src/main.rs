//! Océan tropical : eau turquoise, fond de sable blanc à 15 m, île plantée de cocotiers
//! entourée d'un récif où la houle déferle, patates de corail.
//!
//! Contrôles (clavier AZERTY ou QWERTY, ce sont les positions physiques des touches) :
//! - ZQSD (WASD en QWERTY) : se déplacer
//! - Souris : regarder autour (clic gauche pour capturer la souris, Échap pour la libérer)
//! - Espace / E : monter, Ctrl / A : descendre
//! - Maj : aller plus vite, molette : régler la vitesse
//! - H : afficher / masquer l'aide, F11 : plein écran
//! - F2 / F3 : baisser / augmenter la résolution du rendu 3D
//! - Échap : menu des réglages graphiques (qualité, anti-crénelage, ambiance, luminosité...),
//!   sauvegardés dans `reglages.txt`

mod camera;
mod capture;
mod environment;
mod hud;
mod noise;
mod ocean;
mod particles;
mod post;
mod render_scale;
mod settings;
mod sky_data;
mod terrain;
mod vegetation;

use bevy::prelude::*;
use bevy::window::{PresentMode, WindowResolution};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Océan turquoise".into(),
                // en mode capture automatique, le rendu se fait hors écran
                visible: std::env::var("OCEAN_SHOTS").is_err(),
                resolution: WindowResolution::new(1600, 900),
                present_mode: if std::env::var("OCEAN_NO_VSYNC").is_ok() {
                    PresentMode::AutoNoVsync
                } else {
                    PresentMode::AutoVsync
                },
                ..default()
            }),
            ..default()
        }))
        .add_plugins((
            environment::EnvironmentPlugin,
            ocean::OceanPlugin,
            terrain::TerrainPlugin,
            vegetation::VegetationPlugin,
            particles::ParticlesPlugin,
            camera::FlyCameraPlugin,
            hud::HudPlugin,
            capture::CapturePlugin,
            render_scale::RenderScalePlugin,
            post::PostPlugin,
            settings::SettingsPlugin,
        ))
        .run();
}
