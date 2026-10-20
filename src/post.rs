//! Effets d'objectif écrits à la main (Bevy n'en fournit pas d'équivalent) :
//! - reflets d'objectif (fantômes, halo, voile) quand le soleil est dans le champ ;
//! - grain de film, appliqué en tout dernier (après l'anti-crénelage).

use bevy::{
    anti_alias::{contrast_adaptive_sharpening::cas, fxaa::fxaa, smaa::smaa},
    core_pipeline::{
        Core3dSystems,
        fullscreen_material::{FullscreenMaterial, FullscreenMaterialPlugin},
        tonemapping::tonemapping,
    },
    ecs::{schedule::ScheduleConfigs, system::BoxedSystem},
    prelude::*,
    render::{RenderApp, extract_component::ExtractComponent, render_resource::ShaderType},
    shader::ShaderRef,
    transform::TransformSystems,
    window::PrimaryWindow,
};

use crate::camera::FlyCamera;
use crate::ocean::Ocean;
use crate::sky_data;

pub struct PostPlugin;

impl Plugin for PostPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FullscreenMaterialPlugin::<LensFlare>::default(),
            FullscreenMaterialPlugin::<FilmGrain>::default(),
        ))
        .add_systems(PostUpdate, update_lens_effects.after(TransformSystems::Propagate));
    }
}

/// Reflets d'objectif, appliqués juste après le rendu des tons (et avant l'anti-crénelage) :
/// l'ordre est explicite pour ne pas s'intercaler au hasard entre les autres effets.
#[derive(Component, ExtractComponent, Clone, Copy, ShaderType)]
#[extract_app(RenderApp)]
pub struct LensFlare {
    /// xy : position du soleil à l'écran (0..1), z : atténuation près des bords, w : intensité
    pub sun: Vec4,
    /// x : rapport largeur/hauteur, y : temps, z : seuil de luminance du disque solaire
    /// (après le rendu des tons, le soleil est blanc saturé), w : libre
    pub params: Vec4,
}

impl Default for LensFlare {
    fn default() -> Self {
        Self {
            sun: Vec4::new(0.5, 0.5, 0.0, 1.0),
            params: Vec4::new(16.0 / 9.0, 0.0, 0.9, 0.0),
        }
    }
}

impl FullscreenMaterial for LensFlare {
    fn fragment_shader() -> ShaderRef {
        "shaders/lens_flare.wesl".into()
    }

    fn schedule_configs(system: ScheduleConfigs<BoxedSystem>) -> ScheduleConfigs<BoxedSystem> {
        system
            .in_set(Core3dSystems::PostProcess)
            .after(tonemapping)
            .before(smaa)
            .before(fxaa)
    }
}

/// Grain de film (après l'anti-crénelage et l'accentuation).
#[derive(Component, ExtractComponent, Clone, Copy, ShaderType)]
#[extract_app(RenderApp)]
pub struct FilmGrain {
    /// x : intensité, y : temps, z : taille du grain (pixels), w : libre
    pub params: Vec4,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self {
            params: Vec4::new(0.025, 0.0, 1.0, 0.0),
        }
    }
}

impl FullscreenMaterial for FilmGrain {
    fn fragment_shader() -> ShaderRef {
        "shaders/film_grain.wesl".into()
    }

    fn schedule_configs(system: ScheduleConfigs<BoxedSystem>) -> ScheduleConfigs<BoxedSystem> {
        system.in_set(Core3dSystems::PostProcess).after(cas)
    }
}

/// Position du soleil à l'écran (réfracté si la caméra est sous l'eau) et temps.
fn update_lens_effects(
    time: Res<Time>,
    ocean: Res<Ocean>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut cams: Query<
        (&GlobalTransform, &Projection, Option<&mut LensFlare>, Option<&mut FilmGrain>),
        With<FlyCamera>,
    >,
) {
    let t = time.elapsed_secs_wrapped();
    let aspect = window.resolution.width() / window.resolution.height().max(1.0);
    for (gt, proj, flare, grain) in &mut cams {
        if let Some(mut grain) = grain {
            grain.params.y = t;
        }
        let Some(mut flare) = flare else {
            continue;
        };
        let mut sun = Vec3::from_array(sky_data::SUN_DIRECTION).normalize();
        if ocean.camera_underwater {
            // vu de sous l'eau, le soleil paraît plus haut (réfraction à la surface)
            let horiz = Vec3::new(sun.x, 0.0, sun.z).normalize_or_zero();
            let sin_air = (1.0 - sun.y * sun.y).max(0.0).sqrt();
            let sin_w = sin_air / 1.333;
            sun = (horiz * sin_w + Vec3::Y * (1.0 - sin_w * sin_w).sqrt()).normalize();
        }
        let fov = match proj {
            Projection::Perspective(p) => p.fov,
            _ => 1.0,
        };
        let v = gt.affine().inverse().transform_vector3(sun);
        let tan_half = (fov * 0.5).tan();
        let mut fade = 0.0;
        let mut uv = Vec2::splat(-10.0);
        if v.z < -1e-3 {
            let ndc = Vec2::new(v.x / -v.z / (tan_half * aspect), v.y / -v.z / tan_half);
            uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            let edge = ndc.abs().max_element();
            fade = (1.0 - ((edge - 0.85) / 0.25).clamp(0.0, 1.0)).powi(2);
        }
        flare.sun = Vec4::new(uv.x, uv.y, fade, 1.0);
        flare.params = Vec4::new(aspect, t, 0.9, 0.0);
    }
}
