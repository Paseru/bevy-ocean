//! Résolution de rendu réglable : la scène 3D est rendue dans une image (éventuellement plus
//! petite que la fenêtre), puis affichée en plein écran. Sur un écran Retina, rendre à ~70 %
//! de la résolution physique est presque invisible et double les performances.
//!
//! Touches : F2 / F3 pour diminuer / augmenter la résolution de rendu.

use bevy::{
    camera::RenderTarget,
    image::{ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::TextureFormat,
    ui::{IsDefaultUiCamera, widget::NodeImageMode},
    window::{PrimaryWindow, WindowResized},
};

use crate::camera::FlyCamera;
use crate::settings::GraphicsSettings;

pub struct RenderScalePlugin;

impl Plugin for RenderScalePlugin {
    fn build(&self, app: &mut App) {
        // en mode capture automatique, c'est le module de capture qui choisit la cible
        if std::env::var("OCEAN_SHOTS").is_ok() {
            return;
        }
        let factor = std::env::var("OCEAN_RENDER_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.72f32);
        app.insert_resource(RenderScale {
            factor,
            image: None,
            size: UVec2::ZERO,
        })
        .add_systems(Startup, setup_render_scale)
        .add_systems(Update, (change_scale, resize_target).chain());
    }
}

/// Facteur de résolution du rendu 3D (1.0 = résolution physique de la fenêtre).
#[derive(Resource)]
pub struct RenderScale {
    pub factor: f32,
    image: Option<Handle<Image>>,
    size: UVec2,
}

#[derive(Component)]
struct SceneView;

fn make_target(size: UVec2) -> Image {
    let mut image = Image::new_target_texture(
        size.x.max(1),
        size.y.max(1),
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

fn target_size(window: &Window, factor: f32) -> UVec2 {
    let w = (window.resolution.physical_width() as f32 * factor).round() as u32;
    let h = (window.resolution.physical_height() as f32 * factor).round() as u32;
    UVec2::new(w.max(16), h.max(16))
}

fn setup_render_scale(
    mut commands: Commands,
    mut scale: ResMut<RenderScale>,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
    cameras: Query<Entity, With<FlyCamera>>,
) {
    let size = target_size(&window, scale.factor);
    let handle = images.add(make_target(size));
    for e in &cameras {
        commands.entity(e).insert(RenderTarget::Image(handle.clone().into()));
    }
    // caméra de la fenêtre : affiche l'image de la scène et l'interface
    commands.spawn((
        Name::new("Caméra de l'écran"),
        Camera2d,
        Camera {
            order: 1,
            ..default()
        },
        IsDefaultUiCamera,
    ));
    commands.spawn((
        SceneView,
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: percent(100),
            height: percent(100),
            ..default()
        },
        ImageNode::new(handle.clone()).with_mode(NodeImageMode::Stretch),
        GlobalZIndex(-1),
    ));
    scale.image = Some(handle);
    scale.size = size;
}

/// F2 / F3 : raccourcis pour le réglage "Résolution du rendu" (appliqué par `settings.rs`).
fn change_scale(keys: Res<ButtonInput<KeyCode>>, mut settings: ResMut<GraphicsSettings>) {
    let mut f = settings.render_scale;
    if keys.just_pressed(KeyCode::F2) {
        f = (f - 0.05).max(0.4);
    }
    if keys.just_pressed(KeyCode::F3) {
        f = (f + 0.05).min(1.0);
    }
    if f != settings.render_scale {
        settings.render_scale = f;
        settings.preset = crate::settings::Preset::Perso;
    }
}

fn resize_target(
    mut scale: ResMut<RenderScale>,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window, With<PrimaryWindow>>,
    mut resized: MessageReader<WindowResized>,
    mut cameras: Query<&mut RenderTarget, With<FlyCamera>>,
    mut views: Query<&mut ImageNode, With<SceneView>>,
) {
    resized.clear();
    let size = target_size(&window, scale.factor);
    if size == scale.size {
        return;
    }
    let handle = images.add(make_target(size));
    for mut target in &mut cameras {
        *target = RenderTarget::Image(handle.clone().into());
    }
    for mut view in &mut views {
        view.image = handle.clone();
    }
    if let Some(old) = scale.image.replace(handle) {
        images.remove(&old);
    }
    scale.size = size;
}
