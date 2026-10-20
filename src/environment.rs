//! Ciel (HDRI Poly Haven convertie en cubemap), soleil et dôme céleste.

use bevy::{
    camera::visibility::NoFrustumCulling,
    light::{CascadeShadowConfigBuilder, NotShadowCaster},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError},
    shader::ShaderRef,
    transform::TransformSystems,
};

use crate::ocean::{Ocean, OceanMaterial, OceanSystems, OceanTextures, OceanUniforms, sync_ocean_uniforms};
use crate::sky_data;

pub struct EnvironmentPlugin;

impl Plugin for EnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default())
            .add_systems(Startup, (spawn_sun, spawn_sky_dome))
            .add_systems(
                PostUpdate,
                (
                    sync_ocean_uniforms::<SkyMaterial>.in_set(OceanSystems::Sync),
                    follow_camera.before(TransformSystems::Propagate),
                ),
            );
    }
}

/// Le soleil : lumière directionnelle alignée sur le soleil de l'HDRI.
fn spawn_sun(mut commands: Commands) {
    let dir = Vec3::from_array(sky_data::SUN_DIRECTION).normalize();
    let c = sky_data::SUN_COLOR;
    commands.spawn((
        Name::new("Soleil"),
        DirectionalLight {
            color: Color::linear_rgb(c[0], c[1], c[2]),
            illuminance: sky_data::SUN_ILLUMINANCE,
            shadow_maps_enabled: true,
            // ombres de contact (lancer de rayons dans l'image de profondeur)
            contact_shadows_enabled: true,
            // ombres douces : le soleil n'est pas un point (pénombre au pied des palmiers)
            soft_shadow_size: if std::env::var("OCEAN_PCSS").map(|v| v == "0").unwrap_or(false) {
                None
            } else {
                Some(2.0)
            },
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.1,
            first_cascade_far_bound: 18.0,
            maximum_distance: 450.0,
            overlap_proportion: 0.25,
        }
        .build(),
        Transform::from_translation(dir * 100.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Matériau du dôme céleste.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SkyMaterial {
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

impl OceanMaterial for SkyMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for SkyMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/sky.wesl".into()
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
        Ok(())
    }
}

#[derive(Component)]
struct SkyDome;

/// Rayon du dôme céleste (au-delà du maillage de l'océan).
const SKY_RADIUS: f32 = 45_000.0;

fn spawn_sky_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
    textures: Res<OceanTextures>,
    ocean: Res<Ocean>,
) {
    commands.spawn((
        Name::new("Dôme céleste"),
        SkyDome,
        Mesh3d(meshes.add(Sphere::new(SKY_RADIUS).mesh().uv(96, 48))),
        MeshMaterial3d(materials.add(SkyMaterial {
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

fn follow_camera(
    camera: Query<&Transform, (With<Camera3d>, Without<SkyDome>)>,
    mut dome: Query<&mut Transform, With<SkyDome>>,
) {
    let Some(cam) = camera.iter().next() else {
        return;
    };
    for mut t in &mut dome {
        t.translation = cam.translation;
    }
}
