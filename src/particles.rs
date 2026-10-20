//! Particules en suspension dans l'eau ("neige marine") : de petits points qui flottent
//! autour de la caméra quand elle est sous l'eau, éclairés par la lumière qui descend de la
//! surface et atténués par l'eau. Ils donnent l'échelle et le volume de l'eau.

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
};

use crate::noise::Rng;
use crate::ocean::{Ocean, OceanMaterial, OceanSystems, OceanUniforms, sync_ocean_uniforms};

pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ParticleMaterial>::default())
            .add_systems(Startup, spawn_particles)
            .add_systems(
                PostUpdate,
                sync_ocean_uniforms::<ParticleMaterial>.in_set(OceanSystems::Sync),
            );
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ParticleMaterial {
    #[uniform(0)]
    pub ocean: OceanUniforms,
}

impl OceanMaterial for ParticleMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for ParticleMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/particles.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/particles.wesl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

/// Nombre de particules (dans un cube de 18 m autour de la caméra).
const COUNT: usize = 9000;

fn spawn_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ParticleMaterial>>,
    ocean: Res<Ocean>,
) {
    let mut rng = Rng::new(99);
    let mut pos = Vec::with_capacity(COUNT * 4);
    let mut uv = Vec::with_capacity(COUNT * 4);
    let mut col = Vec::with_capacity(COUNT * 4);
    let mut idx = Vec::with_capacity(COUNT * 6);
    for i in 0..COUNT {
        let seed = [rng.f32(), rng.f32(), rng.f32()];
        // taille (m), luminosité, phase de dérive
        let size = 0.0035 + 0.009 * rng.f32().powf(3.0);
        let extra = [size, rng.range(0.4, 1.0), rng.f32(), 1.0];
        for c in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]] {
            pos.push(seed);
            uv.push(c);
            col.push(extra);
        }
        let b = (i * 4) as u32;
        idx.extend_from_slice(&[b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, col)
        .with_inserted_indices(Indices::U32(idx));
    commands.spawn((
        Name::new("Particules en suspension"),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(ParticleMaterial { ocean: ocean.uniforms })),
        Transform::default(),
        NoFrustumCulling,
        NotShadowCaster,
    ));
}
