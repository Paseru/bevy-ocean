//! Fond marin : sable blanc à -15 m, petite île avec plage et banc de sable,
//! patates de corail et rochers.

use std::f32::consts::{PI, TAU};

use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use crate::noise::{Rng, fbm, perlin, ridged};
use crate::ocean::{FLOOR_DEPTH, Ocean, OceanMaterial, OceanSystems, OceanTextures, OceanUniforms, sync_ocean_uniforms};

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SeabedMaterial>::default())
            .init_resource::<Obstacles>()
            .add_systems(Startup, (spawn_terrain, spawn_reef).chain())
            .add_systems(
                PostUpdate,
                sync_ocean_uniforms::<SeabedMaterial>.in_set(OceanSystems::Sync),
            );
    }
}

// ---------------------------------------------------------------------------------------------
// Relief
// ---------------------------------------------------------------------------------------------

/// Centre de l'île (x, z).
pub const ISLAND_CENTER: Vec2 = Vec2::new(0.0, 0.0);
const ISLAND_RADII: Vec2 = Vec2::new(118.0, 72.0);
const ISLAND_ANGLE: f32 = 0.42;

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    b + (a - b) * h + k * h * (1.0 - h)
}

/// Distance signée approchée à une ellipse (positive à l'intérieur), en mètres.
fn ellipse_d(q: Vec2, radii: Vec2, angle: f32) -> f32 {
    let (s, c) = (-angle).sin_cos();
    let r = Vec2::new(c * q.x - s * q.y, s * q.x + c * q.y);
    let e = (r / radii).length();
    (1.0 - e) * 0.5 * (radii.x + radii.y)
}

/// Distance signée à une "capsule" dont le rayon varie de r0 (en a) à r1 (en b).
fn capsule_d(p: Vec2, a: Vec2, b: Vec2, r0: f32, r1: f32) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    let r = r0 + (r1 - r0) * t;
    r - (p - (a + ab * t)).length()
}

/// Distance signée (approchée, en mètres) au rivage de l'île : positive sur la terre.
/// Corps principal + lobe au nord-est + flèche de sable sous le vent, côte déformée par du
/// bruit (anses, pointes).
pub fn island_distance(p: Vec2) -> f32 {
    let warp = Vec2::new(fbm(p / 150.0, 3, 41), fbm(p / 150.0 + Vec2::splat(7.3), 3, 42)) * 16.0
        + Vec2::new(fbm(p / 42.0, 2, 43), fbm(p / 42.0 + Vec2::splat(3.1), 2, 44)) * 4.5;
    let q = p + warp - ISLAND_CENTER;
    let body = ellipse_d(q, ISLAND_RADII, ISLAND_ANGLE);
    let lobe = ellipse_d(q - Vec2::new(38.0, -52.0), Vec2::new(52.0, 30.0), 0.95);
    let spit = capsule_d(q, Vec2::new(-92.0, -38.0), Vec2::new(-178.0, -30.0), 17.0, 3.0);
    let d = smooth_max(smooth_max(body, lobe, 14.0), spit, 10.0);
    d - 1.8 * fbm(p / 13.0, 2, 12)
}

/// Géométrie du platier : (largeur du platier, position de la crête récifale, profondeur de
/// la crête, part de "passe" (crête interrompue)).
fn reef_geometry(p: Vec2) -> (f32, f32, f32, f32) {
    let angle = (p - ISLAND_CENTER).to_angle();
    let tail = (-((angle - PI).rem_euclid(TAU) - PI).abs() / 0.9).exp(); // pic vers -X
    let shelf_width = 135.0 + 70.0 * fbm(p / 150.0, 3, 31) + 120.0 * tail;
    let crest_pos = shelf_width - 18.0;
    let dir = Vec2::new(angle.cos(), angle.sin());
    // passes : la crête est interrompue par endroits (bruit continu autour de l'île)
    let pass = smoothstep(0.42, 0.62, fbm(dir * 2.3 + Vec2::splat(11.0), 2, 34));
    let crest_top = -0.55 - 0.45 * fbm(p / 55.0, 3, 33).abs();
    (shelf_width, crest_pos, crest_top, pass)
}

/// Hauteur du fond (m) au point (x, z). Le niveau moyen de la mer est 0.
pub fn height_at(x: f32, z: f32) -> f32 {
    let p = Vec2::new(x, z);
    let d = island_distance(p);

    // --- fond du large : sable à -15 m avec de grandes dunes douces et des rides de sable
    let dunes = 0.55 * fbm(p / 110.0, 4, 3);
    let sand_wave_dir = Vec2::new(0.8, 0.6);
    let warp = 2.5 * fbm(p / 45.0, 2, 4);
    let sand_waves = 0.10 * (p.dot(sand_wave_dir) / 2.6 + warp).sin()
        * (0.5 + 0.5 * perlin(p / 60.0, 5));
    let floor = -FLOOR_DEPTH + dunes + sand_waves;

    if d > 0.0 {
        // --- terre ferme : haut de plage (berme), puis l'intérieur boisé un peu plus haut
        let berm = 1.35 * (1.0 - (-d / 8.0).exp());
        let interior = smoothstep(14.0, 55.0, d) * (1.4 + 1.0 * fbm(p / 35.0, 3, 21));
        let bumps = smoothstep(6.0, 20.0, d) * 0.3 * fbm(p / 9.0, 2, 22);
        return berm + interior + bumps;
    }

    // --- sous l'eau : face de plage, lagon, crête récifale, tombant, fond du large
    let od = -d; // distance au rivage, vers le large
    let (shelf_width, crest_pos, crest_top, pass) = reef_geometry(p);
    let beach_face = -od * 0.085;
    let u = od / shelf_width;
    let lagoon = -1.2 - 1.6 * smoothstep(0.05, 0.4, u) - 0.5 * fbm(p / 40.0, 3, 32).abs();
    let flat = smooth_max(beach_face, lagoon, 0.8);
    // crête récifale (là où la houle déferle)
    let bump = (-((od - crest_pos) / 13.0).powi(2)).exp() * (1.0 - pass);
    let reef = flat.max(flat + (crest_top - flat) * bump);
    // tombant vers le fond du large
    let drop = smoothstep(crest_pos + 3.0, crest_pos + 70.0, od);
    let h = reef + (floor - reef) * drop;
    h.max(floor)
}

/// Masques de matériau du relief (couleurs des sommets) :
/// r : récif (corail, débris), g : sol sous la végétation, b : variation du sable.
pub fn terrain_masks(x: f32, z: f32, h: f32) -> [f32; 4] {
    let p = Vec2::new(x, z);
    let d = island_distance(p);
    if d > 0.0 {
        let veg = smoothstep(15.0, 30.0, d + 8.0 * fbm(p / 30.0, 2, 51));
        return [0.0, veg, 0.5 + 0.5 * fbm(p / 20.0, 2, 52), 1.0];
    }
    let od = -d;
    let (_, crest_pos, _, pass) = reef_geometry(p);
    let crest = (-((od - crest_pos) / 14.0).powi(2)).exp() * (1.0 - 0.7 * pass);
    // pâtés de corail épars sur le platier et sur le tombant
    let patches = smoothstep(0.35, 0.65, fbm(p / 18.0, 3, 53)) * smoothstep(-1.3, -2.0, h) * 0.8;
    let fore = smoothstep(crest_pos, crest_pos + 25.0, od) * (1.0 - smoothstep(crest_pos + 40.0, crest_pos + 80.0, od));
    let reef = (crest + patches + 0.6 * fore * smoothstep(0.2, 0.6, fbm(p / 12.0, 2, 54))).clamp(0.0, 1.0);
    [reef, 0.0, 0.5 + 0.5 * fbm(p / 20.0, 2, 52), 1.0]
}

/// Coordonnées de grille : pas fin au centre, de plus en plus grand vers l'extérieur.
fn grid_coords(fine_half: f32, fine_step: f32, outer: f32, growth: f32) -> Vec<f32> {
    let mut pos = Vec::new();
    let n = (fine_half / fine_step).round() as i32;
    for i in -n..=n {
        pos.push(i as f32 * fine_step);
    }
    let mut step = fine_step;
    let mut x = n as f32 * fine_step;
    let mut outer_pos = Vec::new();
    while x < outer {
        step *= growth;
        x += step;
        outer_pos.push(x.min(outer));
    }
    let mut all: Vec<f32> = outer_pos.iter().rev().map(|v| -v).collect();
    all.extend(pos);
    all.extend(outer_pos);
    all
}

fn build_terrain_mesh() -> Mesh {
    // pas fin (1,2 m) jusqu'à 600 m : l'île, le lagon et la crête récifale
    let xs = grid_coords(600.0, 1.2, 3000.0, 1.10);
    let n = xs.len();
    // hauteurs (calcul parallèle)
    let mut heights = vec![0.0f32; n * n];
    let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, chunk) in heights.chunks_mut(rows_per * n).enumerate() {
            let xs = &xs;
            s.spawn(move || {
                for (k, h) in chunk.iter_mut().enumerate() {
                    let idx = ci * rows_per * n + k;
                    let (iz, ix) = (idx / n, idx % n);
                    *h = height_at(xs[ix], xs[iz]);
                }
            });
        }
    });
    let mut positions = Vec::with_capacity(n * n);
    let mut normals = Vec::with_capacity(n * n);
    let mut colors = vec![[0.0f32; 4]; n * n];
    std::thread::scope(|s| {
        for (ci, chunk) in colors.chunks_mut(rows_per * n).enumerate() {
            let xs = &xs;
            let heights = &heights;
            s.spawn(move || {
                for (k, c) in chunk.iter_mut().enumerate() {
                    let idx = ci * rows_per * n + k;
                    let (iz, ix) = (idx / n, idx % n);
                    *c = terrain_masks(xs[ix], xs[iz], heights[idx]);
                }
            });
        }
    });
    for iz in 0..n {
        for ix in 0..n {
            let h = heights[iz * n + ix];
            positions.push([xs[ix], h, xs[iz]]);
            let (x0, x1) = (ix.saturating_sub(1), (ix + 1).min(n - 1));
            let (z0, z1) = (iz.saturating_sub(1), (iz + 1).min(n - 1));
            let dhdx = (heights[iz * n + x1] - heights[iz * n + x0]) / (xs[x1] - xs[x0]);
            let dhdz = (heights[z1 * n + ix] - heights[z0 * n + ix]) / (xs[z1] - xs[z0]);
            normals.push(Vec3::new(-dhdx, 1.0, -dhdz).normalize().to_array());
        }
    }
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for iz in 0..n - 1 {
        for ix in 0..n - 1 {
            let a = (iz * n + ix) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            // triangles orientés vers le haut (+Y)
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

// ---------------------------------------------------------------------------------------------
// Matériau
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct SeabedParams {
    /// Couleur de base (linéaire)
    pub base_color: Vec4,
    /// x : type (0 sable, 1 roche, 2 corail), y : rugosité, z : échelle du motif, w : libre
    pub kind: Vec4,
}

/// Matériau du fond (sable, roches, coraux) : éclairage sous-marin, caustiques, brume.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct SeabedMaterial {
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
    #[uniform(10)]
    pub params: SeabedParams,
    #[texture(11)]
    #[sampler(12)]
    pub sand_albedo: Handle<Image>,
    #[texture(13)]
    pub sand_nrh: Handle<Image>,
    #[texture(14)]
    pub litter_albedo: Handle<Image>,
    #[texture(15)]
    pub litter_nrh: Handle<Image>,
}

impl OceanMaterial for SeabedMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for SeabedMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/seabed.wesl".into()
    }
}

#[derive(Component)]
pub struct Terrain;

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SeabedMaterial>>,
    textures: Res<OceanTextures>,
    ocean: Res<Ocean>,
) {
    let t0 = std::time::Instant::now();
    let mesh = build_terrain_mesh();
    info!("Relief généré en {:.2} s", t0.elapsed().as_secs_f32());
    commands.spawn((
        Name::new("Fond marin et île"),
        Terrain,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(SeabedMaterial {
            ocean: ocean.uniforms,
            sky: textures.sky.clone(),
            fft_slopes: textures.fft_slopes.clone(),
            fft_extra: textures.fft_extra.clone(),
            shore0: textures.shore0.clone(),
            shore1: textures.shore1.clone(),
            foam: textures.foam.clone(),
            sand_albedo: textures.sand_albedo.clone(),
            sand_nrh: textures.sand_nrh.clone(),
            litter_albedo: textures.litter_albedo.clone(),
            litter_nrh: textures.litter_nrh.clone(),
            params: SeabedParams {
                // sable corallien : très clair, mais un albédo réaliste (~0,6) pour ne pas être
                // brûlé en plein soleil
                base_color: Vec4::new(0.74, 0.66, 0.52, 1.0),
                kind: Vec4::new(0.0, 0.85, 1.0, 0.0),
            },
        })),
        Transform::default(),
        NotShadowCaster,
    ));
}

// ---------------------------------------------------------------------------------------------
// Coraux et rochers
// ---------------------------------------------------------------------------------------------

/// Obstacles sphériques (coraux, rochers) pour les collisions de la caméra.
#[derive(Resource, Default)]
pub struct Obstacles {
    pub spheres: Vec<(Vec3, f32)>,
}

/// Déforme une icosphère pour en faire une patate de corail ou un rocher.
fn build_blob(seed: u32, kind: u32) -> Mesh {
    let mut mesh = Sphere::new(1.0).mesh().ico(4).unwrap();
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION).cloned()
    else {
        unreachable!()
    };
    let mut new_pos = Vec::with_capacity(positions.len());
    let mut colors = Vec::with_capacity(positions.len());
    for p in positions {
        let v = Vec3::from_array(p);
        let q = Vec2::new(v.x * 1.7 + v.y * 0.6, v.z * 1.7 - v.y * 0.5);
        let q2 = Vec2::new(v.x * 4.3 - v.z * 1.1, v.y * 4.3 + v.z * 2.0);
        let (r, ao) = if kind == 2 {
            // patate de corail : massive, arrondie, à gros lobes
            let lobes = 0.16 * fbm(q + Vec2::splat(seed as f32 * 3.1), 3, seed);
            let bumps = 0.05 * fbm(q2 * 1.7, 2, seed + 1);
            (1.0 + lobes + bumps, (0.5 + 2.0 * (lobes + bumps)).clamp(0.0, 1.0))
        } else {
            // rocher : formes plus anguleuses
            let big = 0.22 * ridged(q * 0.8 + Vec2::splat(seed as f32), 3, seed);
            let small = 0.06 * fbm(q2 * 2.2, 3, seed + 7);
            (0.85 + big + small, (0.4 + 2.5 * (big - 0.1 + small)).clamp(0.0, 1.0))
        };
        let mut w = v * r;
        // écrasé verticalement, base aplatie (enfouie dans le sable)
        let squash = if kind == 2 { 0.62 } else { 0.55 };
        w.y *= squash;
        if w.y < -0.25 {
            w.y = -0.25 + (w.y + 0.25) * 0.15;
        }
        new_pos.push(w.to_array());
        colors.push([ao, ao, ao, 1.0]);
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    mesh.compute_smooth_normals();
    mesh
}

pub fn spawn_reef(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SeabedMaterial>>,
    mut obstacles: ResMut<Obstacles>,
    textures: Res<OceanTextures>,
    ocean: Res<Ocean>,
) {
    let coral_meshes: Vec<Handle<Mesh>> = (0..6).map(|i| meshes.add(build_blob(100 + i, 2))).collect();
    let rock_meshes: Vec<Handle<Mesh>> = (0..4).map(|i| meshes.add(build_blob(200 + i, 1))).collect();
    let coral_mat = materials.add(SeabedMaterial {
        ocean: ocean.uniforms,
        sky: textures.sky.clone(),
        fft_slopes: textures.fft_slopes.clone(),
        fft_extra: textures.fft_extra.clone(),
        shore0: textures.shore0.clone(),
        shore1: textures.shore1.clone(),
        foam: textures.foam.clone(),
        sand_albedo: textures.sand_albedo.clone(),
        sand_nrh: textures.sand_nrh.clone(),
        litter_albedo: textures.litter_albedo.clone(),
        litter_nrh: textures.litter_nrh.clone(),
        params: SeabedParams {
            base_color: Vec4::new(0.55, 0.43, 0.30, 1.0),
            kind: Vec4::new(2.0, 0.8, 1.0, 0.0),
        },
    });
    let rock_mat = materials.add(SeabedMaterial {
        ocean: ocean.uniforms,
        sky: textures.sky.clone(),
        fft_slopes: textures.fft_slopes.clone(),
        fft_extra: textures.fft_extra.clone(),
        shore0: textures.shore0.clone(),
        shore1: textures.shore1.clone(),
        foam: textures.foam.clone(),
        sand_albedo: textures.sand_albedo.clone(),
        sand_nrh: textures.sand_nrh.clone(),
        litter_albedo: textures.litter_albedo.clone(),
        litter_nrh: textures.litter_nrh.clone(),
        params: SeabedParams {
            base_color: Vec4::new(0.42, 0.40, 0.36, 1.0),
            kind: Vec4::new(1.0, 0.75, 1.0, 0.0),
        },
    });

    let mut rng = Rng::new(42);
    let mut placed: Vec<(Vec2, f32)> = Vec::new();
    let try_place = |rng: &mut Rng, placed: &mut Vec<(Vec2, f32)>, p: Vec2, r: f32| -> bool {
        if placed.iter().any(|(q, rq)| q.distance(p) < (r + rq) * 1.1) {
            return false;
        }
        placed.push((p, r));
        let _ = rng;
        true
    };

    // patates de corail : en bouquets sur le banc de sable et le tombant, quelques-unes au large
    let mut clusters: Vec<Vec2> = Vec::new();
    while clusters.len() < 26 {
        let a = rng.range(0.0, TAU);
        let dist = rng.range(160.0, 460.0);
        let p = ISLAND_CENTER + Vec2::new(a.cos(), a.sin()) * dist;
        let h = height_at(p.x, p.y);
        if (-13.5..=-2.0).contains(&h) {
            clusters.push(p);
        }
    }
    // quelques patates isolées près du point de vue de départ
    clusters.push(Vec2::new(18.0, 330.0));
    clusters.push(Vec2::new(-30.0, 300.0));
    let mut count = 0;
    for c in &clusters {
        let n = 3 + (rng.f32() * 6.0) as usize;
        for _ in 0..n {
            let p = *c + Vec2::new(rng.normal(), rng.normal()) * 7.0;
            let h = height_at(p.x, p.y);
            if h > -1.6 {
                continue;
            }
            let scale = rng.range(0.7, 2.6) * if rng.f32() < 0.15 { 1.6 } else { 1.0 };
            if !try_place(&mut rng, &mut placed, p, scale) {
                continue;
            }
            let rot = Quat::from_rotation_y(rng.range(0.0, TAU))
                * Quat::from_rotation_x(rng.range(-0.08, 0.08));
            let s = Vec3::new(scale * rng.range(0.85, 1.2), scale * rng.range(0.8, 1.25), scale * rng.range(0.85, 1.2));
            let pos = Vec3::new(p.x, h - 0.12 * scale, p.y);
            commands.spawn((
                Name::new("Patate de corail"),
                Mesh3d(coral_meshes[(rng.f32() * coral_meshes.len() as f32) as usize % coral_meshes.len()].clone()),
                MeshMaterial3d(coral_mat.clone()),
                Transform::from_translation(pos).with_rotation(rot).with_scale(s),
            ));
            obstacles.spheres.push((pos + Vec3::Y * 0.2 * scale, scale * 0.95));
            count += 1;
        }
    }
    // rochers de plage, au ras de l'eau
    let mut rocks = 0;
    let mut tries = 0;
    while rocks < 22 && tries < 5000 {
        tries += 1;
        let a = rng.range(0.0, TAU);
        let p = ISLAND_CENTER + Vec2::new(a.cos() * ISLAND_RADII.x * 1.4, a.sin() * ISLAND_RADII.y * 1.4)
            .rotate(Vec2::from_angle(ISLAND_ANGLE)) * rng.range(0.75, 1.15);
        let d = island_distance(p);
        if !(-14.0..=2.0).contains(&d) {
            continue;
        }
        let scale = rng.range(0.5, 1.8);
        if !try_place(&mut rng, &mut placed, p, scale) {
            continue;
        }
        let h = height_at(p.x, p.y);
        let pos = Vec3::new(p.x, h - 0.1 * scale, p.y);
        commands.spawn((
            Name::new("Rocher"),
            Mesh3d(rock_meshes[rocks % rock_meshes.len()].clone()),
            MeshMaterial3d(rock_mat.clone()),
            Transform::from_translation(pos)
                .with_rotation(Quat::from_rotation_y(rng.range(0.0, TAU)))
                .with_scale(Vec3::new(scale * 1.3, scale, scale)),
        ));
        obstacles.spheres.push((pos, scale));
        rocks += 1;
    }
    info!("Récif : {count} patates de corail, {rocks} rochers");
}
