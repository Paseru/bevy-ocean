//! Végétation de l'île, générée procéduralement : cocotiers (tronc courbé, palmes avec leurs
//! folioles, noix de coco, palmes sèches), buissons de bord de plage (Scaevola) et touffes
//! d'herbe sur le haut de plage.
//!
//! Les palmes sont des entités séparées : le vent les fait osciller (transformations mises à
//! jour chaque image), ce qui garde les ombres et les vecteurs de mouvement corrects sans
//! shader de sommets particulier.

use std::f32::consts::{PI, TAU};

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

use crate::noise::{Rng, fbm};
use crate::ocean::{Ocean, OceanMaterial, OceanSystems, OceanTextures, OceanUniforms, sync_ocean_uniforms};
use crate::terrain::{self, Obstacles};

pub struct VegetationPlugin;

impl Plugin for VegetationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<FoliageMaterial>::default())
            .add_systems(Startup, spawn_vegetation.after(terrain::spawn_reef))
            .add_systems(Update, sway)
            .add_systems(
                PostUpdate,
                sync_ocean_uniforms::<FoliageMaterial>.in_set(OceanSystems::Sync),
            );
    }
}

// ---------------------------------------------------------------------------------------------
// Matériau
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct FoliageParams {
    /// Albédo de base (linéaire)
    pub color: Vec4,
    /// x : type (0 tronc, 1 feuille, 2 noix, 3 herbe), y : translucidité, z : rugosité,
    /// w : amplitude des variations de teinte
    pub kind: Vec4,
}

/// Matériau des plantes : éclairage de la scène (soleil, ciel, rebond sur le sable, ombres
/// des nuages, occlusion), translucidité des feuilles à contre-jour, reflets cireux.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct FoliageMaterial {
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
    pub params: FoliageParams,
}

impl OceanMaterial for FoliageMaterial {
    fn ocean_uniforms_mut(&mut self) -> &mut OceanUniforms {
        &mut self.ocean
    }
}

impl Material for FoliageMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/foliage.wesl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // feuilles fines : visibles (et projetant une ombre) des deux côtés
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Construction des maillages
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct MeshBuilder {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    uv: Vec<[f32; 2]>,
    col: Vec<[f32; 4]>,
    idx: Vec<u32>,
}

impl MeshBuilder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2, ao: f32, tint: f32) -> u32 {
        self.pos.push(p.to_array());
        self.nrm.push(n.normalize_or_zero().to_array());
        self.uv.push(uv.to_array());
        self.col.push([ao, tint, 0.0, 1.0]);
        (self.pos.len() - 1) as u32
    }

    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.idx.extend_from_slice(&[a, b, c]);
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.idx.extend_from_slice(&[a, b, c, b, d, c]);
    }

    fn build(self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.pos)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.nrm)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uv)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.col)
            .with_inserted_indices(Indices::U32(self.idx))
    }
}

/// Courbe du tronc : part en biais puis se redresse (cocotier penché vers la mer).
fn trunk_point(t: f32, height: f32, lean: f32, bend: f32) -> Vec3 {
    // x : écart horizontal, y : hauteur. Le tronc se redresse vers le haut (bend > 0).
    let y = t * height;
    let x = lean * height * (t - bend * t * t * 0.5);
    Vec3::new(x, y, 0.0)
}

/// Tronc de cocotier : tube courbe, renflé à la base. uv.y = distance le long du tronc (m).
fn build_trunk(height: f32, lean: f32, bend: f32) -> (Mesh, Vec3) {
    let mut b = MeshBuilder::default();
    let rings = 28;
    let sides = 10;
    let mut prev: Option<Vec<u32>> = None;
    let mut dist = 0.0;
    let mut last = trunk_point(0.0, height, lean, bend);
    for i in 0..=rings {
        let t = i as f32 / rings as f32;
        let c = trunk_point(t, height, lean, bend);
        dist += c.distance(last);
        last = c;
        let tangent = (trunk_point((t + 0.01).min(1.0), height, lean, bend)
            - trunk_point((t - 0.01).max(0.0), height, lean, bend))
        .normalize();
        let side = Vec3::Z;
        let up = side.cross(tangent).normalize();
        // rayon : renflement de la base (bulbe racinaire), léger fuseau
        let r = 0.15 + 0.06 * (1.0 - t) + 0.13 * (-(t * height) / 0.35).exp();
        let mut ring = Vec::with_capacity(sides + 1);
        for j in 0..=sides {
            let a = j as f32 / sides as f32 * TAU;
            let n = up * a.cos() + side * a.sin();
            let ao = 0.55 + 0.45 * (t * 4.0).min(1.0);
            ring.push(b.vertex(c + n * r, n, Vec2::new(j as f32 / sides as f32, dist), ao, 0.0));
        }
        if let Some(p) = &prev {
            for j in 0..sides {
                b.quad(p[j], p[j + 1], ring[j], ring[j + 1]);
            }
        }
        prev = Some(ring);
    }
    // base enfoncée dans le sable
    (b.build(), trunk_point(1.0, height, lean, bend))
}

/// Une palme : rachis arqué le long de +X et deux rangées de folioles qui pendent en V.
/// `elev` : angle de départ (rad), `droop` : retombée du bout (m), `dead` : palme sèche.
fn build_frond(rng: &mut Rng, length: f32, elev: f32, droop: f32, dead: bool) -> Mesh {
    let mut b = MeshBuilder::default();
    let k = droop / (length * length);
    let pos = |s: f32| Vec3::new(elev.cos() * s, elev.sin() * s - k * s * s, 0.0);
    let tan = |s: f32| (pos(s + 0.01) - pos((s - 0.01).max(0.0))).normalize();
    // rachis : ruban étroit
    let n_seg = 14;
    let mut prev: Option<(u32, u32)> = None;
    for i in 0..=n_seg {
        let s = i as f32 / n_seg as f32 * length;
        let t = tan(s);
        let up = Vec3::Z.cross(t).normalize();
        let w = 0.035 * (1.0 - 0.7 * s / length);
        let a = b.vertex(pos(s) + Vec3::Z * w, up, Vec2::new(0.0, s), 0.8, 0.0);
        let c = b.vertex(pos(s) - Vec3::Z * w, up, Vec2::new(1.0, s), 0.8, 0.0);
        if let Some((pa, pc)) = prev {
            b.quad(pa, pc, a, c);
        }
        prev = Some((a, c));
    }
    // folioles
    let step = 0.075;
    let start = 0.2 * length;
    let mut s = start;
    let max_len = 0.85 + 0.15 * length / 4.5;
    while s < length * 0.985 {
        let u = (s - start) / (length - start);
        let len = max_len * (PI * (0.08 + 0.92 * u)).sin().powf(0.6) * rng.range(0.85, 1.05);
        let t = tan(s);
        let up = Vec3::Z.cross(t).normalize();
        for side in [1.0f32, -1.0] {
            if dead && rng.f32() < 0.25 {
                continue; // folioles arrachées
            }
            // vers l'avant et vers le bas (en V), un peu de désordre
            let keel = if dead { 1.2 } else { 0.55 + 0.25 * u };
            let d = (Vec3::Z * side * (1.0 - keel * 0.5) + t * 0.4 - up * keel + Vec3::new(0.0, rng.range(-0.1, 0.1), 0.0)).normalize();
            let wdir = d.cross(up).normalize() * side;
            let base = pos(s) + Vec3::Z * side * 0.02;
            let width = 0.042 * rng.range(0.8, 1.15);
            let tint = rng.range(0.0, 1.0);
            let pts = [0.0f32, 0.3, 0.65, 1.0];
            let wid = [0.35f32, 1.0, 0.75, 0.05];
            let mut prev2: Option<(u32, u32)> = None;
            for (pi, &v) in pts.iter().enumerate() {
                // la foliole retombe sous son propre poids
                let sag = Vec3::new(0.0, -0.35 * len * v * v, 0.0);
                let c = base + d * (v * len) + sag;
                let tangent = (d * len + Vec3::new(0.0, -0.7 * len * v, 0.0)).normalize();
                let n = wdir.cross(tangent).normalize();
                let n = if n.y < 0.0 { -n } else { n };
                let ao = 0.55 + 0.45 * (0.3 + 0.7 * u) * (0.6 + 0.4 * v);
                let a = b.vertex(c + wdir * width * wid[pi], n, Vec2::new(0.0, v), ao, tint);
                let e = b.vertex(c - wdir * width * wid[pi], n, Vec2::new(1.0, v), ao, tint);
                if let Some((pa, pe)) = prev2 {
                    b.quad(pa, pe, a, e);
                }
                prev2 = Some((a, e));
            }
        }
        s += step * rng.range(0.85, 1.15);
    }
    b.build()
}

/// Grappe de noix de coco (sphères aplaties) sous la couronne.
fn build_coconuts(rng: &mut Rng) -> Mesh {
    let mut b = MeshBuilder::default();
    let count = 4 + (rng.f32() * 5.0) as usize;
    for i in 0..count {
        let a = i as f32 / count as f32 * TAU + rng.range(-0.3, 0.3);
        let c = Vec3::new(a.cos() * 0.2, rng.range(-0.35, -0.12), a.sin() * 0.2);
        let r = rng.range(0.1, 0.13);
        let tint = rng.f32();
        let (nu, nv) = (8, 6);
        let base = b.pos.len() as u32;
        for v in 0..=nv {
            let th = v as f32 / nv as f32 * PI;
            for u in 0..=nu {
                let ph = u as f32 / nu as f32 * TAU;
                let n = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
                let p = c + Vec3::new(n.x, n.y * 1.15, n.z) * r;
                b.vertex(p, n, Vec2::new(u as f32 / nu as f32, v as f32 / nv as f32), 0.5 + 0.5 * (n.y * 0.5 + 0.5), tint);
            }
        }
        for v in 0..nv {
            for u in 0..nu {
                let i0 = base + v * (nu + 1) + u;
                let i1 = i0 + nu + 1;
                b.quad(i0, i0 + 1, i1, i1 + 1);
            }
        }
    }
    b.build()
}

/// Buisson de bord de plage (Scaevola) : dôme dense de feuilles larges, surtout en surface ;
/// normales "arrondies" (la lumière enveloppe le buisson comme un volume).
fn build_shrub(rng: &mut Rng, radius: f32, height: f32) -> Mesh {
    let mut b = MeshBuilder::default();
    let leaves = 1000;
    for _ in 0..leaves {
        // direction sur la demi-sphère, point près de la surface du dôme
        let dir = Vec3::new(rng.normal(), rng.normal().abs() + 0.15, rng.normal()).normalize();
        let depth = rng.f32().powf(0.22);
        // dôme un peu irrégulier
        let bump = 1.0 + 0.18 * (dir.x * 5.0 + dir.z * 3.0).sin() * (dir.z * 4.0).cos();
        let c = Vec3::new(dir.x * radius, dir.y * height, dir.z * radius) * depth * bump;
        let radial = Vec3::new(c.x / radius, c.y / height + 0.25, c.z / radius).normalize_or_zero();
        // feuille orientée à peu près vers l'extérieur et le haut
        let up = (radial * 0.8 + Vec3::new(rng.range(-0.4, 0.4), 0.7, rng.range(-0.4, 0.4))).normalize();
        let along = up.cross(Vec3::new(rng.range(-1.0, 1.0), 0.0, rng.range(-1.0, 1.0)).normalize_or(Vec3::X)).normalize_or(Vec3::X);
        let side = up.cross(along).normalize();
        let l = rng.range(0.13, 0.21);
        let w = l * 0.42;
        let n = (up * 0.3 + radial * 0.7).normalize();
        let ao = (0.25 + 0.75 * depth.powf(3.0)).min(1.0) * (0.75 + 0.25 * dir.y);
        let tint = rng.f32();
        // feuille ovale, pliée le long de la nervure (les deux moitiés en V) et un peu arquée
        let fold = up * (w * 0.18);
        let arc = |v: f32| -up * (l * 0.12 * (v - 0.5) * (v - 0.5) * 4.0);
        let base = c - along * l * 0.5;
        let pt = |v: f32, s: f32| -> Vec3 {
            let half_w = w * 0.5 * (std::f32::consts::PI * v).sin().powf(0.8);
            base + along * (l * v) + side * (half_w * s) + fold * s.abs() + arc(v)
        };
        let i_base = b.vertex(pt(0.0, 0.0), n, Vec2::new(0.5, 0.0), ao, tint);
        let i_l1 = b.vertex(pt(0.3, -1.0), n, Vec2::new(0.0, 0.3), ao, tint);
        let i_r1 = b.vertex(pt(0.3, 1.0), n, Vec2::new(1.0, 0.3), ao, tint);
        let i_m1 = b.vertex(pt(0.3, 0.0), n, Vec2::new(0.5, 0.3), ao, tint);
        let i_l2 = b.vertex(pt(0.68, -1.0), n, Vec2::new(0.0, 0.68), ao, tint);
        let i_r2 = b.vertex(pt(0.68, 1.0), n, Vec2::new(1.0, 0.68), ao, tint);
        let i_m2 = b.vertex(pt(0.68, 0.0), n, Vec2::new(0.5, 0.68), ao, tint);
        let i_tip = b.vertex(pt(1.0, 0.0), n, Vec2::new(0.5, 1.0), ao, tint);
        b.tri(i_base, i_m1, i_l1);
        b.tri(i_base, i_r1, i_m1);
        b.quad(i_l1, i_m1, i_l2, i_m2);
        b.quad(i_m1, i_r1, i_m2, i_r2);
        b.tri(i_l2, i_m2, i_tip);
        b.tri(i_m2, i_r2, i_tip);
    }
    b.build()
}

/// Plaque d'herbe : touffes de brins fins et arqués sur ~3 × 3 m.
fn build_grass_patch(rng: &mut Rng) -> Mesh {
    let mut b = MeshBuilder::default();
    let clumps = 26;
    for _ in 0..clumps {
        let cx = rng.range(-1.6, 1.6);
        let cz = rng.range(-1.6, 1.6);
        let blades = 14 + (rng.f32() * 10.0) as usize;
        let h0 = rng.range(0.25, 0.55);
        for _ in 0..blades {
            let a = rng.range(0.0, TAU);
            let lean = rng.range(0.15, 0.7);
            let dir = Vec3::new(a.cos() * lean, 1.0, a.sin() * lean).normalize();
            let h = h0 * rng.range(0.6, 1.1);
            let base = Vec3::new(cx + rng.range(-0.08, 0.08), -0.02, cz + rng.range(-0.08, 0.08));
            let side = Vec3::new(-a.sin(), 0.0, a.cos());
            let w = 0.008;
            let tint = rng.f32();
            let p0 = base + side * w;
            let p1 = base - side * w;
            let mid = base + dir * h * 0.55 + Vec3::new(a.cos(), 0.0, a.sin()) * lean * h * 0.15;
            let tip = base + dir * h + Vec3::new(a.cos(), -0.4, a.sin()) * lean * h * 0.45;
            let n = side.cross(dir).normalize();
            let n = if n.y < 0.0 { -n } else { n };
            let i0 = b.vertex(p0, n, Vec2::new(0.0, 0.0), 0.45, tint);
            let i1 = b.vertex(p1, n, Vec2::new(1.0, 0.0), 0.45, tint);
            let i2 = b.vertex(mid + side * w * 0.7, n, Vec2::new(0.0, 0.55), 0.8, tint);
            let i3 = b.vertex(mid - side * w * 0.7, n, Vec2::new(1.0, 0.55), 0.8, tint);
            let i4 = b.vertex(tip, n, Vec2::new(0.5, 1.0), 1.0, tint);
            b.quad(i0, i1, i2, i3);
            b.tri(i2, i3, i4);
        }
    }
    b.build()
}

// ---------------------------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------------------------

/// Une palme qui oscille dans le vent autour de son point d'attache.
#[derive(Component)]
struct Frond {
    base: Quat,
    axis: Vec3,
    phase: f32,
    amp: f32,
}

/// Couronne d'un cocotier (oscille un peu avec le haut du tronc).
#[derive(Component)]
struct Crown {
    base: Quat,
    phase: f32,
}

fn spawn_vegetation(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<FoliageMaterial>>,
    mut obstacles: ResMut<Obstacles>,
    textures: Res<OceanTextures>,
    ocean: Res<Ocean>,
) {
    if std::env::var("OCEAN_NO_VEG").is_ok() {
        return;
    }
    let t0 = std::time::Instant::now();
    let mut mat = |color: Vec3, kind: f32, translucency: f32, roughness: f32, variation: f32| {
        materials.add(FoliageMaterial {
            ocean: ocean.uniforms,
            sky: textures.sky.clone(),
            fft_slopes: textures.fft_slopes.clone(),
            fft_extra: textures.fft_extra.clone(),
            shore0: textures.shore0.clone(),
            shore1: textures.shore1.clone(),
            foam: textures.foam.clone(),
            params: FoliageParams {
                color: color.extend(1.0),
                kind: Vec4::new(kind, translucency, roughness, variation),
            },
        })
    };
    let trunk_mat = mat(Vec3::new(0.22, 0.19, 0.155), 0.0, 0.0, 0.9, 0.25);
    let leaf_mat = mat(Vec3::new(0.10, 0.155, 0.035), 1.0, 0.5, 0.45, 0.6);
    let dead_mat = mat(Vec3::new(0.28, 0.20, 0.11), 1.0, 0.2, 0.85, 0.4);
    let nut_mat = mat(Vec3::new(0.20, 0.22, 0.05), 2.0, 0.0, 0.5, 0.8);
    let shrub_mat = mat(Vec3::new(0.07, 0.115, 0.026), 1.0, 0.4, 0.55, 0.45);
    let grass_mat = mat(Vec3::new(0.12, 0.14, 0.045), 3.0, 0.35, 0.95, 0.7);

    let mut rng = Rng::new(2024);
    // variantes de palmes : jeunes (dressées), adultes, vieilles (retombantes)
    let frond_meshes: Vec<(Handle<Mesh>, f32)> = (0..9)
        .map(|i| {
            let age = i % 3;
            let (len, elev, droop) = match age {
                0 => (rng.range(3.8, 4.6), 0.95, 0.7),
                1 => (rng.range(4.6, 5.5), 0.45, 1.6),
                _ => (rng.range(4.8, 5.8), 0.05, 2.3),
            };
            (meshes.add(build_frond(&mut rng, len, elev, droop, false)), age as f32)
        })
        .collect();
    let dead_meshes: Vec<Handle<Mesh>> =
        (0..3).map(|_| meshes.add(build_frond(&mut rng, 4.0, -1.05, 0.4, true))).collect();

    // --- cocotiers : sur le haut de plage (penchés vers la mer) et dans l'intérieur
    let mut placed: Vec<Vec2> = Vec::new();
    let mut palms = 0;
    let mut tries = 0;
    while palms < 78 && tries < 20000 {
        tries += 1;
        let p = Vec2::new(rng.range(-240.0, 180.0), rng.range(-170.0, 130.0));
        let d = terrain::island_distance(p);
        if d < 7.0 {
            continue;
        }
        // plus dense en bordure de la végétation, clairières au milieu
        let density = if d < 30.0 { 0.85 } else { 0.35 + 0.4 * (fbm(p / 40.0, 2, 77) * 0.5 + 0.5) };
        if rng.f32() > density {
            continue;
        }
        let min_gap = if d < 25.0 { 5.0 } else { 6.5 };
        if placed.iter().any(|q| q.distance(p) < min_gap) {
            continue;
        }
        placed.push(p);
        let ground = terrain::height_at(p.x, p.y);
        // direction vers la mer : gradient de la distance au rivage
        let e = 1.0;
        let grad = Vec2::new(
            terrain::island_distance(p + Vec2::X * e) - terrain::island_distance(p - Vec2::X * e),
            terrain::island_distance(p + Vec2::Y * e) - terrain::island_distance(p - Vec2::Y * e),
        );
        let seaward = (-grad).normalize_or(Vec2::X);
        let near_beach = (1.0 - (d - 7.0) / 25.0).clamp(0.0, 1.0);
        let lean = rng.range(0.05, 0.18) + 0.32 * near_beach;
        let height = rng.range(8.5, 15.0);
        let bend = rng.range(0.3, 0.9);
        let (trunk_mesh, top) = build_trunk(height, lean, bend);
        let trunk_mesh = meshes.add(trunk_mesh);
        // le tronc penche dans le plan (x, y) local : on l'oriente vers la mer (avec du jeu)
        let yaw = -seaward.to_angle() + rng.range(-0.5, 0.5);
        let rot = Quat::from_rotation_y(yaw);
        let root = commands
            .spawn((
                Name::new("Cocotier"),
                Transform::from_xyz(p.x, ground - 0.15, p.y).with_rotation(rot),
                Visibility::default(),
            ))
            .id();
        let trunk = commands
            .spawn((Mesh3d(trunk_mesh), MeshMaterial3d(trunk_mat.clone()), Transform::default()))
            .id();
        let crown_rot = Quat::from_rotation_y(rng.range(0.0, TAU));
        let crown = commands
            .spawn((
                Transform::from_translation(top).with_rotation(crown_rot),
                Visibility::default(),
                Crown {
                    base: crown_rot,
                    phase: rng.range(0.0, TAU),
                },
            ))
            .id();
        commands.entity(root).add_children(&[trunk, crown]);
        let fronds = 22 + (rng.f32() * 9.0) as usize;
        let mut children = Vec::new();
        for i in 0..fronds {
            let az = i as f32 / fronds as f32 * TAU * 2.618 + rng.range(-0.2, 0.2);
            let (mesh, age) = &frond_meshes[(rng.f32() * frond_meshes.len() as f32) as usize % frond_meshes.len()];
            let tilt = rng.range(-0.15, 0.15);
            let base = Quat::from_rotation_y(az) * Quat::from_rotation_z(tilt);
            let axis = Quat::from_rotation_y(az) * Vec3::Z;
            children.push(
                commands
                    .spawn((
                        Mesh3d(mesh.clone()),
                        MeshMaterial3d(leaf_mat.clone()),
                        Transform::from_translation(Quat::from_rotation_y(az) * Vec3::new(0.12, 0.05 * age, 0.0))
                            .with_rotation(base),
                        Frond {
                            base,
                            axis,
                            phase: rng.range(0.0, TAU),
                            amp: 0.025 + 0.02 * age,
                        },
                    ))
                    .id(),
            );
        }
        // palmes sèches pendantes
        for _ in 0..(1 + (rng.f32() * 3.0) as usize) {
            let az = rng.range(0.0, TAU);
            let base = Quat::from_rotation_y(az);
            children.push(
                commands
                    .spawn((
                        Mesh3d(dead_meshes[(rng.f32() * 3.0) as usize % 3].clone()),
                        MeshMaterial3d(dead_mat.clone()),
                        Transform::from_translation(Vec3::new(0.0, -0.25, 0.0)).with_rotation(base),
                        Frond {
                            base,
                            axis: base * Vec3::Z,
                            phase: rng.range(0.0, TAU),
                            amp: 0.015,
                        },
                    ))
                    .id(),
            );
        }
        // noix de coco
        if rng.f32() < 0.8 {
            children.push(
                commands
                    .spawn((
                        Mesh3d(meshes.add(build_coconuts(&mut rng))),
                        MeshMaterial3d(nut_mat.clone()),
                        Transform::default(),
                    ))
                    .id(),
            );
        }
        commands.entity(crown).add_children(&children);
        // collisions avec le tronc (quelques sphères le long de la courbe)
        for k in 0..4 {
            let tk = k as f32 / 3.0;
            let local = trunk_point(tk, height, lean, bend);
            let world = Vec3::new(p.x, ground, p.y) + rot * local;
            obstacles.spheres.push((world, 0.32));
        }
        palms += 1;
    }

    // --- buissons (Scaevola) : lisière de la végétation et sous-bois clairsemé
    let shrub_meshes: Vec<Handle<Mesh>> = (0..5)
        .map(|_| {
            let r = rng.range(1.0, 1.7);
            let h = rng.range(1.0, 1.7);
            meshes.add(build_shrub(&mut rng, r, h))
        })
        .collect();
    let mut shrubs = 0;
    let mut tries = 0;
    while shrubs < 330 && tries < 40000 {
        tries += 1;
        let p = Vec2::new(rng.range(-240.0, 180.0), rng.range(-170.0, 130.0));
        let d = terrain::island_distance(p);
        if d < 9.0 {
            continue;
        }
        let edge = (-((d - 15.0) / 6.0).powi(2)).exp();
        let inner = if d > 22.0 { 0.55 * (fbm(p / 25.0, 2, 78) * 0.5 + 0.5) } else { 0.0 };
        if rng.f32() > (edge * 0.9 + inner) {
            continue;
        }
        let ground = terrain::height_at(p.x, p.y);
        let s = rng.range(0.8, 1.3);
        commands.spawn((
            Name::new("Buisson"),
            Mesh3d(shrub_meshes[(rng.f32() * 5.0) as usize % 5].clone()),
            MeshMaterial3d(shrub_mat.clone()),
            Transform::from_xyz(p.x, ground - 0.1, p.y)
                .with_rotation(Quat::from_rotation_y(rng.range(0.0, TAU)))
                .with_scale(Vec3::new(s, s * rng.range(0.8, 1.1), s)),
        ));
        shrubs += 1;
    }

    // --- herbe du haut de plage
    let grass_meshes: Vec<Handle<Mesh>> = (0..4).map(|_| meshes.add(build_grass_patch(&mut rng))).collect();
    let mut grass = 0;
    let mut tries = 0;
    while grass < 260 && tries < 30000 {
        tries += 1;
        let p = Vec2::new(rng.range(-240.0, 180.0), rng.range(-170.0, 130.0));
        let d = terrain::island_distance(p);
        if !(8.5..45.0).contains(&d) {
            continue;
        }
        if fbm(p / 18.0, 2, 79) < -0.1 {
            continue;
        }
        let ground = terrain::height_at(p.x, p.y);
        commands.spawn((
            Name::new("Herbe"),
            Mesh3d(grass_meshes[(rng.f32() * 4.0) as usize % 4].clone()),
            MeshMaterial3d(grass_mat.clone()),
            Transform::from_xyz(p.x, ground, p.y).with_rotation(Quat::from_rotation_y(rng.range(0.0, TAU))),
        ));
        grass += 1;
    }
    info!(
        "Végétation : {palms} cocotiers, {shrubs} buissons, {grass} plaques d'herbe ({:.2} s)",
        t0.elapsed().as_secs_f32()
    );
}

/// Le vent (alizé) fait osciller les palmes et les couronnes, avec des rafales.
fn sway(time: Res<Time>, mut fronds: Query<(&Frond, &mut Transform), Without<Crown>>, mut crowns: Query<(&Crown, &mut Transform), Without<Frond>>) {
    let t = time.elapsed_secs_wrapped();
    // rafales : enveloppe lente commune
    let gust = 0.6 + 0.4 * (t * 0.31).sin() * (t * 0.17 + 1.0).sin();
    for (f, mut tr) in &mut fronds {
        let a = f.amp * gust * ((t * 1.7 + f.phase).sin() + 0.45 * (t * 3.9 + f.phase * 1.7).sin());
        tr.rotation = f.base * Quat::from_axis_angle(f.base.inverse() * f.axis, a);
    }
    for (c, mut tr) in &mut crowns {
        let a = 0.012 * gust * (t * 0.8 + c.phase).sin();
        tr.rotation = Quat::from_rotation_x(a) * Quat::from_rotation_z(a * 0.7) * c.base;
    }
}
