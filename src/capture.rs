//! Outil de capture d'écran automatique (utilisé pour régler le rendu).
//!
//! Variables d'environnement :
//! - `OCEAN_SHOTS` : liste de points de vue `x,y,z,yaw_deg,pitch_deg[,temps]` séparés par `;`
//!   (le temps, optionnel, fige la simulation pour ce cliché : utile pour une séquence)
//! - `OCEAN_SHOT_DIR` : dossier de sortie (défaut : `captures`)
//! - `OCEAN_TIME` : fige le temps de la simulation (secondes)
//! - `OCEAN_SHOT_DELAY` : secondes à attendre avant la première capture (défaut : 12)
//! - `OCEAN_SHOT_SIZE` : taille de l'image, par exemple `1920x1080` (rendu hors écran)
//! - `OCEAN_SHOT_WAIT` : images à attendre après chaque déplacement (défaut : 45), le temps
//!   que l'exposition automatique et la mise au point se stabilisent

use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::{
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, save_to_disk},
    },
};

use crate::camera::FlyCamera;
use crate::ocean::SimulationTime;

pub struct CapturePlugin;

impl Plugin for CapturePlugin {
    fn build(&self, app: &mut App) {
        // capture de la fenêtre elle-même (pour vérifier le rendu final, avec l'interface)
        if let Ok(path) = std::env::var("OCEAN_WINDOW_SHOT") {
            app.insert_resource(WindowShot { path, done: false })
                .add_systems(Update, window_shot);
        }
        let shots: Vec<[f32; 6]> = std::env::var("OCEAN_SHOTS")
            .ok()
            .map(|s| {
                s.split(';')
                    .filter_map(|p| {
                        let v: Vec<f32> = p.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                        match v.len() {
                            5 => Some([v[0], v[1], v[2], v[3], v[4], f32::NAN]),
                            6 => Some([v[0], v[1], v[2], v[3], v[4], v[5]]),
                            _ => None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        if shots.is_empty() {
            return;
        }
        let dir = std::env::var("OCEAN_SHOT_DIR").unwrap_or_else(|_| "captures".into());
        let delay = std::env::var("OCEAN_SHOT_DELAY").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0);
        let _ = std::fs::create_dir_all(&dir);
        let size = std::env::var("OCEAN_SHOT_SIZE")
            .ok()
            .and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some(UVec2::new(w.parse().ok()?, h.parse().ok()?))
            })
            .unwrap_or(UVec2::new(1920, 1080));
        app.insert_resource(CaptureQueue {
            shots,
            dir,
            delay,
            frame: 0,
            index: 0,
            wait_after_move: 0,
            size,
            target: None,
        })
        .add_systems(Startup, (freeze_time, offscreen_target))
        .add_systems(Update, run_captures);
    }
}

#[derive(Resource)]
struct CaptureQueue {
    shots: Vec<[f32; 6]>,
    dir: String,
    delay: f32,
    frame: u32,
    index: usize,
    wait_after_move: u32,
    size: UVec2,
    target: Option<Handle<Image>>,
}

/// Rendu hors écran : la capture ne dépend pas de la visibilité de la fenêtre.
fn offscreen_target(
    mut commands: Commands,
    mut queue: ResMut<CaptureQueue>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<Entity, With<FlyCamera>>,
) {
    let image = Image::new_target_texture(
        queue.size.x,
        queue.size.y,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    let handle = images.add(image);
    for e in &cameras {
        commands.entity(e).insert(RenderTarget::Image(handle.clone().into()));
    }
    queue.target = Some(handle);
}

fn freeze_time(mut sim: ResMut<SimulationTime>) {
    if let Some(t) = std::env::var("OCEAN_TIME").ok().and_then(|v| v.parse().ok()) {
        sim.frozen = Some(t);
    }
}

fn run_captures(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut queue: ResMut<CaptureQueue>,
    mut sim: ResMut<SimulationTime>,
    mut cam: Query<(&mut Transform, &mut FlyCamera)>,
    mut exit: MessageWriter<AppExit>,
) {
    queue.frame += 1;
    if time.elapsed_secs() < queue.delay {
        // place la caméra sur le premier point de vue dès le début
        if let Some(s) = queue.shots.first().copied() {
            place(&mut cam, s);
        }
        return;
    }
    if queue.index >= queue.shots.len() {
        // laisse le temps d'écrire la dernière image sur le disque
        if queue.wait_after_move > 150 {
            exit.write(AppExit::Success);
        }
        queue.wait_after_move += 1;
        return;
    }
    let s = queue.shots[queue.index];
    place(&mut cam, s);
    if s[5].is_finite() {
        sim.frozen = Some(s[5] as f64);
    }
    queue.wait_after_move += 1;
    // quelques images pour que l'exposition et les matériaux se stabilisent
    let wait: u32 = std::env::var("OCEAN_SHOT_WAIT").ok().and_then(|v| v.parse().ok()).unwrap_or(45);
    if queue.wait_after_move >= wait {
        let path = format!("{}/shot_{:02}.png", queue.dir, queue.index);
        let shot = match &queue.target {
            Some(h) => Screenshot::image(h.clone()),
            None => Screenshot::primary_window(),
        };
        commands.spawn(shot).observe(save_to_disk(path));
        queue.index += 1;
        queue.wait_after_move = 0;
    }
}

fn place(cam: &mut Query<(&mut Transform, &mut FlyCamera)>, s: [f32; 6]) {
    for (mut t, mut c) in cam.iter_mut() {
        c.yaw = s[3].to_radians();
        c.pitch = s[4].to_radians();
        c.velocity = Vec3::ZERO;
        t.translation = Vec3::new(s[0], s[1], s[2]);
        t.rotation = Quat::from_euler(EulerRot::YXZ, c.yaw, c.pitch, 0.0);
    }
}

#[derive(Resource)]
struct WindowShot {
    path: String,
    done: bool,
}

fn window_shot(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut shot: ResMut<WindowShot>,
    mut exit: MessageWriter<AppExit>,
) {
    let t = time.elapsed_secs();
    if !shot.done && t > 14.0 {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.path.clone()));
        shot.done = true;
    }
    if t > 17.0 {
        exit.write(AppExit::Success);
    }
}
