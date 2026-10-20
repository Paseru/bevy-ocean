//! Dôme céleste : HDRI du ciel + disque solaire analytique.

import bevy_pbr::render::{forward_io::VertexOutput, mesh_view_bindings::view};
import super::ocean_common::{
    PI, camera_underwater, debug_mode, sun_dir, sun_illuminance, sun_angular_radius, sky_radiance,
    underwater_background, phase_hg, sun_halo_strength,
};

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_position.xyz - view.world_position);
    var color: vec3<f32>;
    if camera_underwater() {
        color = underwater_background(dir, in.position.xy);
    } else if dir.y >= 0.0 {
        color = sky_radiance(dir, 0.0);
        // halo autour du soleil (diffusion vers l'avant par les aérosols et l'humidité) :
        // un cœur éblouissant et une auréole large, que le bloom étale encore
        let cos_s = dot(dir, sun_dir());
        let halo = sun_halo_strength();
        if halo > 0.0 {
            color += sun_illuminance() * halo
                * (0.0012 * phase_hg(cos_s, 0.96) + 0.02 * phase_hg(cos_s, 0.6));
        }
        // disque solaire avec assombrissement centre-bord
        let r = sun_angular_radius();
        let ang = acos(clamp(dot(dir, sun_dir()), -1.0, 1.0));
        if ang < r * 1.5 {
            let x = clamp(ang / r, 0.0, 1.0);
            let mu = sqrt(max(1.0 - x * x, 0.0));
            let u = 0.6;
            let limb = (1.0 - u * (1.0 - mu)) / (1.0 - u / 3.0);
            let edge = 1.0 - smoothstep(0.97, 1.03, ang / r);
            color += sun_illuminance() / (PI * r * r) * limb * edge;
        }
    } else {
        // sous l'horizon (au-delà du maillage de l'océan) : reflet du ciel bas
        color = sky_radiance(normalize(vec3(dir.x, 0.03, dir.z)), 2.0) * 0.85;
    }
    if debug_mode() == 1u {
        return vec4(1.0, 0.0, 1.0, 1.0);
    }
    return vec4(color * view.exposure, 1.0);
}
