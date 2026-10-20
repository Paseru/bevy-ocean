//! Données du ciel, générées à partir de l'HDRI "Kloofendal 48d Partly Cloudy (Pure Sky)"
//! (Poly Haven, CC0) par un script de pré-traitement : direction et intensité du soleil,
//! harmoniques sphériques (irradiance) du ciel sans le soleil.
//! Ne pas éditer à la main.

/// Direction (normalisée) vers le soleil, repère monde (Y en haut).
pub const SUN_DIRECTION: [f32; 3] = [0.549410, 0.741723, 0.384701];
/// Couleur linéaire du soleil (normalisée).
pub const SUN_COLOR: [f32; 3] = [0.9927, 1.0000, 0.9079];
/// Éclairement du soleil, perpendiculaire aux rayons (lux).
pub const SUN_ILLUMINANCE: f32 = 100000.0;
/// Les valeurs de la cubemap sont stockées en kcd/m² : facteur pour revenir en cd/m².
pub const SKY_CUBE_SCALE: f32 = 1000.0;
/// Irradiance du ciel (sans soleil) en harmoniques sphériques d'ordre 2 (lux, RGB).
/// Base : 1, y, z, x, xy, yz, 3z²-1, xz, x²-y² (constantes de normalisation incluses côté shader).
pub const SKY_SH_IRRADIANCE: [[f32; 3]; 9] = [
    [55687.051, 74631.078, 104742.240],
    [30797.599, 29442.472, 43059.095],
    [9456.920, 10934.518, 14375.253],
    [14043.381, 16711.565, 22903.820],
    [6249.655, 7199.741, 9723.827],
    [4312.087, 4777.441, 6132.311],
    [-1552.202, -1697.769, -2304.340],
    [3424.013, 3805.916, 4921.243],
    [-1844.217, -1567.200, -1447.127]
];
