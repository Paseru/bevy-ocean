//! Passe principale du maillage "mouvement de l'eau" : rien à dessiner (triangles réduits à
//! un point). Ce maillage ne sert qu'à la passe préalable (vecteurs de mouvement).

@vertex
fn vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4(0.0, 0.0, 0.5, 1.0);
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4(0.0);
}
