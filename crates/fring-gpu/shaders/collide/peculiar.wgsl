// Peculiar velocity from the double-single state (SPEC 5.2, 12):
// u = v - v_circ(rho) phi_hat, rho = sqrt(x^2 + y^2), phi_hat = (-y, x, 0) / rho.
// v_circ is the circular speed in the equatorial plane with J2, J4, J6:
// v^2 = (GM / rho) (1 + 3/2 J2 q^2 - 15/8 J4 q^4 + 35/16 J6 q^6), q = R / rho
// (the coefficients are the Legendre values at the equator, as in
// `physics::circular::frequencies`). GM, R, J2, J4, J6 come from the uniform of
// the caller (from `params/` through `SaturnField`). Rust twin:
// `fring_model::physics::circular_speed` and
// `fring_collide::frame::peculiar_velocity`, run with the DS type of ds.rs.

struct U3 {
    x: vec2<f32>,
    y: vec2<f32>,
    z: vec2<f32>,
}

fn vcirc_ds(rho: vec2<f32>, gm: vec2<f32>, rr: vec2<f32>, j2: vec2<f32>, j4: vec2<f32>, j6: vec2<f32>) -> vec2<f32> {
    let q = ds_div(rr, rho);
    let q2 = ds_mul(q, q);
    let q4 = ds_mul(q2, q2);
    let q6 = ds_mul(q4, q2);
    let t2 = ds_mul_f(ds_mul(j2, q2), 1.5);
    let t4 = ds_mul_f(ds_mul(j4, q4), -1.875);
    let t6 = ds_mul_f(ds_mul(j6, q6), 2.1875);
    let fac = ds_add(ds_add(ds_add(ds(1.0), t2), t4), t6);
    return ds_sqrt(ds_mul(ds_div(gm, rho), fac));
}

fn peculiar_ds(x: vec2<f32>, y: vec2<f32>, vx: vec2<f32>, vy: vec2<f32>, vz: vec2<f32>,
               gm: vec2<f32>, rr: vec2<f32>, j2: vec2<f32>, j4: vec2<f32>, j6: vec2<f32>) -> U3 {
    let rho = ds_sqrt(ds_add(ds_mul(x, x), ds_mul(y, y)));
    let f = ds_div(vcirc_ds(rho, gm, rr, j2, j4, j6), rho);
    return U3(ds_add(vx, ds_mul(f, y)), ds_sub(vy, ds_mul(f, x)), vz);
}
