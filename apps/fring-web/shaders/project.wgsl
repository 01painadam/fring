// W2: turns the double-single state of GpuStepper into render instances.
//
// State buffer (ICR-005 item 5): 6 planes (x, y, z, vx, vy, vz; inertial, km and
// km/s). Particle i of plane c is the pair at f32 words 2 (c cap + i) (high) and
// 2 (c cap + i) + 1 (low). Only x and y are read here.
//
// Output: one vec4 per particle at instance index `first + i`:
//   xy = position in the view plane, km, RELATIVE TO THE EYE,
//   z  = disc diameter in pixels, w = colour id (0 = particle).
//
// The host sends q = origin + R(-angle) eye as a double-single pair (f64 on the
// host), so that the view position relative to the eye is R (p - q). The
// difference p - q is the only large subtraction: hi - q_hi is exact when p is
// near q (Sterbenz), and the low parts are small. The error is then about
// 6e-8 of |p - q|, not of |p|: below 1e-3 px for anything in the window.
//
// GUARD (M16, as ds.wgsl): Metal fast math may re-associate sums. Each
// high difference goes through `opaque`, a select on a run-time condition (the
// uniform `one` is 1.0), so that (hi - q_hi) + (lo - q_lo) cannot be turned into
// (hi + lo) - (q_hi + q_lo), which would drop the low parts. (A first version
// that multiplied both differences by `one`, as ds.wgsl does for its sums, lost
// the low parts on the M4: the compiler pulled the common factor out and then
// re-associated.) A native test (tests/project_gpu.rs) checks the result on the
// GPU against the Rust twin `project_twin`.

struct Proj {
    q_hi: vec2<f32>,
    q_lo: vec2<f32>,
    rot: vec2<f32>,     // cos, sin of the view turn
    size_px: f32,
    one: f32,
    n: u32,
    cap: u32,
    first: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read_write> inst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: Proj;

// A value that the compiler cannot see through: `select` with a condition that
// is known only at run time (one < 0.5 is false). The compiler cannot move the
// terms of the difference across the select, so it cannot re-associate the sum
// (hi - q_hi) + (lo - q_lo) into (hi + lo) - (q_hi + q_lo).
fn opaque(x: f32) -> f32 {
    return select(x, 0.0, u.one < 0.5);
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) {
        return;
    }
    let ix = 2u * i;
    let iy = 2u * (u.cap + i);
    let dxh = opaque(state[ix] - u.q_hi.x);
    let dyh = opaque(state[iy] - u.q_hi.y);
    let dx = dxh + (state[ix + 1u] - u.q_lo.x);
    let dy = dyh + (state[iy + 1u] - u.q_lo.y);
    let vx = u.rot.x * dx - u.rot.y * dy;
    let vy = u.rot.y * dx + u.rot.x * dy;
    inst[u.first + i] = vec4<f32>(vx, vy, u.size_px, 0.0);
}
