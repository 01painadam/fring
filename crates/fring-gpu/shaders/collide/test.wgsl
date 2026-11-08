// Test kernels for the functions of rng.wgsl, pair.wgsl and peculiar.wgsl
// (task K5). Not used by a run. Buffers: 0 input, 1 output, 2 uniform.

@group(0) @binding(0) var<storage, read> t_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(2) var<uniform> t_u: array<vec4<u32>, 4>;

// Input per item: counter (4 words), key (2 words). Output: 4 words.
@compute @workgroup_size(64)
fn t_philox(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= t_u[0].x) { return; }
    let b = 6u * i;
    let w = philox4x32_10(vec4<u32>(t_in[b], t_in[b + 1u], t_in[b + 2u], t_in[b + 3u]), vec2<u32>(t_in[b + 4u], t_in[b + 5u]));
    t_out[4u * i] = w.x;
    t_out[4u * i + 1u] = w.y;
    t_out[4u * i + 2u] = w.z;
    t_out[4u * i + 3u] = w.w;
}

// Input per item: a, b. Output: hi, lo, u01_f32(a) as bits.
@compute @workgroup_size(64)
fn t_mulhilo(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= t_u[0].x) { return; }
    let r = mulhilo(t_in[2u * i], t_in[2u * i + 1u]);
    t_out[3u * i] = r.x;
    t_out[3u * i + 1u] = r.y;
    t_out[3u * i + 2u] = bitcast<u32>(u01_f32(t_in[2u * i]));
}

// Input per item: g (3), u1, u2 (f32 bits). Uniform: n, kind, eps bits, vc bits; [1].x = p bits.
// Output per item: n_hat (3), delta (3), loss, eps, hit, |n_hat|^2 - 1... (8 words used of 12).
@compute @workgroup_size(64)
fn t_pair(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= t_u[0].x) { return; }
    let b = 5u * i;
    let g = vec3<f32>(bitcast<f32>(t_in[b]), bitcast<f32>(t_in[b + 1u]), bitcast<f32>(t_in[b + 2u]));
    let u2 = bitcast<f32>(t_in[b + 4u]);
    // u2 < 0: a receding pair, n_hat along g (gn > 0), for the check of "nothing changes".
    var n_hat = g / sqrt(dot(g, g));
    if (u2 >= 0.0) { n_hat = impact_normal(g, bitcast<f32>(t_in[b + 3u]), u2); }
    let imp = collide_pair(g, n_hat, t_u[0].y, bitcast<f32>(t_u[0].z), bitcast<f32>(t_u[0].w), bitcast<f32>(t_u[1].x));
    let o = 12u * i;
    t_out[o] = bitcast<u32>(n_hat.x);
    t_out[o + 1u] = bitcast<u32>(n_hat.y);
    t_out[o + 2u] = bitcast<u32>(n_hat.z);
    t_out[o + 3u] = bitcast<u32>(imp.d.x);
    t_out[o + 4u] = bitcast<u32>(imp.d.y);
    t_out[o + 5u] = bitcast<u32>(imp.d.z);
    t_out[o + 6u] = bitcast<u32>(imp.loss);
    t_out[o + 7u] = bitcast<u32>(imp.eps);
    t_out[o + 8u] = select(0u, 1u, imp.hit);
}

// Input per item: x, y, vx, vy, vz as DS (10 words). Uniform: [0].x n, [0].y one bits;
// [1] GM hi lo, R hi lo; [2] J2 hi lo, J4 hi lo; [3] J6 hi lo.
// Output per item: u (3 DS), v_circ (DS) = 8 words.
@compute @workgroup_size(64)
fn t_peculiar(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = bitcast<f32>(t_u[0].y);
    let i = id.x;
    if (i >= t_u[0].x) { return; }
    let b = 10u * i;
    let x = vec2<f32>(bitcast<f32>(t_in[b]), bitcast<f32>(t_in[b + 1u]));
    let y = vec2<f32>(bitcast<f32>(t_in[b + 2u]), bitcast<f32>(t_in[b + 3u]));
    let vx = vec2<f32>(bitcast<f32>(t_in[b + 4u]), bitcast<f32>(t_in[b + 5u]));
    let vy = vec2<f32>(bitcast<f32>(t_in[b + 6u]), bitcast<f32>(t_in[b + 7u]));
    let vz = vec2<f32>(bitcast<f32>(t_in[b + 8u]), bitcast<f32>(t_in[b + 9u]));
    let gm = bitcast<vec2<f32>>(t_u[1].xy);
    let rr = bitcast<vec2<f32>>(t_u[1].zw);
    let j2 = bitcast<vec2<f32>>(t_u[2].xy);
    let j4 = bitcast<vec2<f32>>(t_u[2].zw);
    let j6 = bitcast<vec2<f32>>(t_u[3].xy);
    let u = peculiar_ds(x, y, vx, vy, vz, gm, rr, j2, j4, j6);
    let rho = ds_sqrt(ds_add(ds_mul(x, x), ds_mul(y, y)));
    let vc = vcirc_ds(rho, gm, rr, j2, j4, j6);
    let o = 8u * i;
    t_out[o] = bitcast<u32>(u.x.x);
    t_out[o + 1u] = bitcast<u32>(u.x.y);
    t_out[o + 2u] = bitcast<u32>(u.y.x);
    t_out[o + 3u] = bitcast<u32>(u.y.y);
    t_out[o + 4u] = bitcast<u32>(u.z.x);
    t_out[o + 5u] = bitcast<u32>(u.z.y);
    t_out[o + 6u] = bitcast<u32>(vc.x);
    t_out[o + 7u] = bitcast<u32>(vc.y);
}
