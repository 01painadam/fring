// Precision probe for task G1 (append to ds.wgsl and trig.wgsl). Each entry
// point reads four f32 per element from `inp` and writes four f32 per element
// to `outp`. The host (src/probe.rs) compares the results with exact f64
// values and with the Rust twins.

@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> guard: vec4<f32>;

// ---- Plain (unguarded) algorithms, to show what fast math does ----

fn two_sum_plain(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, e);
}

fn fast_two_sum_plain(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    return vec2<f32>(s, b - (s - a));
}

// Veltkamp split and Dekker's product (no fma).
fn split_plain(a: f32) -> vec2<f32> {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    return vec2<f32>(hi, a - hi);
}

fn two_prod_dekker_plain(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let sa = split_plain(a);
    let sb = split_plain(b);
    let e = ((sa.x * sb.x - p) + sa.x * sb.y + sa.y * sb.x) + sa.y * sb.y;
    return vec2<f32>(p, e);
}

fn ds_add_plain(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum_plain(a.x, b.x);
    let t = two_sum_plain(a.y, b.y);
    let v = fast_two_sum_plain(s.x, s.y + t.x);
    return fast_two_sum_plain(v.x, t.y + v.y);
}

fn ds_mul_plain(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let c = two_prod(a.x, b.x);
    let cl2 = fma(a.y, b.x, fma(a.x, b.y, a.y * b.y));
    return fast_two_sum_plain(c.x, c.y + cl2);
}

@compute @workgroup_size(64)
fn probe_sum_plain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let s = two_sum_plain(a, b);
    let swap = abs(b) > abs(a);
    let f = fast_two_sum_plain(select(a, b, swap), select(b, a, swap));
    outp[i] = vec4<f32>(s.x, s.y, f.x, f.y);
}

@compute @workgroup_size(64)
fn probe_ds_plain(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let s = ds_add_plain(inp[i].xy, inp[i].zw);
    let p = ds_mul_plain(inp[i].xy, inp[i].zw);
    outp[i] = vec4<f32>(s.x, s.y, p.x, p.y);
}

// ---- Guarded library (ds.wgsl, trig.wgsl) ----

@compute @workgroup_size(64)
fn probe_sum(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let s = two_sum(a, b);
    let swap = abs(b) > abs(a);
    let f = fast_two_sum(select(a, b, swap), select(b, a, swap));
    outp[i] = vec4<f32>(s.x, s.y, f.x, f.y);
}

// Product: fma error term, a plain `a * b - p` (fused by the compiler?), and
// Dekker's product without fma.
@compute @workgroup_size(64)
fn probe_prod(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let t = two_prod(a, b);
    let plain = a * b - t.x;
    let d = two_prod_dekker_plain(a, b);
    outp[i] = vec4<f32>(t.x, t.y, plain, d.y);
}

@compute @workgroup_size(64)
fn probe_divsqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    outp[i] = vec4<f32>(a / b, sqrt(abs(a)), inverseSqrt(abs(a)), 1.0 / b);
}

// Built-in and own sine and cosine of x.
@compute @workgroup_size(64)
fn probe_trig(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let x = inp[i].x;
    let own = sin_cos_f32(x);
    outp[i] = vec4<f32>(sin(x), cos(x), own.x, own.y);
}

@compute @workgroup_size(64)
fn probe_ds(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let s = ds_add(inp[i].xy, inp[i].zw);
    let p = ds_mul(inp[i].xy, inp[i].zw);
    outp[i] = vec4<f32>(s.x, s.y, p.x, p.y);
}

@compute @workgroup_size(64)
fn probe_ds_div_sqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let q = ds_div(inp[i].xy, inp[i].zw);
    let r = ds_sqrt(inp[i].xy * sign(inp[i].x));
    outp[i] = vec4<f32>(q.x, q.y, r.x, r.y);
}

// DS sine and 1 - cosine of a small angle (z, w) in [-0.25, 0.25].
@compute @workgroup_size(64)
fn probe_ds_trig(@builtin(global_invocation_id) id: vec3<u32>) {
    ds_one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let t = ds_sin_cos_small(inp[i].zw);
    outp[i] = vec4<f32>(t.s.x, t.s.y, t.omc.x, t.omc.y);
}
