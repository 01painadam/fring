// Precision probe for task G1 (append to ds.wgsl). Each entry point reads
// four f32 per element from `inp` and writes four f32 per element to `outp`.
// The host (src/probe.rs) compares the results with exact f64 values.

@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<f32>>;

// TwoSum and Fast2Sum of (x, y): does the compiler keep the error terms?
@compute @workgroup_size(64)
fn probe_sum(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let s = two_sum(a, b);
    var big = a;
    var small = b;
    if (abs(b) > abs(a)) {
        big = b;
        small = a;
    }
    let f = fast_two_sum(big, small);
    outp[i] = vec4<f32>(s.x, s.y, f.x, f.y);
}

// Product of (x, y): fma error term, a plain `x * y - p` (to see whether the
// compiler fuses it), and Dekker's product without fma.
@compute @workgroup_size(64)
fn probe_prod(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let t = two_prod(a, b);
    let plain = a * b - t.x;
    let d = two_prod_dekker(a, b);
    outp[i] = vec4<f32>(t.x, t.y, plain, d.y);
}

// Division, square root, inverse square root, reciprocal.
@compute @workgroup_size(64)
fn probe_divsqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    outp[i] = vec4<f32>(a / b, sqrt(abs(a)), inverseSqrt(abs(a)), 1.0 / b);
}

// Built-in sine and cosine of x; sine and cosine of z (small angles).
@compute @workgroup_size(64)
fn probe_trig(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let x = inp[i].x;
    let z = inp[i].z;
    outp[i] = vec4<f32>(sin(x), cos(x), sin(z), cos(z));
}

// DS sum and product of (x, y) and (z, w).
@compute @workgroup_size(64)
fn probe_ds(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].xy;
    let b = inp[i].zw;
    let s = ds_add(a, b);
    let p = ds_mul(a, b);
    outp[i] = vec4<f32>(s.x, s.y, p.x, p.y);
}

// DS quotient of (x, y) by (z, w).
@compute @workgroup_size(64)
fn probe_dsdiv(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let q = ds_div(inp[i].xy, inp[i].zw);
    outp[i] = vec4<f32>(q.x, q.y, 0.0, 0.0);
}

// Work-around test: a runtime 1.0 that the compiler cannot see (from a
// uniform buffer) hides the sums from re-association.
@group(0) @binding(2) var<uniform> guard: vec4<f32>;
var<private> one: f32;

fn two_sum_g(a: f32, b: f32) -> vec2<f32> {
    let s = (a + b) * one;
    let bb = (s - a) * one;
    let av = (s - bb) * one;
    let e = ((a - av) * one) + (b - bb);
    return vec2<f32>(s, e);
}

// TwoSum as Fast2Sum after ordering by magnitude (no branch).
fn two_sum_sel(a: f32, b: f32) -> vec2<f32> {
    let swap = abs(b) > abs(a);
    return fast_two_sum_g(select(a, b, swap), select(b, a, swap));
}

fn two_prod_g(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    return vec2<f32>(p, fma(a, b, -p));
}

fn ds_add_g(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum_sel(a.x, b.x);
    let t = two_sum_sel(a.y, b.y);
    let c = s.y + t.x;
    let v = fast_two_sum_g(s.x, c);
    let w = t.y + v.y;
    return fast_two_sum_g(v.x, w);
}

fn ds_mul_g(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let c = two_prod_g(a.x, b.x);
    let tl0 = a.y * b.y;
    let tl1 = fma(a.x, b.y, tl0);
    let cl2 = fma(a.y, b.x, tl1);
    let cl3 = c.y + cl2;
    return fast_two_sum_g(c.x, cl3);
}

@compute @workgroup_size(64)
fn probe_sum_sel(@builtin(global_invocation_id) id: vec3<u32>) {
    one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let s = two_sum_sel(inp[i].x, inp[i].y);
    outp[i] = vec4<f32>(s.x, s.y, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn probe_ds_g(@builtin(global_invocation_id) id: vec3<u32>) {
    one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].xy;
    let b = inp[i].zw;
    let s = ds_add_g(a, b);
    let p = ds_mul_g(a, b);
    outp[i] = vec4<f32>(s.x, s.y, p.x, p.y);
}

fn fast_two_sum_g(a: f32, b: f32) -> vec2<f32> {
    let s = (a + b) * one;
    let z = (s - a) * one;
    return vec2<f32>(s, b - z);
}

@compute @workgroup_size(64)
fn probe_sum_g(@builtin(global_invocation_id) id: vec3<u32>) {
    one = guard.x;
    let i = id.x;
    if (i >= arrayLength(&inp)) { return; }
    let a = inp[i].x;
    let b = inp[i].y;
    let s = two_sum_g(a, b);
    var big = a;
    var small = b;
    if (abs(b) > abs(a)) {
        big = b;
        small = a;
    }
    let f = fast_two_sum_g(big, small);
    outp[i] = vec4<f32>(s.x, s.y, f.x, f.y);
}
