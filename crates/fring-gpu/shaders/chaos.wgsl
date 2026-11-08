// Chaos indicator (task G3, ICR-003, ADR 0008). Append to ds.wgsl, trig.wgsl
// and wh.wgsl. Rust twin: crates/fring-gpu/src/chaos.rs.
//
// Extra bindings (group 0), used by the entry point chaos_renorm only:
// 5: shadow state, same layout as binding 0.
// 6: accumulators, 4 planes of cap vec2<f32>: plane 0 = L (DS), 1 = S (DS),
//    2 = sum of Y (DS), 3 = (j, flags) as f32. Particle i is at plane * cap + i.
// 7: constants, vec4: x = d0 (km).
// Binding 0 is the state of the particle, 3 is the control (y = particles,
// z = cap), 2 the run constants (ds_one) and 4 the status buffer (ICR-004: a
// lost particle is frozen with its shadow; its accumulators stop and its flags
// get bit 1, value 2).

@group(0) @binding(5) var<storage, read_write> sh: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> acc: array<vec2<f32>>;
@group(0) @binding(7) var<uniform> cpr: vec4<f32>;

fn load_sp(i: u32) -> V3 {
    let c = prm_cap();
    return V3(sh[i], sh[c + i], sh[2u * c + i]);
}
fn load_sv(i: u32) -> V3 {
    let c = prm_cap();
    return V3(sh[3u * c + i], sh[4u * c + i], sh[5u * c + i]);
}
fn store_s(i: u32, p: V3, v: V3) {
    let c = prm_cap();
    sh[i] = p.x; sh[c + i] = p.y; sh[2u * c + i] = p.z;
    sh[3u * c + i] = v.x; sh[4u * c + i] = v.y; sh[5u * c + i] = v.z;
}

// One renormalisation of particle i (the global step is a multiple of K).
@compute @workgroup_size(WG_SIZE)
fn chaos_renorm(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    ds_one = prm_one();
    let i = id.y * nw.x * u32(WG_SIZE) + id.x;
    if (i >= prm_n_particles()) { return; }
    let c = prm_cap();
    let d0 = cpr.x;
    if (is_dropped(i)) {
        // The pair is frozen (main and shadow are not advanced): no renorm.
        var f3 = acc[3u * c + i];
        f3.y = f32(u32(f3.y) | 2u);
        acc[3u * c + i] = f3;
        return;
    }
    let p = load_p(i);
    let v = load_v(i);
    let q = load_sp(i);
    let w = load_sv(i);
    let dp = V3(ds_sub(q.x, p.x), ds_sub(q.y, p.y), ds_sub(q.z, p.z));
    let d = sqrt(dp.x.x * dp.x.x + dp.y.x * dp.y.x + dp.z.x * dp.z.x);
    var a3 = acc[3u * c + i];
    let j = a3.x + 1.0;
    var flags = a3.y;
    var ell = 0.0;
    if (d > 0.0 && d < 1.0e18) {
        ell = log(d / d0);
        let s = d0 / d;
        let dv = V3(ds_sub(w.x, v.x), ds_sub(w.y, v.y), ds_sub(w.z, v.z));
        store_s(i, v3_add(p, v3_mul_f(dp, s)), v3_add(v, v3_mul_f(dv, s)));
    } else {
        flags = 1.0;
    }
    acc[i] = ds_add_f(acc[i], ell);
    let sm = ds_add_f(acc[c + i], (j - 0.5) * ell);
    acc[c + i] = sm;
    acc[2u * c + i] = ds_add_f(acc[2u * c + i], 2.0 * sm.x / j);
    a3 = vec2<f32>(j, flags);
    acc[3u * c + i] = a3;
}
