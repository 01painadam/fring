// W1: draws discs (particles, moons, Saturn) as instanced quads.
//
// Instance (vertex buffer, 16 bytes, step mode instance):
//   xy = position in km, rotating frame, RELATIVE TO THE EYE (the host subtracts
//        the camera centre in f64 before it rounds to f32),
//   z  = diameter in pixels,
//   w  = colour id (0 particle, 1 Saturn, 2 + k moon k).
// Uniform: scale = clip units per km (x, y), viewport = window size in pixels.
//
// The shader does only a multiply and an add per vertex: no trigonometry and no
// division by small numbers, so Metal fast math does not change the result.

struct Frame {
    scale: vec2<f32>,
    viewport: vec2<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) colour: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) inst: vec4<f32>,
) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let centre = inst.xy * frame.scale;
    // Half the diameter in pixels, converted to clip units (the clip window is 2 wide).
    let offset = c * inst.z * 0.5 * 2.0 / frame.viewport;
    var out: VsOut;
    out.clip = vec4<f32>(centre + offset, 0.0, 1.0);
    out.corner = c;
    out.colour = inst.w;
    return out;
}

fn colour_of(id: f32) -> vec4<f32> {
    if (id < 0.5) {
        return vec4<f32>(0.62, 0.82, 1.0, 0.75);   // particle
    }
    if (id < 1.5) {
        return vec4<f32>(0.34, 0.30, 0.22, 1.0);   // Saturn (dull, behind the ring)
    }
    if (id < 2.5) {
        return vec4<f32>(1.0, 0.62, 0.20, 1.0);    // first moon (Prometheus)
    }
    return vec4<f32>(0.45, 0.95, 0.55, 1.0);       // second moon (Pandora)
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.corner);
    if (d > 1.0) {
        discard;
    }
    var col = colour_of(in.colour);
    // Soft edge, about one pixel wide for small discs.
    col.a = col.a * (1.0 - smoothstep(0.7, 1.0, d));
    return vec4<f32>(col.rgb * col.a, col.a);   // premultiplied alpha
}
