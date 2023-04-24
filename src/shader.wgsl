// Vertex shader
struct UniformData {
    view_position: vec4<f32>,
    view_projection: mat4x4<f32>,
    planet_rotation: mat4x4<f32>,
    noise_offset: vec4<f32>
}
@group(0) @binding(0)
var<uniform> data: UniformData;

struct VertexInput {
    @location(0) position: vec3<f32>
};

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) position: vec3<f32>,
  @location(1) planet_coordinate: vec3<f32>
};

@vertex
fn vs_main(
    vertex: VertexInput
) -> VertexOutput {
    var out: VertexOutput;
    let rotated_position = data.planet_rotation * vec4<f32>(vertex.position, 1.0);
    out.position = rotated_position.xyz * 2.0;
    out.planet_coordinate = vertex.position;
    out.clip_position = data.view_projection * rotated_position;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.position);
    let light_direction = vec3<f32>(1.0, 0.0, 0.0);
    let noise_coordinate = in.planet_coordinate + data.noise_offset.xyz;
    //let sampled_normal: vec4<f32> = textureSample(normal_texture, normal_sampler, in.texture_coordinates);
    //let normal: vec3<f32> = normalize(sampled_normal.xyz * 2.0 - 1.0);

    let latitude = 1.0 - logistic(abs(noise_coordinate.y) * 0.5, 0.5, 10.0);

    let ice_coverage = 2.0;
    let ice_caps = logistic(abs(noise_coordinate.y) * 0.5, 1.0, 50.0) * ice_coverage;

    let sea_level = 0.2;

    let heat = 0.2 + latitude - ice_caps +
        simplex(noise_coordinate * 0.5) * 0.02 +
        simplex(noise_coordinate) * 0.1;

    let precipitation = latitude * 0.1 +
        normalized_simplex(noise_coordinate * 0.5) * 0.6 +
        normalized_simplex(noise_coordinate) * 0.3;

    let height =
        ice_caps * 5.0 - sea_level +
        simplex(noise_coordinate * 0.5) +
        simplex(noise_coordinate) +
        simplex(noise_coordinate * 2.0) * 0.5 +
        simplex(noise_coordinate * 4.0) * 0.25 +
        simplex(noise_coordinate * 8.0) * 0.125 +
        simplex(noise_coordinate * 16.0) * 0.08 +
        simplex(noise_coordinate * 32.0) * 0.04 +
        simplex(noise_coordinate * 64.0) * 0.02;

    let wet_color = vec3<f32>(0.08, 0.16, 0.15);
    let dry_color = vec3<f32>(0.8, 0.6, 0.5);
    let grass_color = vec3<f32>(0.2, 0.25, 0.2);
    let snow_color = vec3<f32>(1.0, 1.0, 1.0);
    let shallow_color = vec3<f32>(0.1, 0.2, 0.19);
    let deep_color = vec3<f32>(0.02, 0.05, 0.1);
    let cliff_color = vec3<f32>(0.17, 0.15, 0.2);

    let terrain_color =
        band(height, -3.0, -0.1) * deep_color +
        band(height, -0.2, 0.0) * (
            band(heat, -3.0, 0.5) * deep_color +
            band(heat, 0.5, 1.5) * shallow_color
        ) +
        band(height, 0.0, 0.9) * (
            band(heat, -3.0, 0.1) * snow_color +
            band(heat, 0.1, 0.2) * cliff_color + 
            band(heat, 0.2, 1.5) * dry_color
        ) +
        band(height, 0.9, 20.0) * (
            band(heat, -3.0, 0.6) * snow_color +
            band(heat, 0.6, 1.5) * cliff_color
        );
    
    let diffuse_color = vec4<f32>(terrain_color, 1.0);//textureSample(diffuse_texture, diffuse_sampler, in.uv);
    let raw_geometry_term = dot(normal, light_direction);
    let geometry_term = max(raw_geometry_term, 0.0);
    let diffuse_component = diffuse_color * geometry_term;

    let ambient_component = diffuse_color * 0.005;

    let haze_color = vec4<f32>(1.0, 0.6, 0.2, 1.0);
    let haze_strength = band(raw_geometry_term, 0.05, 0.15);

    let specular_color = vec4<f32>(band(height, -3.0, 0.0));
    let half_vector = normalize(normalize(data.view_position.xyz) + light_direction);
    let blinn_term = pow(max(dot(normal, half_vector), 0.0), 40.0);
    let specular_component = (
        haze_strength * haze_color * 10.0 +
        band(raw_geometry_term, 0.15, 1.0) * specular_color
    ) * blinn_term * geometry_term * 0.2;

    
    let haze_component = haze_strength * haze_color * 0.02;
    
    return diffuse_component + ambient_component + specular_component + haze_component;
}

fn logistic(x: f32, x_0: f32, k: f32) -> f32 {
    return 1.0 / (1.0 + exp(-k * (x - x_0)));
}

fn band(x: f32, min: f32, max: f32) -> f32 {
    return logistic(x, min, 20.0) - logistic(x, max, 20.0);
}

//  Simplex algorithm in WGSL. MIT License. © Ian McEwan, Stefan Gustavson, Munrocket
//  Sourced from https://docs.rs/crate/bevy_shader_utils/0.3.0/source/shaders/simplex_noise_3d.wgsl
fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn taylorInvSqrt4(r: vec4<f32>) -> vec4<f32> { return 1.79284291400159 - 0.85373472095314 * r; }

fn simplex(v: vec3<f32>) -> f32 {
  let C = vec2<f32>(1. / 6., 1. / 3.);
  let D = vec4<f32>(0., 0.5, 1., 2.);

  // First corner
  var i: vec3<f32>  = floor(v + dot(v, C.yyy));
  let x0 = v - i + dot(i, C.xxx);

  // Other corners
  let g = step(x0.yzx, x0.xyz);
  let l = 1.0 - g;
  let i1 = min(g.xyz, l.zxy);
  let i2 = max(g.xyz, l.zxy);

  // x0 = x0 - 0. + 0. * C
  let x1 = x0 - i1 + 1. * C.xxx;
  let x2 = x0 - i2 + 2. * C.xxx;
  let x3 = x0 - 1. + 3. * C.xxx;

  // Permutations
  i = i % vec3<f32>(289.);
  let p = permute4(permute4(permute4(
      i.z + vec4<f32>(0., i1.z, i2.z, 1. )) +
      i.y + vec4<f32>(0., i1.y, i2.y, 1. )) +
      i.x + vec4<f32>(0., i1.x, i2.x, 1. ));

  // Gradients (NxN points uniformly over a square, mapped onto an octahedron.)
  var n_: f32 = 1. / 7.; // N=7
  let ns = n_ * D.wyz - D.xzx;

  let j = p - 49. * floor(p * ns.z * ns.z); // mod(p, N*N)

  let x_ = floor(j * ns.z);
  let y_ = floor(j - 7.0 * x_); // mod(j, N)

  let x = x_ *ns.x + ns.yyyy;
  let y = y_ *ns.x + ns.yyyy;
  let h = 1.0 - abs(x) - abs(y);

  let b0 = vec4<f32>( x.xy, y.xy );
  let b1 = vec4<f32>( x.zw, y.zw );

  let s0 = floor(b0)*2.0 + 1.0;
  let s1 = floor(b1)*2.0 + 1.0;
  let sh = -step(h, vec4<f32>(0.));

  let a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  let a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  var p0: vec3<f32> = vec3<f32>(a0.xy, h.x);
  var p1: vec3<f32> = vec3<f32>(a0.zw, h.y);
  var p2: vec3<f32> = vec3<f32>(a1.xy, h.z);
  var p3: vec3<f32> = vec3<f32>(a1.zw, h.w);

  // Normalise gradients
  let norm = taylorInvSqrt4(vec4<f32>(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 = p0 * norm.x;
  p1 = p1 * norm.y;
  p2 = p2 * norm.z;
  p3 = p3 * norm.w;

  // Mix final noise value
  var m: vec4<f32> = 0.6 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
  m = max(m, vec4<f32>(0.));
  m = m * m;
  return 42. * dot(m * m, vec4<f32>(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

fn normalized_simplex(v: vec3<f32>) -> f32 {
    return simplex(v) * 0.5 + 0.5;
}