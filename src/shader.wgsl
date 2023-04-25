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
  @location(0) normal: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) bitangent: vec3<f32>,
  @location(3) noise_coordinate: vec3<f32>,
  @location(4) latitude: f32,
  @location(5) heat: f32,
  @location(6) rough_height: f32,
  @location(7) precipitation: f32,
  @location(8) cloud_cover: f32,
};

const sea_level = 0.2;
const sky_clearness = 0.1;
const light_direction = vec3<f32>(1., 0., 0.);
const wet_color = vec3<f32>(0.08, 0.16, 0.15);
const dry_color = vec3<f32>(0.8, 0.6, 0.5);
const grass_color = vec3<f32>(0.2, 0.25, 0.2);
const snow_color = vec3<f32>(1.0, 1.0, 1.0);
const shallow_color = vec3<f32>(0.1, 0.2, 0.19);
const deep_color = vec3<f32>(0.02, 0.05, 0.1);
const cliff_color = vec3<f32>(0.17, 0.15, 0.2);
const specular_color = vec3<f32>(1., 1., 1.);
const sunrise_color = vec3<f32>(1.0, 0.6, 0.2);
const noon_color = vec3<f32>(0.6, 0.8, 1.0);

@vertex
fn vs_main(
    vertex: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    // Rotate the planet and send resulting positions
    let rotated_position = data.planet_rotation * vec4<f32>(vertex.position, 1.0);
    out.normal = normalize(rotated_position.xyz);
    out.clip_position = data.view_projection * rotated_position;

    // Find the tangent (east) and bitangent (north) vectors for this position
    // on the sphere
    let normal = normalize(vertex.position);
    out.tangent = cross(vec3<f32>(0., 1., 0.), normal);
    out.bitangent = cross(normal, out.tangent);

    // Create a noise coordinate using both position and a random offset
    out.noise_coordinate = vertex.position + data.noise_offset.xyz;

    // Rough approximation of the latitude (angle above equator)
    out.latitude = 1.0 - dot(normalize(vertex.position), normalize(vec3<f32>(vertex.position.x, 0.0, vertex.position.z)));

    // Compute noise octaves for generating surface maps
    let octave_1 = simplex(out.noise_coordinate);
    let octave_2 = simplex(out.noise_coordinate * 1.5);
    let octave_3 = simplex(out.noise_coordinate * 1.6);
    let octave_4 = simplex(out.noise_coordinate * 2.0);
    let octave_5 = simplex(out.noise_coordinate * 3.0);
    let octave_6 = simplex(out.noise_coordinate * 3.1);
    let octave_7 = simplex(out.noise_coordinate * 4.0);
    let octave_8 = simplex(out.noise_coordinate * 6.0);
    let octave_9 = simplex(out.noise_coordinate * 6.1);

    // Create a slightly noisy heat map, with equatorial regions getting hotter
    let vertical_heat = 1.0 - out.latitude * 1.3;
    let heat_noise = 0.2 + rebound(octave_1) * 0.8;
    out.heat = bound(vertical_heat * heat_noise, 0.0, 1.0);

    // Create a rough height map out of ice caps (high latitude), several
    // noise layers, and a global sea level
    let ice_caps = bound(out.latitude - 0.5, 0.0, 0.2) * 4.5;
    let height_noise = octave_1 + octave_4 * 0.65 + octave_7 * 0.25 - sea_level;
    out.rough_height = height_noise + ice_caps;

    // Create a precipitation map made out of an inverted height map (land gets
    // less rain than sea) and some noise
    let inverse_height = 0.3 - out.rough_height * 0.9;
    let precipitation_noise = octave_8 * 0.3;
    out.precipitation = bound(inverse_height + precipitation_noise, 0.0, 1.0);

    // Create wind speed, which effects how elongated (on the x-z plane) clouds
    // are
    let wind_speed = octave_2 * octave_3 + octave_5 * 0.5;
    let wind_noise_coordinate = vec3<f32>(
        out.noise_coordinate.x * rebound(wind_speed) * 2.0,
        out.noise_coordinate.y * 3.0,
        out.noise_coordinate.z * rebound(wind_speed) * 2.0
    );

    // For cloud cover, use composite noise of wind speed, along with a few
    // additional noise layers
    let cloud_noise_1 = simplex(wind_noise_coordinate);
    let cloud_noise_2 = octave_6;
    let cloud_noise_3 = octave_8 * octave_9;
    out.cloud_cover = cloud_noise_1 + cloud_noise_2 * 0.5 + cloud_noise_3 * 0.15 - sky_clearness;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create high frequency noise octaves to add detail to surface maps
    let octave_1 = simplex(in.noise_coordinate * 12.0);
    let octave_2 = simplex(in.noise_coordinate * 16.0);
    let octave_3 = simplex(in.noise_coordinate * 32.0);
    let octave_4 = simplex(in.noise_coordinate * 28.0);
    let octave_5 = simplex(in.noise_coordinate * 64.0);

    // Add detail to height map
    let height_noise = octave_1 * 0.2 + octave_3 * 0.1 + octave_5 * 0.05;
    let detailed_height = in.rough_height + height_noise;

    // Add detail to cloud map
    let cloud_noise = octave_2 * 0.2 + octave_4 * 0.5;
    let detailed_cloud_cover = in.cloud_cover + cloud_noise;

    // Create context variables for use while branching
    let raw_geometry_term = dot(in.normal, light_direction);
    let geometry_term = max(raw_geometry_term, 0.0);
    let atmosphere_geometry_term = max(logistic(raw_geometry_term, 0.1, 1.), 0.0);
    let atmosphere_color = blend(atmosphere_geometry_term, 0.75, sunrise_color, noon_color);
    var terrain_color: vec3<f32>;
    var specular_component: vec4<f32>;

    // Branch based on our surface maps for coloring the terrain
    // First branch deals with land (heights greater than zero)
    if (detailed_height >= 0.) {
        terrain_color = dry_color;

        // Usually, at a planet scale, the speculars will only show on the
        // surface of the ocean. Therefore, we leave the specular component
        // black.
        specular_component = vec4<f32>(0.0);

    // Now we deal with water (heights less than zero)
    } else {
        let depth = abs(detailed_height);
        terrain_color = blend(depth, 0.1, shallow_color, deep_color) / (depth * 0.2 + 1.0);

        // Compute specular component
        let half_vector = normalize(normalize(data.view_position.xyz) + light_direction);
        let blinn_term = pow(max(dot(in.normal, half_vector), 0.0), 40.0);
        specular_component = vec4<f32>(atmosphere_color, 1.0) * blinn_term * geometry_term * 0.2;
    }

    // Now that we have the terrain color, the diffuse/ambient color is easy to
    // compute
    let diffuse_color = vec4<f32>(terrain_color, 1.0);
    let diffuse_component = diffuse_color * geometry_term;
    let ambient_component = diffuse_color * 0.01;

    let haze_component = vec4<f32>(atmosphere_color, 1.0) * band(atmosphere_geometry_term, 0.0, 0.15) * 0.2;

    // Composite all color components
    return diffuse_component + ambient_component + specular_component + haze_component;
}

// Helpers

fn logistic(x: f32, x_0: f32, k: f32) -> f32 {
    return 1.0 / (1.0 + exp(-k * (x - x_0)));
}

fn band(x: f32, min: f32, max: f32) -> f32 {
    return logistic(x, min, 20.0) - logistic(x, max, 20.0);
}

fn bound(x: f32, min: f32, max: f32) -> f32 {
    return min(max(x, min), max);
}

fn rebound(x: f32) -> f32 {
    return x * 0.5 + 0.5;
}

fn blend(x: f32, limit: f32, from_color: vec3<f32>, to_color: vec3<f32>) -> vec3<f32> {
    let factor = 1.0 / limit;
    let discriminator = bound(x * factor, 0.0, 1.0);
    return from_color * (1.0 - discriminator) + to_color * discriminator;
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