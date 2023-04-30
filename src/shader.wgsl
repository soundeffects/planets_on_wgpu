struct UniformData {
    view_position: vec4<f32>,
    view_projection: mat4x4<f32>,
    planet_rotation: mat4x4<f32>,
    noise_offset: vec4<f32>,
    cloud_offset: vec4<f32>,
    clouds_disabled: vec4<i32>,
}
@group(0) @binding(0)
var<uniform> data: UniformData;

struct VertexInput {
    @location(0) position: vec3<f32>
};

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) view_direction: vec3<f32>,
  @location(2) noise_coordinate: vec3<f32>,
  @location(3) heat: f32,
  @location(4) rough_height: f32,
  @location(5) precipitation: f32,
  @location(6) cloud_cover: f32,
};

const sea_level = 0.2;
const sky_clearness = -0.1;
const wetness = 0.5;
const snow_caps = 0.7;
const coldness = 0.4;
const cloud_speed = 0.0005;
const light_direction = vec3<f32>(1., 0., 0.);
const wet_color = vec3<f32>(0.08, 0.16, 0.15);
const dry_color = vec3<f32>(0.8, 0.6, 0.5);
const grass_color = vec3<f32>(0.2, 0.25, 0.2);
const snow_color = vec3<f32>(1.0, 1.0, 1.0);
const shallow_color = vec3<f32>(0.1, 0.2, 0.19);
const deep_color = vec3<f32>(0.02, 0.07, 0.12);
const cliff_color = vec3<f32>(0.17, 0.15, 0.2);
const cloud_color = vec3<f32>(0.9, 0.9, 1.0);
const specular_color = vec3<f32>(1., 1., 1.);
const sunrise_color = vec3<f32>(0.7, 0.3, 0.);
const day_color = vec3<f32>(0.4, 0.4, 1.);
const night_color = vec3<f32>(0.1, 0.1, 0.1);

@vertex
fn vs_main(
    vertex: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    // Rotate the planet and send resulting positions
    let rotated_position = data.planet_rotation * vec4<f32>(vertex.position, 1.0);
    out.normal = normalize(rotated_position.xyz);
    out.view_direction = normalize(data.view_position.xyz - rotated_position.xyz);
    out.clip_position = data.view_projection * rotated_position;

    // Create a noise coordinate using both position and a random offset
    out.noise_coordinate = vertex.position + data.noise_offset.xyz;

    // Rough approximation of the latitude (angle above equator)
    let latitude = 1.0 - dot(normalize(vertex.position), normalize(vec3<f32>(vertex.position.x, 0.0, vertex.position.z)));

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
    let vertical_heat = 1.0 - latitude * 1.3;
    let heat_noise = 0.2 + rebound(octave_1) * 0.8;
    out.heat = bound(vertical_heat * heat_noise, 0.0, 1.0);

    // Create a rough height map out of ice caps (high latitude), several
    // noise layers, and a global sea level
    let ice_caps = bound(latitude - 0.5, 0.0, 0.2) * 4.5;
    let height_noise = octave_1 + octave_4 * 0.65 + octave_7 * 0.25 - sea_level;
    out.rough_height = height_noise + ice_caps;

    // Create a precipitation map made out of an inverted height map (land gets
    // less rain than sea) and some noise
    let inverse_height = out.rough_height * 0.5;
    let precipitation_noise = octave_8 * 0.3;
    out.precipitation = bound(wetness - inverse_height + precipitation_noise, 0.0, 1.0);

    if (data.clouds_disabled.x == 0) {
        // Create wind speed, which effects how elongated (on the x-z plane) clouds
        // are
        let wind_speed = octave_2 * octave_3 + octave_5 * 0.5;
        let wind_speed_coordinate = vec3<f32>(
            out.noise_coordinate.x * rebound(wind_speed) * 2.0,
            out.noise_coordinate.y * 3.0,
            out.noise_coordinate.z * rebound(wind_speed) * 2.0
        );
        
        // Adjust cloud coordinate to cloud offset (for moving clouds)
        let cloud_coordinate = wind_speed_coordinate + data.cloud_offset.xyz * cloud_speed;

        // For cloud cover, use composite noise of wind speed, along with a few
        // additional noise layers
        let cloud_noise_1 = simplex(cloud_coordinate);
        let cloud_noise_2 = simplex(cloud_coordinate * 3.05);
        let cloud_noise_3 = simplex(cloud_coordinate * 6.05);
        out.cloud_cover = cloud_noise_1 + cloud_noise_2 * 0.5 + cloud_noise_3 * 0.15 - sky_clearness;
    }

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Add detail to height map
    let height_1 = simplex(in.noise_coordinate * 12.0);
    let height_2 = simplex(in.noise_coordinate * 32.0);
    let height_3 = simplex(in.noise_coordinate * 64.0);
    let height_noise = height_1 * 0.2 + height_2 * 0.1 + height_3 * 0.05;
    let detailed_height = in.rough_height + height_noise;

    var detailed_cloud_cover: f32;
    if (data.clouds_disabled.x == 0) {
        // Add detail to cloud map
        let cloud_coordinate = in.noise_coordinate + data.cloud_offset.xyz * cloud_speed;
        let cloud_1 = simplex(cloud_coordinate * 16.0);
        let cloud_2 = simplex(cloud_coordinate * 28.0);
        let cloud_noise = cloud_1 * 0.2 + cloud_2 * 0.2;
        detailed_cloud_cover = in.cloud_cover + cloud_noise;
    }   

    // Compute the atmosphere color based on the angle of incident as light
    // enters the atmosphere
    let light_incident = rebound(dot(in.normal, light_direction));
    let night_ramp = logistic(light_incident, 0.45, 20.0);
    let day_ramp = logistic(light_incident, 0.55, 20.0);
    var atmosphere_color = blend(night_ramp, 1.0, night_color, sunrise_color);
    atmosphere_color = blend(day_ramp, 1.0, atmosphere_color, day_color);

    // Create context variables for use while branching
    let height_dx = dpdx(detailed_height);
    let height_dy = dpdy(detailed_height);
    var geometry_term: f32;
    var terrain_color: vec3<f32>;

    // Branch based on our surface maps for coloring the terrain
    // First branch deals with land (heights greater than zero)
    if (detailed_height >= 0.) {
        // Blend between dry and wet
        let wet_layer = blend(in.precipitation, 0.45, dry_color, wet_color);

        // Blend between cliffs and flatlands
        let ridge_layer = blend(detailed_height, 1.0, wet_layer, cliff_color);

        // Add snow at the tops of ridges
        let snow_ramp = logistic(detailed_height, snow_caps, 10.0);
        let ridge_snow_layer = blend(snow_ramp, 1.0, ridge_layer, snow_color);

        // Blend between cold poles and warm equatorial regions
        let equator_ramp = logistic(in.heat, coldness, 10.0);
        let cold_layer = blend(equator_ramp, 0.5, snow_color, ridge_snow_layer);

        // Add some grass color to flatlands
        let grass_layer = blend(detailed_height, 0.4, grass_color, vec3<f32>(0.));

        // Composite the final terrain color
        terrain_color = cold_layer + grass_layer * 0.4;

        // Add screen-space derivatives of height to the normal for a subtle
        // shadow on ridges at sunrise
        let terrain_normal = vec3<f32>(
            in.normal.x + height_dx * 0.3,
            in.normal.y + height_dy * 0.3,
            in.normal.z
        );

        // Compute geometry term based on this adjusted normal
        geometry_term = max(dot(terrain_normal, light_direction), 0.0);


    // Now we deal with water (heights less than zero)
    } else {
        // Blend between shallow and deep oceans
        let depth = abs(detailed_height);
        terrain_color = blend(depth, 0.1, shallow_color, deep_color) / (depth * 0.2 + 1.0);

        // Compute geometry term trivially, since the ocean's surface does not
        // affect the normal
        geometry_term = max(dot(in.normal, light_direction), 0.0);

        // Add specular lighting to the water
        let half_vector = normalize(normalize(data.view_position.xyz) + light_direction);
        let blinn_term = pow(max(dot(in.normal, half_vector), 0.0), 40.0);
        terrain_color += atmosphere_color * blinn_term * geometry_term * 0.2;
    }

    // Some more context variables for use when branching at clouds
    let cloud_dx = dpdx(detailed_cloud_cover);
    let cloud_dy = dpdy(detailed_cloud_cover);

    // Blend with cloud layer
    if (data.clouds_disabled.x == 0 && detailed_cloud_cover >= 0.) {
        // Blend into the cloud color when covered by clouds
        terrain_color = blend(detailed_cloud_cover, 0.5, terrain_color, cloud_color);

        // Add screen-space derivatives of cloud cover to the normal for a
        // subtle bumpiness
        let cloud_normal = vec3<f32>(
            in.normal.x + cloud_dx * 0.15,
            in.normal.y + cloud_dy * 0.15,
            in.normal.z
        );

        // Overwrite geometry term for clouds
        geometry_term = max(dot(cloud_normal, light_direction), 0.0);
    }

    // Now we can compute diffuse and ambient lighting for the given color
    let diffuse_color = vec4<f32>(terrain_color, 1.0);
    let diffuse_component = diffuse_color * geometry_term;
    let ambient_component = diffuse_color * 0.01;

    // Add atmosphere color if the viewing angle of incident is steep (meaning
    // atmosphere color will be added at the edges of the sphere)
    let view_incident = 1.0 - max(dot(in.view_direction, in.normal), 0.0);
    let view_incident_ramp = logistic(view_incident, 0.8, 10.0);
    let atmosphere_component = vec4<f32>(atmosphere_color, 1.0) * view_incident_ramp * 0.2;

    // Composite all color components
    return diffuse_component + ambient_component + atmosphere_component;
}

// Helpers

fn logistic(x: f32, x_0: f32, k: f32) -> f32 {
    return 1.0 / (1.0 + exp(-k * (x - x_0)));
}

fn bound(x: f32, bound_min: f32, bound_max: f32) -> f32 {
    return min(max(x, bound_min), bound_max);
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
  let m2 = m * m;
  let m4 = m2 * m2;

  let px = vec4<f32>(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3));

  return 42. * dot(m4, px);
}