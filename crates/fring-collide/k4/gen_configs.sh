#!/bin/sh
# Writes the K4 configs. Usage: gen_configs.sh <name> <model> <seed> <count> <width_deg> <years> <vc> [cx cy cz every h frames]
name=$1; model=$2; seed=$3; count=$4; width=$5; years=$6; vc=$7
cx=${8:-0.7}; cy=${9:-0.8}; cz=${10:-0.03}; every=${11:-1}; h=${12:-60.0}; fr=${13:-1782}
if [ "$model" = stochastic ]; then
COL="model = \"stochastic\"
particle_radius_km = 0.01
weight = 1.0
cell_radial_km = $cx
cell_azimuthal_km = $cy
cell_vertical_km = $cz
every_steps = $every"
else
COL="model = \"hard_sphere\"
particle_radius_km = 0.01"
fi
cat > configs/$name.toml <<EOT
# K4 cross-check (stochastic against hard sphere), arc, core e = 0 plus velocity seed e = 4e-7.
# Form test: v_c scaled so that c_eq is about 6 Omega s (s = 10 m); not a physical constant.
[run]
name = "$name"
backend = "cpu"
seed = $seed
params_dir = "params"

[time]
step_s = $h
duration_years = $years

[frame]
reference_semi_major_axis_km = 140220.0

[domain]
kind = "arc"
width_deg = $width
centre_longitude_deg = 0.0

[moons]
include = []

[particles]
count = $count

[particles.distribution]
kind = "core"
semi_major_axis_km = 140220.0
sigma_semi_major_axis_km = 3.0
eccentricity = 4e-7
apse_longitude_deg = 0.0

[collisions]
$COL

[collisions.restitution]
law = "power_law"
exponent = 0.234
reference_speed_km_s = $vc

[output]
frame_every_steps = $fr
stats_every_steps = 89
velocities = true
EOT
