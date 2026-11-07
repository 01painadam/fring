#!/bin/zsh
# A5-ANALYSIS (b): closest approach to both moons for one a5-cl config, joined with the classes of the e10 GPU run of the same bin.
# usage (repository root): docs/results/configs/a5/run_closest.sh <ecc|circ> <bin 2 digits> [5k]
R=/Users/adampain/Projects/fring/runs
n=a5-cl${3}-$1-b$2
./target/release/fring-analyse closest docs/results/configs/a5/$n.toml $R/$n.csv prometheus,pandora - - - $R/e10-$1-b$2 > $R/logs/$n.log 2>&1
echo "$n,$?" >> $R/logs/a5-status-closest.csv
