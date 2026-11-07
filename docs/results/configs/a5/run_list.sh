#!/bin/zsh
# A5-ANALYSIS: runs every config named in a list file, one after the other (one GPU run at a time).
# usage (from the repository root): docs/results/configs/a5/run_list.sh <list.txt> <tag>
# Raw runs go to /Users/adampain/Projects/fring/runs/<name> (git-ignored), logs to runs/logs/ there.
RUNS=/Users/adampain/Projects/fring/runs
mkdir -p $RUNS/logs
STATUS=$RUNS/logs/a5-status-$2.csv
[ -f $STATUS ] || echo "name,exit,seconds" > $STATUS
for n in $(cat $1); do
  t0=$(date +%s)
  ./target/release/fring-run --config docs/results/configs/a5/$n.toml --out $RUNS/$n --quiet > $RUNS/logs/$n.log 2>&1
  rc=$?
  echo "$n,$rc,$(( $(date +%s) - t0 ))" >> $STATUS
done
