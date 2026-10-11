#!/bin/bash
# Queued after run_final3.sh: deduplicated real-site sensitivity run (ConvNeXt) and the efficiency benchmark (quiet machine).
export TZ=Asia/Kolkata; export BNULL=200; export SPLIT=1
cd /path/to/sharc_work/scripts; P=python
while [ ! -f ../logs/FINAL3_DONE ]; do sleep 60; done
echo "== $(date '+%F %T IST') dedup E2 sensitivity"
STAGES=E2 DEDUP=1 BACKBONE=convnext_tiny SEEDS=0,1,2,3,4 REPS=100 RUN_TAG=_convnext5_dedup $P e_v8.py > ../logs/v8_dedup.log 2>&1; tail -n 1 ../logs/v8_dedup.log
echo "== $(date '+%F %T IST') efficiency benchmark"
$P bench.py > ../logs/bench.log 2>&1; tail -n 1 ../logs/bench.log
echo "== $(date '+%F %T IST') finished"; touch ../logs/AFTER3_DONE
