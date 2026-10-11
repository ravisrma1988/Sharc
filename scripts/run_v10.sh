#!/bin/bash
# v10 preregistered deployment grid (PREREG_DEPLOYMENT_GRID.md, SHA-256 0518621e46f6a5e2...). Frozen v9 predictions; no training.
export TZ=Asia/Kolkata; export BNULL=200
cd /path/to/sharc_work/scripts; P=python; rm -f ../results/e_v10_SMOKETEST_raw.parquet ../results/e_smoke10_raw.parquet
echo "prereg sha256: $(sha256sum ../PREREG_DEPLOYMENT_GRID.md | cut -c1-64)"
for cfg in "convnext_tiny 0,1,2,3,4 _convnext5" "resnet50 0,1,2 _resnet50" "dinov2s_frozen 0,1,2 _dinov2"; do
  set -- $cfg; echo "== $(date '+%F %T IST') v10 $1"
  BACKBONE=$1 SEEDS=$2 REPS=100 RUN_TAG=$3 $P e_v10.py > ../logs/v10$3.log 2>&1; grep -E 'stage|DONE' ../logs/v10$3.log; grep -c Traceback ../logs/v10$3.log
done
echo "== $(date '+%F %T IST') finished"; touch ../logs/V10_DONE
