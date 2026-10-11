#!/bin/bash
# FINAL rerun 3 (audit response): thr_crc fix + v8 complete grid (all baselines, gate/label split, three-way accounting,
# concept shift, gate-component ablations) for 3 backbones; ablations and calibration scale with the corrected code. BNULL=200.
export TZ=Asia/Kolkata; export BNULL=200; export SPLIT=1
cd /path/to/sharc_work/scripts; P=python
mkdir -p ../results/final2_pre_crcfix && mv ../results/e_v6_* ../results/e_v6scale_* ../results/e_v5abl_* ../results/paper ../results/final2_pre_crcfix/ 2>/dev/null
rm -f ../results/e_smoke8_*
for cfg in "convnext_tiny 0,1,2,3,4 _convnext5" "resnet50 0,1,2 _resnet50" "dinov2s_frozen 0,1,2 _dinov2"; do
  set -- $cfg; echo "== $(date '+%F %T IST') v8 $1"
  BACKBONE=$1 SEEDS=$2 REPS=100 RUN_TAG=$3 $P e_v8.py > ../logs/final3_v8$3.log 2>&1; tail -n 1 ../logs/final3_v8$3.log
done
echo "== $(date '+%F %T IST') ablations"
BACKBONE=convnext_tiny SEEDS=0,1,2,3,4 REPS_ABL=30 RUN_TAG=_convnext5 $P e_v5_ablate.py > ../logs/final3_v5abl.log 2>&1; tail -n 1 ../logs/final3_v5abl.log
echo "== $(date '+%F %T IST') calibration scale"
BACKBONE=convnext_tiny SEEDS=0,1,2,3,4 REPS_ABL=30 RUN_TAG=_convnext5 $P e_v6_scale.py > ../logs/final3_v6scale.log 2>&1; tail -n 1 ../logs/final3_v6scale.log
echo "== $(date '+%F %T IST') finished"; touch ../logs/FINAL3_DONE
