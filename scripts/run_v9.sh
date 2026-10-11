#!/bin/bash
# v9 FINAL (theory re-audit): disjoint source pools for gate vs CRC, lineage assertions; then dedup sensitivity and the analysis package.
export TZ=Asia/Kolkata; export BNULL=200
cd /path/to/sharc_work/scripts; P=python; rm -f ../results/e_smoke9_* ../results/lineage_smoke9.csv
for cfg in "convnext_tiny 0,1,2,3,4 _convnext5" "resnet50 0,1,2 _resnet50" "dinov2s_frozen 0,1,2 _dinov2"; do
  set -- $cfg; echo "== $(date '+%F %T IST') v9 $1"
  BACKBONE=$1 SEEDS=$2 REPS=100 RUN_TAG=$3 $P e_v9.py > ../logs/v9$3.log 2>&1; tail -n 1 ../logs/v9$3.log; grep -c Traceback ../logs/v9$3.log
done
echo "== $(date '+%F %T IST') v9 dedup E2 (ConvNeXt)"
STAGES=E2 DEDUP=1 BACKBONE=convnext_tiny SEEDS=0,1,2,3,4 REPS=100 RUN_TAG=_convnext5_dedup $P e_v9.py > ../logs/v9_dedup.log 2>&1; tail -n 1 ../logs/v9_dedup.log
echo "== $(date '+%F %T IST') analysis"
export VER=v9 OUTDIR=paper_final_v9
$P v9_analysis_v8.py > ../logs/v9_analysis.log 2>&1; tail -n 1 ../logs/v9_analysis.log
$P v9_sample_flow.py > ../logs/v9_sampleflow.log 2>&1; tail -n 1 ../logs/v9_sampleflow.log
$P v9_fig_final.py > ../logs/v9_figs.log 2>&1; tail -n 1 ../logs/v9_figs.log
O=../results/paper_final_v9; cp ../results/unit_tests.csv ../results/cp_exceedance_simulation.csv ../results/lineage_v9_*.csv ../results/duplicates_confirmed_strict.csv ../results/dedup_drop_list.csv ../results/efficiency_benchmark.csv $O/ 2>/dev/null
sha256sum ../scripts/*.py ../scripts/*.sh > $O/SHA256_scripts.txt; sha256sum ../results/e_v9_*_raw.parquet ../results/lineage_v9_*.csv > $O/SHA256_results.txt
echo "== $(date '+%F %T IST') finished"; touch ../logs/V9_DONE
