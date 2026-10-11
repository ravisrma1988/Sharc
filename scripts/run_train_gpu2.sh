#!/bin/bash
cd /path/to/sharc_work
for s in 0 1 2; do
  python scripts/train_gpu.py $s convnext_tiny.fb_in22k 15 > logs/traingpu_convnext_s$s.log 2>&1
done
echo TRAIN_ALL_DONE > logs/train_done.flag
