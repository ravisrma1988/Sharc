# SHARC: shift-audited referral calibration

Code and aggregated results for the manuscript

> **SHARC: shift-audited referral calibration with selective target labelling for multi-site diabetic retinopathy screening**
> Ravi Sharma, Rashmi Rathi Upadhyay, Garima Rathi, Vaishali Gupta (under review)

SHARC decides, from the unlabelled images of a new screening site, whether a referral threshold calibrated on a source programme can be reused or whether a small target annotation must be acquired. It combines conformal risk control on source data, an audit of the prevalence-only shift hypothesis based on model scores and image features, and a label-until-K rule that grades target units in random order until K referable units are found.

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/` | All code used for the reported results: data preparation, model training and inference, the SHARC experiments, analysis, verification and unit tests |
| `paper_results/v9/` | Aggregated result tables of the main experiment grid (TA1 to TA9), duplicate audit, efficiency benchmark and SHA-256 manifests |
| `paper_results/v10/` | Aggregated result tables of the preregistered deployment grid (TD1 to TD4) |
| `paper_results/lineage/` | Pool sizes recorded by the run-time disjointness assertions |
| `preregistration/` | Protocol of the deployment grid, time-stamped before any code for that experiment was written |
| `paper_figures/` | Scripts that draw Figures 2 to 6 of the manuscript from the values in `figdata.py` |

No retinal images, cached image arrays, per-image model scores or trained model weights are included. The image datasets are licensed by their providers and must be obtained from the original sources listed below.

## Data

| Cohort | Source |
|---|---|
| EyePACS (Kaggle Diabetic Retinopathy Detection) | https://www.kaggle.com/competitions/diabetic-retinopathy-detection |
| EyeQ quality labels | https://github.com/HzFu/EyeQ |
| Messidor-2 | https://www.adcis.net/en/third-party/messidor2/ |
| DeepDRiD | https://github.com/deepdrdoc/DeepDRiD |
| DDR | https://github.com/nkicsl/DDR-dataset |
| APTOS 2019 | https://www.kaggle.com/competitions/aptos2019-blindness-detection |
| IDRiD | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |

## Setup

1. Install the packages in `requirements.txt` (Python 3.12, a CUDA-capable GPU for training and inference).
2. Choose a working directory for data and outputs and replace the placeholder `/path/to/sharc_work` in the scripts, for example

   ```bash
   sed -i 's#/path/to/sharc_work#/your/working/dir#g' scripts/*
   ```

   The scripts expect `data/`, `models/`, `preds/`, `results/` and `logs/` under that directory and are run from its `scripts/` folder.
3. Download the datasets into `data/raw/` from the sources above.

## Reproducing the results

| Step | Scripts |
|---|---|
| Preprocess and cache images | `prep_cache.py`, `patch_deepdrid.py`, `pack_uint8.py`, `extract_verify.py`, `prep_eyepacs_test.py` |
| Train models and run inference | `train_gpu.py` (via `run_train_gpu2.sh`), `dino_frozen.py`, `dino_degrade.py`, `degrade_infer.py`, `infer_test.py` |
| Main experiment grid (v9) | `run_v9.sh`, which runs `e_v9.py` (imports `e_v2.py` to `e_v8.py`) |
| Ablations and calibration scale | `run_final3.sh`, `after_final3.sh` |
| Preregistered deployment grid (v10) | `run_v10.sh`, which runs `e_v10.py` |
| Analysis tables | `v9_analysis_v8.py`, `v9_sample_flow.py`, `analysis_v10.py`, `v10_fix.py`, `export_slim.py` |
| Verification | `unit_tests.py` (23 tests), `p0_consistency.py` (18,000 code-versus-formula checks), `p0_verify_tables.py`, `v10_verify_and_score.py`, `v9_final_checks.py`, `cp_sim.py` |
| Duplicate audit and timing | `dup_audit.py`, `dup_verify.py`, `dup_hp.py`, `dup_labels.py`, `bench.py` |
| Figures | `paper_figures/make_figs.py` |

The experiment modules are versioned (`e_v2.py` to `e_v10.py`) because each version imports components of earlier ones. The reported results come from `e_v9.py` (main grid) and `e_v10.py` (preregistered grid). Runs use fixed seeds, and every run asserts that the source and target data pools are disjoint by patient.

## Preregistration

`preregistration/PREREG_DEPLOYMENT_GRID.md` has SHA-256

```
0518621e46f6a5e2a2936510de8431062fee4de0642e242325788ccca9a5786c
```

which matches the value reported in the manuscript.

## Licence

No licence has been chosen yet, so all rights are reserved by the authors. Please contact the corresponding author (rashmirathi.u@srmap.edu.in) about reuse.
