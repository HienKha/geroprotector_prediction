# Reproducing the analyses

Run all commands from the repository root. Use a unique lowercase run identifier each time; runners deliberately refuse to overwrite existing output directories.

## 1. Install and test

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[traditional,test]'
pytest -q
python scripts/freeze_package_manifest.py --check
```

For foundation-model experiments, install the optional `v6` dependencies and stage the checkpoint and license files required by the selected protocol.

## 2. Prepare Stage 0

```bash
export GERO_PYTHON="${GERO_PYTHON:-python3}"

bash scripts/prepare_stage0.sh \
  "/absolute/path/to/Geroprotectors-Project-INGER/5.Chemical space/Geroprotectors by ML.csv" \
  /absolute/path/to/Geroprotectors-Project-INGER/0.Data/Geroprotectors_Clean_Descriptors_2024.csv \
  /absolute/path/to/Geroprotectors-Project-INGER/0.Data/No_geroprotectors_and_Toxicos.csv
```

Do not continue if curation, split construction, or leakage auditing fails.

## 3. Source-study 405-row benchmark

Set the two labeled input paths explicitly:

```bash
export GERO_POSITIVE=/absolute/path/to/Geroprotectors-Project-INGER/0.Data/Geroprotectors_Clean_Descriptors_2024.csv
export GERO_NEGATIVE=/absolute/path/to/Geroprotectors-Project-INGER/0.Data/No_geroprotectors_and_Toxicos.csv
```

Run the conventional baselines:

```bash
bash scripts/run_traditional_paper405.sh \
  traditional405_reproduction \
  "$GERO_POSITIVE" "$GERO_NEGATIVE"
```

Run the fixed component and ensemble comparison:

```bash
bash scripts/run_fixed_blend_paper405.sh \
  fixedblend405_reproduction \
  "$GERO_POSITIVE" "$GERO_NEGATIVE"
```

These commands preserve the source study's 324/81 split. Threshold-dependent comparisons use the protocol-defined common cutoff; threshold-free ranking metrics are computed from continuous scores.

## 4. Robustness and insight analyses

The full insight suite is intentionally sequential because several stages share one GPU:

```bash
bash scripts/run_insight_suite.sh --stamp reproduction --resume
```

The suite verifies completed upstream artifacts before reuse. It then runs model-wide QED analyses, repeated rank stability, chemistry-aware cross-validation, AgeXtend endpoint-aligned retraining, DrugAge *C. elegans* endpoint reconstruction, and the integrated report. External datasets and all required checkpoints must already be staged.

Individual runners are available under `scripts/` when only one prespecified analysis is needed. Their corresponding YAML files are under `configs/`.

## 5. Verify outputs

```bash
python scripts/verify_completed_run.py outputs/RUN_ID
```

A valid completed run must contain its completion record and declared hashes. Do not compare or report a partially written directory as a completed experiment.

## Scope of exact reproduction

The source and configuration files here reproduce the implemented workflows. Exact numerical reproduction additionally requires the pinned datasets, provider checkpoints, third-party model code, and software versions recorded by each protocol. Historical AgeXtend and Kapsiani-Howlin reference-model refits remain limited by training-state or split details that were not released; the code reports these limitations rather than silently approximating them.
