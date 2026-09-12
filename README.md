# Geroprotector prediction benchmark

This repository contains the source code, locked protocol configurations, schemas, and tests for the study:

> *Structure-based geroprotector classifiers systematically under-rank higher-QED endpoint-positive compounds: a benchmarking and explainability analysis*

The study compares conventional machine learning, molecular similarity, neural tabular models, and tabular foundation models. It evaluates the released Geroprotectors.org/ChEMBL split, repeated random partitions, chemistry-aware partitions, retrospective DrugAge and AgeXtend stress tests, and endpoint-aligned retraining. No model is treated as a universal winner.

Raw datasets, model checkpoints, fitted models, predictions, figures, and numerical result artifacts are intentionally excluded. They are either governed by their original providers or generated locally by the workflows below.

## Quick start

The locked environment targets Python 3.12.

```bash
git clone https://github.com/HienKha/geroprotector_prediction.git
cd geroprotector_prediction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[traditional,test]'
pytest -q
```

Install the optional foundation-model stack only when those experiments are required:

```bash
python -m pip install -e '.[v6]'
```

Some experiments also require TabFM, whose upstream implementation and checkpoint are not redistributed here. See [DATA_AND_MODEL_ACCESS.md](DATA_AND_MODEL_ACCESS.md).

## Reproduce the data and split registry first

The D1 source files come from `BioAgeLab/Geroprotectors-Project-INGER` at commit `c8f458925f5ea2beeba87c7c2dda62eefacf618c`. Place the three source files outside this repository or under a git-ignored local data directory, then run:

```bash
export GERO_PYTHON="${GERO_PYTHON:-python3}"

bash scripts/prepare_stage0.sh \
  "/absolute/path/to/Geroprotectors-Project-INGER/5.Chemical space/Geroprotectors by ML.csv" \
  /absolute/path/to/Geroprotectors-Project-INGER/0.Data/Geroprotectors_Clean_Descriptors_2024.csv \
  /absolute/path/to/Geroprotectors-Project-INGER/0.Data/No_geroprotectors_and_Toxicos.csv
```

This stage performs curation, creates the shared split registry, and runs the leakage checks. It must finish before any model pipeline is run. The 1,488 previously predicted candidates remain unlabeled and never become positive training examples.

For the source study's released 405-row split and the paper-aligned runners, use the raw 206-positive and 199-reference files directly as documented in [REPRODUCING.md](REPRODUCING.md).

## Repository map

- `src/geroprotector/`: model, validation, curation, ablation, explainability, and reporting code.
- `configs/`: locked data, validation, model, and analysis protocols.
- `scripts/`: command-line runners and integrity checks.
- `schemas/`: machine-readable artifact contracts.
- `tests/`: contract, leakage, configuration, and integration tests.
- `LEAKAGE_CONTRACT.md`: scientific and implementation boundaries.
- `REPRODUCING.md`: run order, external inputs, and verification commands.

Every run writes to a new directory under `outputs/` and refuses to overwrite an existing run. Runtime products are excluded from Git so that source code and generated evidence remain separate.

## Interpretation boundaries

- The released D1 test split is descriptive because it was inspected during exploratory development.
- DrugAge and AgeXtend scoring by D1-fitted models is retrospective, endpoint-mismatched stress testing, not prospective external validation.
- Endpoint-aligned retraining evaluates transport of an architecture and protocol, not transport of D1-fitted parameters.
- Weak ChEMBL references are not experimentally confirmed inactive compounds.
- QED analyses describe score and retrieval behavior; they do not establish a biological mechanism or causal effect.

## Integrity

After changing source files, regenerate the source inventory:

```bash
python scripts/freeze_package_manifest.py
python scripts/freeze_package_manifest.py --check
```

Completed runs include manifests and SHA-256 records. Verify any completed run with:

```bash
python scripts/verify_completed_run.py outputs/RUN_ID
```

## Citation and license

Citation metadata are provided in [CITATION.cff](CITATION.cff). A software license has not yet been declared by the author team; until one is added, the source remains under default copyright protection.
