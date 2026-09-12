# Data and model access

No raw dataset, fitted checkpoint, prediction table, or manuscript result is distributed in this repository.

## D1 benchmark

The source-study repository is:

- Repository: `https://github.com/BioAgeLab/Geroprotectors-Project-INGER`
- Pinned commit: `c8f458925f5ea2beeba87c7c2dda62eefacf618c`
- Reported positives: `0.Data/Geroprotectors_Clean_Descriptors_2024.csv`
- ChEMBL reference compounds: `0.Data/No_geroprotectors_and_Toxicos.csv`
- Previously predicted candidates, retained as unlabeled only: `5.Chemical space/Geroprotectors by ML.csv`

The expected hashes, row counts, encodings, and column names are in `configs/data.yaml`. Acquisition fails if a pinned input does not match its declared hash.

## DrugAge

DrugAge Build 5 must be obtained from `https://genomics.senescence.info/drugs/`. The mapping and endpoint protocols are recorded in:

- `configs/drugage_celegans_benchmark_protocol.yaml`
- `configs/screening_blend_external_protocol.yaml`
- `configs/kapsiani_historical_benchmark_protocol.yaml`

Place local files under `external_data/drugage_build5/` or set `DRUGAGE_SOURCE_DIR`.

## AgeXtend

AgeXtend source materials are described by DOI `10.1038/s43587-024-00763-4` and its public repository. Required file identities and paths are recorded in:

- `configs/agextend_endpoint_benchmark_protocol.yaml`
- `configs/agextend_official_reference_protocol.yaml`

Place local files under `external_data/agextend_2024/` or set `AGEXTEND_SOURCE_DIR`.

## Foundation-model checkpoints

Checkpoints are not redistributed. Obtain them under the upstream providers' terms and verify the hashes declared in the relevant protocol file.

By default, runners look for:

```text
checkpoints/tabpfn-v2-classifier.ckpt
checkpoints/tabpfn-v3-classifier-v3_default.ckpt
```

For TabPFN-v2, `TABPFN_V2_CHECKPOINT` may point to another local path. Network telemetry is disabled by the foundation-model runners.

## DataWarrior and OpenChemLib

The exact Source-SVM descriptor reproduction requires the DataWarrior/OpenChemLib resources identified by `configs/screening_blend_external_protocol.yaml`. Put local tools under the git-ignored `tools/` directory and retain their upstream licenses.
