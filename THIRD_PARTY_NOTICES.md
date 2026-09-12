# Third-party software and data

The BSD 3-Clause License in this repository applies only to original project source code and documentation. It does not relicense third-party software, datasets, model weights, or checkpoints.

No third-party model implementation, model checkpoint, raw dataset, or generated result artifact is included in this repository. The workflows may use separately installed or locally staged resources, including:

- RDKit, scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, TabNet, TabM, and TabPFN packages;
- the BiSHop reference implementation, which is expected under `third_party/bishop/` only for the corresponding optional run and is checked against protocol hashes;
- TabFM source code and pretrained weights, which must be obtained separately under the upstream terms;
- TabPFN checkpoints, including license-gated model versions, which must be obtained from the upstream provider;
- DataWarrior and OpenChemLib resources used for the Source-SVM descriptor reproduction; and
- Geroprotectors.org/ChEMBL, DrugAge, AgeXtend, and Kapsiani-Howlin source data.

Users are responsible for obtaining these resources from their original providers and complying with the applicable licenses, access terms, citation requirements, and data-use conditions. Paths under `third_party/`, `external_data/`, `external_sources/`, `checkpoints/`, and `tools/` are excluded from the repository and from the project license.
