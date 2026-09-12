# Leakage and logic contract for V5/V6

This contract states what the prepared code prevents and what it cannot solve.

## Data boundary

- Source role and observed label are assigned from the pinned input role before chemical
  standardization.
- Every raw row remains in provenance/standardization ledgers with an inclusion or
  exclusion reason.
- Connectivity duplicates with consistent labels collapse deterministically; an entire
  connectivity group is excluded when labels conflict.
- The 1,488 prior-model candidates retain `label = null` and contribute zero supervised
  targets.
- HAGR/DrugAge and external-path tokens are rejected by internal config/path guards. HAGR
  labels are never an allowed development input.

## Split boundary

- The primary grouping graph is built label-free from standardized structures using Morgan
  radius 2, 2,048 bits, no chirality and edge threshold `0.40`.
- A connected component is indivisible. Exact identity and component overlap across each
  outer fit/test partition must be zero.
- Five deterministic outer repeats and their grouped inner registries are built once and
  reused by references, V5 and V6.
- Model predictions are not inputs to split construction.
- The current implementation does not yet build the declared secondary registries.

## Nested selection, calibration and threshold boundary

- The model factory receives outer-training labels and locked inner folds, never outer-test
  outcomes for fitting or selection.
- Each cross-fit calibration prediction is produced by fitting the complete candidate
  pipeline on the corresponding calibration-training complement.
- Platt calibration, beta-calibration sensitivity and the MCC threshold are fit only from
  cross-fitted predictions for the active outer-training partition.
- The final base pipeline refits on all outer-training compounds and writes exactly one
  final outer prediction record per compound. V6 additionally performs prespecified,
  label-free repeated/batch/order inference calls on those query features solely to audit
  inference semantics. Aggregate outer metrics are deferred until the run is sealed.

Repeated predictions are not independent samples. Reporting first averages the five OOF
probabilities for each compound and bootstraps paired differences by primary component.

## V5 boundary

- The global fingerprint bank is molecule-local: it uses structure and pinned featurizer
  settings only, with no labels or cohort-fit statistics.
- Stage A resolution comparison is confined to the active fit partition and its grouped
  selection folds.
- For every Stage-C selection fold, Stage B receives only that fold's training rows. Stage B
  then creates a further grouped sub-CV wholly inside those rows: importance estimators fit
  on sub-train and permutation importance uses only the corresponding sub-validation. The
  enclosing selection-validation labels cannot affect the weights used to predict them.
- SVD, Nyström landmarks, descriptor preprocessing, weighting and downstream estimators are
  fit only on the active training IDs and support ordinary out-of-sample transform.
- All attempted Stage-C candidates, including failures, are retained in the selection
  ledger. Outer outcomes and HAGR do not choose a branch.

Stage A and Stage C are hierarchical selection steps on the same active inner-CV evidence;
their scores are selection statistics, not an unbiased performance estimate. The untouched
outer fold is the performance unit.

## V6 boundary

- Descriptor cleaning, median imputation, constant-column removal and Morgan SVD are fit
  inside the active training partition.
- Foundation checkpoints and license files must be regular project-local files with exact
  hashes; enabled package versions must match exactly.
- Telemetry and implicit Hugging Face/transformer network access are disabled by the run
  wrappers.
- Context and query identities must be disjoint. The canonical policy is one query per
  inference call.
- Repeated-call, query-order, context-row-order, matched-feature-permutation, whole-batch and
  random-composition behavior is audited without using query labels. Required canonical
  checks fail loudly.
- V6 panel/checkpoint/ensemble selection uses grouped inner OOF only. No HAGR or outer-test
  metric selects the inference policy.

## Artifact boundary

- Resolved config, environment, data/split references, model/transform/calibrator artifacts,
  selection attempts and predictions are hash-bound.
- Per-job outputs are immutable and reusable only under the same run binding.
- The full source tree is snapshotted. Editing code, configs, schemas, scripts, tests or
  docs during training prevents continuation; reporting also requires the matching source
  hash.
- Reporting verifies the completed bundle before computing metrics and writes a separate
  immutable report directory.
- The report-time source classifier is not newly fitted: because source and label are
  bijective, it reuses sealed R2 OOF predictions and labels the descriptor, nearest-neighbor
  and chemistry-matching analyses exploratory.

## What this does not fix

The observed positive and weak-reference classes come from different sources, so
source-label confounding is perfect. Weak references are not confirmed inactive compounds.
Chemical grouping reduces analogue/identity optimism, but cannot identify whether a model
learned a biological signal instead of source-specific chemistry. Internal results must not
be described as causal geroprotection, human efficacy, clinical utility, untouched external
validation or state of the art.

HAGR is a reused historical stress test, not a fresh external set. A transport claim needs
an endpoint-aligned cohort whose outcomes remained inaccessible until code, model,
calibration, applicability rule and analysis plan were frozen.
