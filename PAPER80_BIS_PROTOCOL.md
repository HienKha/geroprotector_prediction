# V5bis/V6bis paper-holdout protocol

This isolated package adds only two pipeline identities:

- `V5BIS_PAPER80`, based on `V5_ELIXIRFP_REBUILT`;
- `V6BIS_PAPER80`, based on `V6_TABULAR_FOUNDATION_MODELS`.

The original V5, V6 and V7–V16 configurations and algorithms are retained. Bis
configs extend their base configs and change only the evaluation design and artifact
namespace.

The split identity lock is
`configs/paper80_v3_1_test_identities.txt`, containing the 77 test connectivity
identities from the completed V3-1 identity-safe reconstruction. The remaining 305
curated identities are outer-train. The file is SHA-256 pinned in `paper80.yaml`.

The three-fold selection/calibration component registry is fit from the 305
outer-training structures only. Test structures are excluded from that construction,
so they cannot transitively merge two training groups. A separate post-split audit on
all structures finds 18 global similarity components crossing train/test; this is
reported as expected analogue proximity in the random split and is never used for
architecture, calibration or threshold selection. Exact identity overlap is zero.

The source publication used 206 reported positives and 199 weak references (405
rows), a shuffled one-time 80/20 split with seed 42, seven descriptors and a linear
SVC. Bis pipelines intentionally do not recreate duplicate raw rows: identity
resolution and conflict removal precede split generation, preserving the Stage-0
scientific safety contract.

Primary scientific evidence remains the original grouped V5/V6 design. Bis results
must be reported separately and may be useful for an apples-to-apples same-test
contextual model comparison and compute-efficient exploratory iteration.
