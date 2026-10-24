# Direction mechanism diagnostic — 2026-10-24

Base commit: `70c96394106bb347fb2f79c43341cc840c3f883b`.
Protocol: `docs/codex_tasks/direction_diagnostic_20261024.md`.
Outputs: `local_outputs/direction_mechanism_diagnostic_20261024_v1`.

## Question and result

Is the weak direction pilot primarily unable to learn its targets, or insensitive
to direction at inference? Four-patch memorization succeeds. Frozen decoder
interventions show extremely small ownership changes, even with GT directions.
This does not prove that direction-based methods cannot help. It argues against
scaling direction training without first testing the propagation mechanism.

## Four-patch memorization

TRAIN scene 0, view 0, fixed patches 0/1/64/65, 200 updates, AdamW lr 0.001,
direction MSE only. Each arm starts from a copy of its original v2 checkpoint.
These overfit checkpoints are NOT used on DEV.

| Arm | Initial median angle | Final median angle | Final MSE |
|---|---:|---:|---:|
| DS shared direction | 76.752° | 3.923° | 0.016825 |
| DC candidate-conditioned direction | 59.585° | 3.332° | 0.016459 |

This is memorization, not held-out generalization or ownership improvement.

## Frozen original models, two historical DEV scenes

Identical inputs, candidates, XYZ, camera and model weights. Oracle direction is
broadcast identically to every candidate slot, without GT candidate routing.
Random directions are fixed per view and shared across overlapping tiles.
1,361,282 evaluated points; original boundary errors: 63,996.

| Arm / direction | Total errors | Boundary errors | Newly damaged points |
|---|---:|---:|---:|
| DS learned | 271025 | 62900 | 2714 |
| DS GT oracle | 270890 | 62884 | 2705 |
| DS zero | 270985 | 62916 | 2712 |
| DS random | 270967 | 62910 | 2708 |
| DC learned | 273264 | 62889 | 3988 |
| DC GT oracle | 273136 | 62863 | 3969 |
| DC zero | 273231 | 62887 | 3963 |
| DC random | 273223 | 62887 | 3966 |

Oracle saves only 16 / 26 boundary errors versus learned directions. This is a
GT-assisted diagnostic, NOT deployable performance or a mathematical upper bound:
replacement can shift the direction distribution, and the decoder was not jointly
trained with the replacement. Candidate alternatives also receive the same oracle.
Both scenes have been repeatedly used for development, not independent testing.

## Verification

- Diagnostic runner completed, exit 0; two focused intervention tests passed.
- Learned predictions exactly reproduce saved v2 baselines before GT access.
- Independent audit recomputed all 16 full-domain results and verified source
  hashes, view/pixel/source ordering, original IDs, candidate and unassigned rules.
- Four comparison figures generated; scene 199 DC comparison visually inspected.
- Global XYZ projection diagnostic generated under `projection_audit/`.
- Existing checkpoints, source XYZ and labels were not modified.

Commands (Python: `E:/anaconda/envs/ai/python.exe`):
`diagnose_direction_mechanism.py`, `audit_direction_mechanism.py`,
`visualize_spatial_projection_audit.py`, and the focused test module
`tests/test_direction_intervention.py`. Consult each script CLI for output arguments.

## Next controlled experiment (not yet executed)

Test whether the current four-neighbor feature mixing adds useful information at
all: compare no propagation, spatial-only propagation, and jointly trained
GT-direction propagation on a small fixed diagnostic split. Keep the candidate
pool, initialization and budget fixed. Inspect propagation feature changes and
gradient magnitudes alongside boundary errors. GT is diagnostic only.

Hypotheses, not findings: neighboring features may already be similar; the
original-support bias and skip path may dominate the direction-modulated path.
Only if a controlled oracle-trained path helps should we invest in replacing its
oracle input with predicted evidence. Do not promote this pilot to the mainline.
