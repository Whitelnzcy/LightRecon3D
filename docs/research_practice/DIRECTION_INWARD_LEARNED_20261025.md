# Learned inward boundary supervision: paired control

Base training HEADb8422c9, implementation checkpoint e0df056. Same fixed direction
propagation route. TRAIN4/historicalDEV2,800steps,seed42,batch4,grid1,bias2,
radius.02,agreementOFF,strict deterministic. Only TRAIN direction target changed:
boundary points point toward nearest same-instance interior; interior unchanged.
No GT used in prediction. Both none/learned share this supervision, separating
auxiliary target effects from propagation. Saved zero-target controls use same
data, budget and deterministic settings. Single-seed exploratory result only.

## Full-domain results

| Target / propagation | Total errors | Boundary errors | Damage |
|---|---:|---:|---:|
| Zero / none | 229797 | 55926 | 12620 |
| Zero / learned | 237959 | 58456 | 19762 |
| Inward / none | 227704 | 55735 | 8768 |
| Inward / learned | 227458 | 56066 | 8940 |

Changing targets improves learned propagation by10501 total errors,2390 boundary
errors (4.09%) and10822 fewer damaged points compared with zero/learned. But
inward/learned versus inward/none:246 fewer total errors,331 MORE boundary errors,
172 MORE damaged points. Thus target completion has promising development evidence,
but direction propagation's independent boundary benefit is NOT established.

Per-scene inward none/learned boundary errors: scene19915054/15730, scene190
40681/40336. Not uniformly positive. Do not claim stable improvement or promote.

## Verification

19 direction tests pass; syntax checks pass. Independent audit recomputes all4
scene/arm results, validates source hashes, source/pixel/view ordering, candidate
IDs, unassigned preservation, and exact match to arrays saved BEFORE GT access.
Four-way comparison recomputed from both runs' pre-GT prediction archives.
Global XYZ projection diagnostic generated. Scene199 comparison and scene190
error-change figure visually inspected: substantial residual errors remain.
All prior files/checkpoints/caches and unrelated worktree edits preserved.

Artifacts: local_outputs/direction_inward_learned_20261025_s42, including
completion.json, audit.json, target_comparison/, error_changes_none_vs_learned/,
projection_audit/. All predictions are fixed original point IDs, no XYZ movement.

## Commands

Python E:/anaconda/envs/ai/python.exe:
`train_direction_control.py --output local_outputs/direction_inward_learned_20261025_s42 --experiment control --control-arms none learned --boundary-target inward --seed 42 --steps 800 --deterministic`
`audit_direction_mechanism.py --run local_outputs/direction_inward_learned_20261025_s42`
`compare_direction_target_runs.py --run local_outputs/direction_inward_learned_20261025_s42`
`visualize_direction_error_changes.py --run local_outputs/direction_inward_learned_20261025_s42 --baseline none --method learned`
`visualize_spatial_projection_audit.py --run local_outputs/direction_inward_learned_20261025_s42 --output local_outputs/direction_inward_learned_20261025_s42/projection_audit`

Next (not executed): measure boundary-only versus interior direction quality, then
repeat the paired target controls with another seed. Keep research line unchanged;
do not increase model complexity or claim oracle performance as learned results.
