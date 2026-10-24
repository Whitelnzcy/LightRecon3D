# Direction sensitivity, reach and reproducibility

Base HEAD b540791. Same fixed direction-propagation research line. No XYZ edits,
backbone replacement, graph cuts or new candidate generation.

## Frozen sensitivity

Output `local_outputs/direction_sensitivity_20261024_v1`; checkpoint is the learned
arm of direction_control_20261024_v1. First16 GT-free tiles of every view;
overlapping sample occurrences, NOT independent points or full-domain metrics.
GT read after forwards only to partition boundary<=5px / interior.

| Boundary stratum | scene199 | scene190 |
|---|---:|---:|
| mean absolute propagation change / mean absolute original feature | 1.984% | 2.252% |
| label flips with zero direction | 0.0668% | 0.0483% |
| label flips with identity propagation | 0.1055% | 0.1620% |
| label flips removing output bias | 52.916% | 62.244% |

Large bias sensitivity is NOT improved accuracy. Local propagation changes
features little, and changing direction affects few decisions in this frozen
model. This motivates controlled reach and bias experiments, not a new method.

## Grid reach: 200steps,TRAIN4,DEV2

Only four-neighbor step1/4/8 changes, same radius.02 normalized globalXYZ,
loss, bias+2, initialization and minibatch sequence within each seed.

| Seed | Step | Total errors | Boundary errors | Damage |
|---|---:|---:|---:|---:|
| 42 | 1 | 270223 | 62788 | 3748 |
| 42 | 4 | 277434 | 63405 | 3527 |
| 42 | 8 | 289273 | 63238 | 4191 |
| 43 | 1 | 297341 | 64070 | 1224 |
| 43 | 4 | 284185 | 63632 | 1265 |
| 43 | 8 | 271786 | 63884 | 3101 |

No consistent boundary benefit for wider reach. Do not promote; retain step1.
Both runs independently audited (six scene/arm results each), sources/index
mappings verified. Scene190 seed42 figure inspected; artifacts persist. Seed43
global XYZ projection audit generated. No independent-test claim.

Earlier same-seed baseline runs also vary numerically; default GPU execution
was not deterministic. Do not attribute small cross-run differences to methods.
New runner supports strict deterministic algorithms, cuBLAS workspace :4096:8,
cuDNN deterministic, benchmark disabled and TF32 disabled.

## Deterministic replay

`local_outputs/direction_replay_20261024_v1`: seed42,30updates,two identical arms.
Every parameter equal after EVERY update; full predictions equal on both scenes.
Independent audit passed. At30steps both predictions remain original: this is a
reproducibility check, NOT efficacy. Does not establish cross-device reproducibility.

## Bias experiment in progress

`local_outputs/direction_bias_20261024_s42`: deterministic200steps, output bias
0/1/2, grid1 unchanged. Candidate original-support input retained. Duplicate bias2
asserts bitwise equality each step. Results pending at this progress checkpoint.

## Commands and files

Python: E:/anaconda/envs/ai/python.exe.
`diagnose_direction_sensitivity.py --output local_outputs/direction_sensitivity_20261024_v1`
`train_direction_control.py --output local_outputs/direction_reach_20261024_s42 --experiment reach --seed 42 --steps 200`
Repeat seed43 with s43 directory. Audit with `audit_direction_mechanism.py --run DIR`.
Replay: `--experiment replay --seed 42 --steps 30 --deterministic`.
Bias: `--experiment bias --seed 42 --steps 200 --deterministic`.
15 direction-related tests passed. Model defaults preserve old behavior and
state_dict keys; new checkpoint schema2 explicitly stores grid_step/support_bias.
Changed files: model, runner, sensitivity diagnostic, direction tests, task and
handoff documents. Unrelated worktree changes and all previous outputs preserved.
