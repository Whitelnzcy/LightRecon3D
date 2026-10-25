# Direction training-budget diagnostic

Base HEAD6b0bcc3. Fixed research line; grid1,bias2,radius.02,TRAIN4,DEV2,seed42,
batch4, same loss. Strict deterministic. None vs learned direction at200/800steps;
each short/long pair bitwise equal during first200steps. All frozen before DEV.
Learned200 also bitwise equals prior bias2 checkpoint across independent runs.

## Full-domain DEV results

| Arm | Total errors | Boundary errors | Damage |
|---|---:|---:|---:|
| none200 | 255502 | 62047 | 6936 |
| learned200 | 239788 | 61509 | 19152 |
| none800 | 229797 | 55926 | 12620 |
| learned800 | 237959 | 58456 | 19762 |

Training longer helps both arms, but at800steps direction propagation loses to
no propagation by8162 total errors,2530 boundary errors,7142 damaged points.
Do not attribute the overall training gain to direction propagation.
HistoricalDEV2 is repeatedly used development data, not independent testing.

## Full TRAIN-fit results

| Arm | Total errors | Boundary errors | Damage |
|---|---:|---:|---:|
| none200 | 721817 | 156261 | 15099 |
| learned200 | 690128 | 148359 | 67238 |
| none800 | 621495 | 141681 | 20424 |
| learned800 | 621444 | 142278 | 30032 |

At800steps training total errors nearly tie; learned direction damages9608 extra
correct points. Not simply a DEV-only generalization failure. Direction quality
on fixed first4 patches/view0 improves unevenly: learned800 median angles
26.77/47.03/73.56/75.08deg for TRAIN0/1/2/3. These patches are not a full-cloud
direction statistic, nor held-out evaluation.

## Verification and artifacts

`local_outputs/direction_budget_20261024_s42` contains checkpoints, predictions,
protocol hashes, history, full-domain metrics and fixed-view comparisons.
`training_fit/` stores full TRAIN predictions/metrics and patch direction quality.
Independent DEV audit recomputed all8 scene/arm results and verified mappings.
15 direction tests and syntax checks pass. Full-view red-error and green/magenta
change figures generated using saved predictions, no visual point displacement.
Original sources and previous results preserved. No mainline replacement.

## Oracle800 diagnostic completed

`local_outputs/direction_oracle800_20261024_s42`: matched800step oracle-direction
training/evaluation, seed42, deterministic, same architecture/loss/bias/grid.
GT direction broadcast to ALL candidates. Diagnostic only, not deployable or a
mathematical upper bound. Completed: total224727,boundary57896,damage14991.
Independent audit passed both scenes. Better total error than none800, but1970
MORE boundary errors and2371 more damaged points. Thus an accurate direction
input alone does not establish boundary benefit for the current operator.

Commands use E:/anaconda/envs/ai/python.exe:
`train_direction_control.py --output local_outputs/direction_budget_20261024_s42 --experiment budget --seed 42 --steps 800 --deterministic`
`evaluate_direction_training.py --run local_outputs/direction_budget_20261024_s42`
`audit_direction_mechanism.py --run local_outputs/direction_budget_20261024_s42`
`visualize_direction_error_changes.py --run local_outputs/direction_budget_20261024_s42 --baseline none800 --method learned800`
Oracle: same runner `--experiment control --control-arms oracle --steps 800 --seed 42 --deterministic` with separate output directory.
