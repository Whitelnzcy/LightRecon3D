# Loss provenance and capacity control

Reference checked against primary paper sections3.1,3.3,3.4:
https://arxiv.org/html/2212.12402 (Push-the-Boundary,3DV2022).
Official repository https://github.com/shenglandu/PushBoundary.

## Actual implemented objective

L = weighted CE +0.5 present-class soft Dice +0.3 direction MSE.
CE weight is2 for originally correct pixels,1 otherwise. Mean denominator is
valid point count, NOT sum of weights. Dice macro-averages GT-present candidate
classes per patch, smoothing1. Missing candidate/invalid/originally unassigned
targets ignored in training; still counted in full-domain evaluation.
Direction MSE is mean over three coordinates and eligible points, selected
using TRAIN GT ownership. Direction targets are fixed global XYZ unit vectors;
current inward variant fills boundary points toward same-plane interior.

Paper reference: semantic CE, weighted binary boundary CE, direction MSE;
total Ls+3Lb+0.3Ld. Thus MSE and numeric0.3 have a literature analogue. Our
reductions/data/tasks differ; copying the scalar does NOT calibrate loss scales.
Our Dice0.5 and original-correct CE2 weighting are engineering choices, not
claimed to reproduce that paper or proven optimal by an ablation.

Important deviations: frozen RGB boundary evidence (no joint boundary BCE in
this run), plane-candidate ownership rather than semantic categories, registered
four-neighbor same-resolution propagation instead of hierarchical point-cloud
upsampling. Paper describes unit direction predictions; our auxiliary MSE is
on raw Tanh outputs, normalization occurs only in propagation. This allows
shrinkage; do not call our loss/architecture an exact reproduction. Boundary
inward completion and grid-target variant are our adaptations. Balance trial
has not beaten uniform target loss on boundary error and is not the default.

## Capacity test now launched

Original3200 seed42 none/learned already completed and audited. New width2
3200 seed42 none/learned shares data/loss/sampling/optimizer/grid/bias with it.
All hidden widths doubled (16->32,24->48,32->64), input/output dimensions fixed.
Do not change receptive field or loss simultaneously. Each width's two arms
share identical initialization and batches; cross-width weights cannot match.
29 direction tests passed before launch. Checkpoint schema4 records width;
default1 remains compatible with older checkpoints. Output:
local_outputs/direction_capacity_w2_20261025_s42.
Results pending: audit predictions, direction quality and train fit before
claiming capacity is the bottleneck. No promotion or efficacy claim yet.
