# Learned3D direction pilot: conditioning not established

Base HEAD5352564756c4777a62aa2969a293c736efc3ef42. User asked continue.
Task docs/codex_tasks/direction_pilot_20261024.md. No source XYZ or IDs changed.

## Implementation and protocol

PushBoundary-inspired SAME-RESOLUTION feature propagation, grid4neighbors+center,
weights use normalized global3D offsets and frozen RGB boundary. NOT full original
hierarchical kNN/PointNet++/KPConv reproduction. Existing explicit view/pixel
mapping preserved. No intersection-based rejection. Boundary source receipt,
cache SHA and exact view transform checked; GT not part of model forward.

D0 distance-only weights; DS direction with candidate-mean context; DC direction
with per-candidate mask/residual context. Same parameters/init and TRAIN4 batches,
seed42,200steps,batch4,64px. All auxiliary-supervise global nearest GT boundary
direction: zero boundary targets, k4 on full valid predictedXYZ. Same0.3MSE added
to existing CE+.5Dice. Correct-candidate gather is TRAIN supervision only.
Alternative candidate directions lack direct vector supervision; segmentation
still supplies gradients. Both proposals and inference use no GT routing.
Fixed spatial radius0.02scene diagonal. Frozen RGB prior, no boundary retraining.

v1 implementation omitted explicitXYZ in direction head (though used in weights).
This limits global-frame direction prediction. Preserve diagnostic run, then v2
adds normalizedXYZ3 identically to every head. No loss/budget tuning. v1weights
remain loadable via include_xyz=False, v2True. No rotation-equivariance claim.
The width change changes random initialization of later layers despite same seed;
do NOT attribute v1-v2 aggregate differences causally to coordinates alone.

## Results

v2 historicalDEV,1361282 valid points:

|arm|total errors|boundary errors|damaged correct points|
|---|---:|---:|---:|
|Original|308558|63996|0|
|D0|299427|63845|2026|
|DS|271025|62900|2714|
|DC|273264|62889|3988|

DC vs DS:11 fewer boundary errors(0.0175%),2239 MORE total errors,1274 MORE
damages. No evidence of useful candidate-conditioning gain at this budget.
Per scene boundary DS/DC:19917949/17580;19044951/45309 (opposite directions).
Do NOT replace historical M3 (51831 boundary errors,17418damage); architectures,
auxiliaryloss differ, so cross-run comparisons not controlled mechanism tests.

TRAIN v2 total/boundary/damage:
D0805811/163734/2141; DS758082/159134/5773; DC762974/158174/5931.
v1 diagnosticDEV total/boundary/damage:
D0245765/60913/9305; DS266771/63394/4793; DC242134/60668/11328.
Both versions retained; do not select v1 just because some counts are better.

Direction quality, frozen v2, TRAINscene0 first16 GT-free patches per view(80
patches,304326 valid occurrences, overlaps counted), nonzero targets only angles:
D0 median69.65deg/norm0.1285; DS71.97deg/0.1264; DC64.99deg/0.1925.
MSE D00.30785,DS0.30872,DC0.28671,all-zero predictor0.31649. Weak direction fit
even in TRAIN diagnostic. Small norms can make normalized directions unreliable;
not proof of sole failure cause. Do not interpret ownership changes as evidence
that correct directions were learned.200steps/one seed insufficient to reject
the general approach. No independent test or novelty claim.

## Verification and outputs

Outputs local_outputs/direction_train4_dev2_20261024_v1 and_v2.
Both train/eval processes exit0. v2loop21.352s,allocated729331200bytes,
reserved1136656384bytes, excludes prep/eval/CUDA overhead. All direction heads
have nonzero step3 gradients.42 related tests pass, syntax checks pass.
Both runs' source/prior hashes, row indices, candidate IDs, unassigned invariance
and fullDEV metrics verified. TRAIN fullmetrics not independently recomputed.
Six comparison figures per run; viewed both v2DEVfigures. New projection audit
generated for v2, P95 remains0.1230/0.1085px. No depth-accuracy implication.
Generic report gate means are dummyzero for these models, NOT confidence scores.

Commands via E:/anaconda/envs/ai/python.exe:
```
train_spatial_line_ablation.py --output local_outputs/direction_train4_dev2_20261024_v2 --steps 200 --direction-pilot
audit_direction_pilot.py --run local_outputs/direction_train4_dev2_20261024_v2
audit_direction_quality.py --run local_outputs/direction_train4_dev2_20261024_v2
visualize_spatial_projection_audit.py --run local_outputs/direction_train4_dev2_20261024_v2 --output local_outputs/direction_train4_dev2_20261024_v2/projection_audit
```
v1 same commands without direction_quality, with old19-channel head. Tests:
test_spatial*.py, test_boundary_direction_propagation.py,test_pushboundary_targets.py,
test_independent_evidence.py,test_line_support_patches.py.

Next proposed diagnostic: single-scene/small-patch fit to separate optimization
failure from uninformative vector targets; oracle direction intervention as an
explicit GT diagnostic only, plus zero/random direction controls. Do not add
more mechanisms or scale training until direction fit and causal utility are
demonstrated. Missing candidate pool remains separate known limitation.
