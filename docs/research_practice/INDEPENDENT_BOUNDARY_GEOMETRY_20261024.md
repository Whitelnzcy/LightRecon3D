# Independent RGB boundary versus raw plane geometry

Base commit: 01862717e43f63d77f255b74784e891d743bff66.
Branch: codex/line-support-refinement-20261023. User authorized four-input pilot.
Task: ../codex_tasks/independent_boundary_geometry_20261024.md.
Output: local_outputs/independent_evidence_train4_dev2_20261024_v1/.

## What changed and why

Audit found old M3 used RGB-refined intersections (`after`) and RGB matching
scores; it was never a pure plane-intersection control. New inputs recompute
raw intersections from fixed predicted plane equations and recovered projection,
and independently infer a frozen RGB-only BoundaryNet on all30 original images.
No RGB search is used for G; no line availability limits B's application.
No change to XYZ, candidate pool, source rows, loss or keep/reassign mechanism.

N/B/G/BG share active46107-parameter architecture, initial weights, same200 steps,
batch4, seed42,64px samples, observed-pixel application, weightedCE+.5Dice.
Six baseline channels are repeated into the extra branch; evidence adds to those
channels (five geometry, one boundary). This preserves active capacity but mixes
semantic channels: B is added to the repeated residual channel. Thus a weak B
result does NOT establish that boundary features are intrinsically ineffective;
separate feature encoders/concatenation remain an untested alternative.
Raw pair features still infer side orientation from original predicted ownership.
All pair intersections are candidates, not verified visible structural segments.

Boundary checkpoint SHA256:
70d3b95ec8635f320876ca6d67f3665ed3c232bc1eebed99e30d719b0ddfb370.
TRAIN00000--00003 are in upstream train_scenes.00199/00190 are in upstream
test_scenes, BUT repeatedly inspected downstream, so named historical_DEV here,
not independent tests. No fresh generalization or significance claim.

## Results: two historical development scenes, counts lower is better

| Input | Total errors | Boundary errors | Originally correct points damaged |
|---|---:|---:|---:|
| Original assignment |308558|63996|0|
| N: no extra prior |254364|62609|11205|
| B: RGB boundary |257236|62344|9208|
| G: raw intersections |243255|60413|10525|
| BG: both |241555|60051|11244|

B vs N:265 fewer boundary errors (0.42%),1997 fewer damages, but2872 more total
errors and4869 fewer corrected points. BG vs N:2558 fewer boundary errors(4.09%),
12809 fewer total errors,39 more damages. BG vs G:362 fewer boundary errors,
1700 fewer total errors but719 more damages. Not a strong improvement.

Per scene boundary errors N/B/G/BG:
00199:17647/17508/15572/16241;
00190:44962/44836/44841/43810.
Combined input does not dominate geometry on every scene.

Previous RGB-refined M3 had51831 boundary errors and17418 damages. New BG has
60051 and11244: fewer damages but worse boundaries. Old M3 also had different
application masks and feature encoding, so cross-run comparison does not isolate
the benefit of prior-guided line correction. Do NOT replace mainline.

TRAIN total/boundary/damage:
N695955/152835/23967; B709473/153996/23587;
G666313/148622/27599; BG665327/148750/29455.

## Checks and artifacts

Training process exited0; training loop43.154seconds, allocated295241216bytes,
reserved534773760bytes (includes resident frozen boundary model, excludes prep
time and non-PyTorch allocations). All four gates received gradients.
32 related unit tests pass; five modified/new implementation scripts compile.
Audit verifies305 raw intersections against source plane equations, source/RGB/
checkpoint/cache hashes, unchanged point rows/unassigned IDs, fixed candidate IDs.
DEV full metrics exactly recomputed against frozen evaluation targets; outputs
saved before loading DEV targets. No recomputed TRAIN GT metric audit claimed.
Projection diagnostic P95:0.1230px/0.1085px, not a depth accuracy measurement.
Viewed independent input panels, both DEV output panels and projection diagnostic.
Six scene comparison figures plus independent input figure are in figures/.

Commands (E:/anaconda/envs/ai/python.exe):
```
train_spatial_line_ablation.py --output local_outputs/independent_evidence_train4_dev2_20261024_v1 --steps 200 --independent-evidence
audit_independent_evidence.py --run local_outputs/independent_evidence_train4_dev2_20261024_v1
visualize_independent_evidence.py --run local_outputs/independent_evidence_train4_dev2_20261024_v1
visualize_spatial_projection_audit.py --run local_outputs/independent_evidence_train4_dev2_20261024_v1 --output local_outputs/independent_evidence_train4_dev2_20261024_v1/projection_audit
```
Tests: unittest discovery test_independent_evidence.py, test_spatial*.py,
test_line_support_patches.py. Frozen checkpoint and source receipts stay local.
Existing untracked train_rgb_structure_boundary.py is a hashed legacy dependency;
not silently staged with unrelated historical files. See protocol code hashes.
Gate means in report are sampled at raw-geometry available pixels by the reused
diagnostic, NOT the actual application mask (observed pixels for all four arms).

## Next hypothesis, not implemented

Use RGB boundary as independent evidence of where a finite structural boundary
exists; use plane geometry to condition candidate pair identity rather than to
declare every infinite intersection a real boundary. First separate boundary
and geometry channels cleanly, and compare the same N/B/G/BG protocol with
explicit candidate-pair side/ownership supervision. Avoid simultaneous loss,
architecture, scale changes. Candidate completeness, independent scenes,
multiseed and matched current Plane-DUSt3R comparison remain unresolved.
