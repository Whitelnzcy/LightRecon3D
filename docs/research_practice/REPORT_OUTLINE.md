# Research Practice Report Outline

Target: at least 8,000 Chinese characters and 30 references. First complete
draft due 2026-07-26.

Current state on 2026-07-17: `REPORT_DRAFT.md` has been revised around the
completed 17-scene recovered batch. Its body before the
evidence index contains 9,519 Chinese characters under a strict CJK count, and
`REFERENCES.md` contains 30 primary-source entries. The remaining work is final
figure selection, GB/T 7714 metadata cleanup and
Word/template rendering.

## 1. Introduction and problem definition (about 1,000 characters)

* indoor weak-texture and clutter failure modes;
* why unknown camera poses make reconstruction harder;
* why editable bounded structural primitives are useful;
* final task and honest scope.

## 2. Related work (about 1,500 characters)

* pose-free pointmap foundation models;
* indoor plane instance segmentation;
* line detection and structural-line reconstruction;
* multi-view bounded plane reconstruction and association;
* robust geometric optimization and its limitations;
* lightweight structural heads.

## 3. Final system architecture (about 1,000 characters)

* frozen DUSt3R alignment;
* Stage1/Stage2 plane support path;
* structural-line path;
* exact provenance registry;
* topology-logistic global identity recovery;
* optional learned intra-fragment split;
* bounded editable outputs.

## 4. Method (about 2,500 characters)

* image preprocessing and coordinate transforms;
* plane instance support prediction;
* local geometry validation and SVD refit;
* 2D line detection and exact 3D lifting;
* topology/geometric identity features and shared logistic model;
* average-linkage remerge and deterministic plane estimation;
* learned intra-fragment split as the only B/M variable;
* exact provenance, prediction freeze barrier and non-mutating evaluation.

## 5. Data and evaluation protocol (about 1,000 characters)

* Structured3D scenes and retained split;
* identical-cache fairness rule;
* point-aligned identity and metric GT construction;
* accuracy, support, identity and efficiency metrics;
* reproducibility and checksum controls.

## 6. Experiments (about 1,800 characters)

* Stage1 accuracy, speed and model size;
* development-only direct SVD, global RANSAC and guided RANSAC comparison;
* final 128-scene prospective B16/M16 comparison;
* structural-line diagnostics;
* ablations;
* qualitative results;
* archived negative feedback experiment.

Final E16 result for the main B/M table:

```text
128 independent Structured3D scenes
124 five-view scenes and 4 four-view scenes
256/256 predictions frozen before final GT parse
128/128 paired support hashes identical
B16 / M16 Plane F1: 0.000000 / 0.000000
B16 / M16 matched IoU: 0.000000 / 0.000000
B16 / M16 partition pair F1: 0.358892 / 0.358968
mean delta pair F1: +0.0000758
95% paired bootstrap CI: [-0.0000552, +0.0002065]
B16 / M16 predicted planes: 878 / 1292
prediction full-cache label rate: about 0.111
GT plane label rate: about 0.972
mechanical verdict: Mixed_Result
final default: B16
research extension: M16
```

The zero Plane F1 is a support-completeness result. It must not be described
as proof that the models are wholly useless. The sparse prediction domain
cannot reach the frozen full-support IoU threshold. Use partition pair F1 for
the relative B/M comparison and state that its confidence interval crosses
zero.

Current large-scale result for Table 3:

```text
17 eligible independent Structured3D validation scenes
17 valid identical-cache paired comparisons after immutable recovery
global RANSAC pairwise F1 mean: 0.632406
learning-guided RANSAC pairwise F1 mean: 0.706348
paired mean / median gain: +0.073942 / +0.064718
scene wins: 16/17
exact two-sided sign test: p=0.000275
matched IoU mean: 0.489136 -> 0.678036
overmerge excess mean: 2.882353 -> 1.470588
VOI mean (nats, lower): 1.493965 -> 1.172062
Rand Index mean: 0.789837 -> 0.834620
Segmentation Covering symmetric mean: 0.561563 -> 0.658550
method-stage runtime mean: 19.826422 s -> 17.892183 s
quality gate: passed; no universal acceleration claim
```

Write this as an improvement under the frozen project protocol, not as a
published-dataset state-of-the-art claim. Report that the first run passed
14/17 and the immutable recovery completed the remaining three, producing a
17/17 final ledger. State that `scene_00194` decreased by about 0.002. The
earlier eight-scene audit is the frozen confirmation experiment. Raw manual identity remains a negative
ablation (`3/8` wins; median F1 delta `-0.106110`). Conflict-drop scores must
always be printed together with their collapsed coverage.

Frozen W3 result for Tables 2 and 5:

```text
Stage1 sampled support records: 80 pairs from 8 independent scenes
Stage1 pairwise precision / recall / F1 mean: 0.883814 / 0.629410 / 0.724166
Stage1 purity / completeness / combined F1 mean: 0.928410 / 0.726850 / 0.809942
Stage1 head: 79,906,572 parameters, 304.898 MiB checkpoint
Stage1 head per-image latency: P50 75.160 ms, P95 101.401 ms
Stage1 head incremental peak allocation: 361.899 MiB
Stage2 MLP: 203,521 parameters, 0.783 MiB checkpoint
Stage2 64-candidate latency: P50 0.451 ms, P95 0.483 ms
five-view DUSt3R inference + 300-step alignment: P50 9.123 s, P95 11.922 s
structural-line stage: mean 1.725 s over 8 scenes
```

The Stage1 head is 14.0% of the shared DUSt3R parameter count and must not be
called tiny. The alignment views were resized from `1280 x 720` to `512 x 288`
under the image-size-512 setting. The guided method passed the quality path,
not the efficiency path; its runtime is not a promoted contribution.

## 7. Failure analysis and limitations (about 800 characters)

* Stage1 direct-support coverage is the primary bottleneck;
* sparse prediction support versus complete GT support;
* learned split changes only already assigned points;
* M creates additional plane fragments without stable identity improvement;
* depth discontinuities for image lines;
* single-backbone limitation;
* plane residual versus true geometry conflict.

## 8. Conclusion (about 400 characters)

Summarize the completed system and measured trade-offs without unsupported
novelty or superiority claims.

## Required figures and tables

```text
Figure 1  final system diagram
Figure 2  exact coordinate/provenance mapping
Figure 3  plane and line front-end outputs
Figure 4  actual B16 input, image-space identities, and global 3D result
Figure 5  global RANSAC/guided RANSAC/fusion comparison
Figure 6  actual frozen B16/M16 reconstruction comparison
Figure 7  E16 B/M paired-effect distribution
Figure 8  similar-result and extra-split B/M galleries
Figure 9  complete GT support versus sparse prediction support
Table 1   related-method scope comparison
Table 2   Stage1 accuracy/speed/model-size result
Table 3   multi-scene plane identity/support result
Table 4   ablations
Table 5   end-to-end runtime and memory
```

## Claim ledger

Every result sentence must cite an archived JSON/CSV path. Published-method
numbers must retain their original dataset, protocol and source; numbers from
incompatible protocols must not be placed in one superiority ranking.

The 17-scene guided RANSAC result is development evidence. The prospective
128-scene E16 result controls the final B16/M16 method positioning. Do not
blend the two protocols into one superiority claim.
