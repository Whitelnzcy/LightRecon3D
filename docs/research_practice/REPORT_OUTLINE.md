# Research Practice Report Outline

Target: at least 8,000 Chinese characters and 30 references. First complete
draft due 2026-07-26.

Current state on 2026-07-17: `REPORT_DRAFT.md` has been revised around the
17-scene selection and 14 valid large-scale comparisons. Its body before the
evidence index contains 9,446 Chinese characters under a strict CJK count, and
`REFERENCES.md` contains 30 primary-source entries. The remaining work is recovery-result
insertion if available, final figure selection, GB/T 7714 metadata cleanup and
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
* global fusion and editable outputs.

## 4. Method (about 2,500 characters)

* image preprocessing and coordinate transforms;
* plane instance support prediction;
* local geometry validation and SVD refit;
* 2D line detection and exact 3D lifting;
* learning-support-guided plane hypotheses, global consensus and weighted refit;
* bounded connected components, exact provenance and conflict handling;
* why alignment feedback is excluded from the final method.

## 5. Data and evaluation protocol (about 1,000 characters)

* Structured3D scenes and retained split;
* identical-cache fairness rule;
* point-aligned identity and metric GT construction;
* accuracy, support, identity and efficiency metrics;
* reproducibility and checksum controls.

## 6. Experiments (about 1,800 characters)

* Stage1 accuracy, speed and model size;
* direct SVD, global RANSAC, guided RANSAC and support-fusion comparison;
* structural-line diagnostics;
* ablations;
* qualitative results;
* archived negative feedback experiment.

Current large-scale result for Table 3:

```text
17 eligible independent Structured3D validation scenes
14 valid identical-cache paired comparisons, 3 infrastructure failures
global RANSAC pairwise F1 mean: 0.644675
learning-guided RANSAC pairwise F1 mean: 0.721296
paired mean / median gain: +0.076621 / +0.069861
scene wins: 14/14
exact two-sided sign test: p=0.000122
matched IoU mean: 0.500448 -> 0.688247
overmerge excess mean: 2.928571 -> 1.500000
method-stage runtime mean: 19.945028 s -> 18.356680 s
quality gate: passed; no universal acceleration claim
```

Write this as an improvement under the frozen project protocol, not as a
published-dataset state-of-the-art claim. Always report the 14/17 completion
rate and the three infrastructure failures. The earlier eight-scene audit is
the frozen confirmation experiment. Raw manual identity remains a negative
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

* fragmentation and over-merge;
* sparse support and coverage;
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
Figure 4  global bounded editable result
Figure 5  global RANSAC/guided RANSAC/fusion comparison
Figure 6  failure gallery
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
