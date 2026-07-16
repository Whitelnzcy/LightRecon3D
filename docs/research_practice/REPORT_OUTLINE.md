# Research Practice Report Outline

Target: at least 8,000 Chinese characters and 30 references. First complete
draft due 2026-07-26.

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
