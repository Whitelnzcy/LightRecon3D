# Research Practice Completion Task

Date: 2026-07-15

Deadline: 2026-07-31 23:00 Asia/Shanghai

Status: active and scope-frozen. W1 engineering output is complete. W2 input
preflight passed 3/3 retained groups and the identical-cache smoke produced
all three groups. Its aggregate result is mixed; the per-group gate audit is
implemented locally and awaiting a CPU-only server run.

## Objective

Complete the approved undergraduate research-practice project as an auditable
engineering and experimental result. A top-tier paper or strong novelty claim
is not a completion requirement.

The final project title remains:

```text
Lightweight multi-feature indoor reconstruction without pose priors
```

The final implementation is narrower and evidence-based:

```text
unordered indoor images
-> frozen DUSt3R pointmaps and global alignment
-> lightweight plane-instance support prediction
-> local plane validation/refit
-> lightweight structural-line extraction and exact 3D lifting
-> exact (alignment_view_index, x, y) provenance
-> frozen-geometry bounded plane fusion
-> editable point/line/plane outputs
```

PlaneGraph-BA and live plane feedback are disabled in the final path. They are
retained only as a negative ablation showing that lower plane incidence loss
does not necessarily improve registered structural geometry.

## Definition of done

Code and reproducibility:

* one command runs the retained final pipeline from a manifest;
* every run records Git SHA, input paths, checksums, coordinate conventions,
  configuration, hardware and runtime;
* outputs include NPZ, PLY/GLB where applicable, JSON manifest and diagnostic
  images;
* scripts refuse to overwrite an existing output directory;
* `roma`, `trimesh` and OpenCV are explicitly checked by server entrypoints.

Experiments:

* at least eight valid Structured3D scenes, with a target of ten;
* Stage1 accuracy plus latency, trainable parameters, checkpoint size and peak
  memory;
* direct global SVD, sequential RANSAC, manual/support fusion and the retained
  provenance-aware method on identical caches;
* per-scene and aggregate identity/support/geometry metrics;
* point/line/plane qualitative outputs;
* ablations for local refit, association, conflict handling and structural
  lines;
* the failed feedback result is reported honestly and is not rerun at scale.

Final artifacts:

* Chinese practice report of at least 8,000 Chinese characters;
* at least 30 human-readable references;
* method diagram, result tables, ablation table and failure gallery;
* reproducibility README and final experiment index;
* defense slides and a precomputed demonstration.

Completion does not depend on beating every published method. If a method does
not win, the report must identify the controlled comparison, failure mode and
engineering consequence.

## Scope locks

Do not start any of the following before the deadline:

* training a new foundation model or line detector;
* resuming plane-feedback P2-P5;
* adding a second pointmap backbone;
* reproducing several external training pipelines;
* Gaussian-splatting, semantic or user-interface expansions;
* scene-specific threshold tuning;
* paper-submission work that does not serve the practice report.

## Work packages

### W1. Structural lines

Use OpenCV LSD on the saved DUSt3R-aligned RGB pointmap images. Preserve line
endpoints in explicit `(x, y)` pointmap coordinates, sample the identical
pointmap and confidence arrays, split depth discontinuities and robustly fit a
3D segment. Optionally associate both sides of a line with plane labels by an
exact provenance join.

Required outputs are line NPZ, edge PLY, per-view overlays, manifest, runtime
and diagnostics. No learned line model is required.

### W2. Batch identical-cache experiments

Generalize the retained single-scene scripts to a manifest-driven batch
runner. It must validate paths and checksums before execution, preserve failure
rows and emit JSON, CSV and Markdown summaries. A three-scene smoke precedes
the final 8-12-scene run.

### W3. Efficiency and ablations

Measure parameter counts, checkpoint bytes, per-image latency, P50/P95 latency,
peak GPU memory, alignment time and downstream method time. Freeze one common
input resolution and hardware record.

Required ablations:

```text
A0  DUSt3R + direct global SVD
A1  A0 + Stage1 support
A2  A1 + Stage2 local refit
A3  A2 + current manual merge
A4  A2 + retained provenance/conflict handling
A5  A4 + structural line output
NEG PlaneGraph-BA / live feedback on the archived scene only
```

### W4. Report and defense

Write the report in parallel with implementation. Numbers enter the report
only through archived JSON/CSV artifacts. Produce a static demo that does not
depend on live GPU execution during the defense.

## Calendar and gates

```text
Jul 15  scope, acceptance matrix, report outline
Jul 16  2D structural-line extraction
Jul 17  exact 3D line lifting and PLY/overlay diagnostics
Jul 18  batch runner and summary schema
Jul 19  three-scene server smoke
Jul 20  fixes and final-code freeze
Jul 21  8-12-scene final run
Jul 22  accuracy/identity/support tables
Jul 23  speed, model-size and memory benchmark
Jul 24  final ablations
Jul 25  figures and failure gallery
Jul 26  complete 8,000-character report draft
Jul 27  30-reference and claim audit
Jul 28  reproducibility package and report render
Jul 29  defense slides and demo
Jul 30  full consistency and fresh-directory verification
Jul 31  feedback, formatting and submission buffer
```

The structural-line implementation must not delay the July 19 smoke. If 3D
line association is weak, retain the line output and report association rate;
do not train a replacement model.

The provenance-aware identity method is promoted to a report improvement only
if its cross-scene result survives retained conflicts and coverage reporting.
Otherwise use the strongest deterministic baseline and report the negative
result.

## Server interaction contract

The local implementation is committed and pushed before every server request.
The user should need no more than three server interactions:

1. three-scene smoke;
2. final 8-12-scene batch;
3. efficiency and final ablations.

Each interaction is a single copy-paste command and produces a compact summary
bundle. Large caches and reconstructions remain on the server and are never
committed.

## Immediate next step

Run the CPU-only per-group smoke gate against the existing batch JSON. Use its
explicit stop/go result to choose raw manual aggregation or RANSAC as the
retained deterministic method, then freeze the final 8-12 independent-scene
manifest. Conflict-drop results are coverage-collapse ablations, and
structural lines remain output-only A5.
