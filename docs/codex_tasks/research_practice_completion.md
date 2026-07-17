# Research Practice Completion Task

Date: 2026-07-15

Deadline: 2026-07-31 23:00 Asia/Shanghai

Status: active and scope-frozen. W1 engineering output is complete. The W2
three-group identical-cache smoke and per-group gate are complete. Raw manual
identity aggregation failed its pre-registered cross-group gate and is frozen
as an ablation. Learning-support-guided RANSAC improved F1 in all three smoke
groups but failed the pre-registered quality and efficiency promotion paths;
the final eight-independent-scene GPU batch and paired audit have now passed
completely. Guided RANSAC won pairwise F1 on all `8/8` scenes, achieved mean
and median paired gains of `+0.039248` and `+0.035580`, and passed the frozen
quality path. It is the final method; global RANSAC is the primary baseline.
The all-eligible validation run selected 17 independent scenes. The first pass
completed 14, and the immutable recovery completed the remaining three. The
final combined ledger passes 17/17 scenes; guided RANSAC wins pairwise F1 in
16/17, with mean F1 `0.632406 -> 0.706348`. The efficiency path failed and no
universal acceleration claim is permitted. W3 is complete: the frozen server
benchmark measured Stage1 accuracy over 80 pair records, model footprint, GPU
latency/memory and archived per-stage runtime. W4 report, figures,
reproducibility index and optional paper-style draft are now the active work
package.

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
* mentor-facing progress report with embedded result figures;
* optional paper-style draft based on the same frozen experiments.

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
A4  A2 + learning-support-guided RANSAC proposal/consensus/refit
A5  A4 + retained provenance/conflict diagnostics and structural line output
NEG PlaneGraph-BA / live feedback on the archived scene only
```

### W4. Report and optional paper draft

Write the report in parallel with implementation. Numbers enter the report
only through archived JSON/CSV artifacts. Embed the archived result figures in
the mentor-facing and final reports. After the practice report is complete,
reuse the same methods, numbers and limitations in an optional paper-style
draft without selecting new scenes post hoc.

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
Jul 29  optional paper-style draft
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

Freeze report Tables 2, 3 and 5 from the recovered 17-scene bundle and W3
JSON/CSV artifacts. Use the 17/17 combined ledger as the final main result and
retain the original 14/17 run only as recovery provenance. Complete the figure
index, failure-gallery selection, reference formatting and Word rendering. Do
not rerun the final batch or W3 benchmark unless an artifact-integrity check
fails.
