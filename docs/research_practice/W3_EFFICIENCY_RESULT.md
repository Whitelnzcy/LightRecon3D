# W3 efficiency result record

Date: 2026-07-16

Status: completed on the server. No rerun is required.

## Controlling artifacts

The canonical machine-readable result remains on the experiment server:

```text
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/efficiency_results.json
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/stage1_accuracy_records.csv
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/stage1_accuracy_per_scene.csv
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/archived_stage_timings.csv
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/gpu_latency.csv
/gemini/data-1/lightrecon_runs/research_practice_efficiency_20260716_v2/efficiency_report.md
```

The result was produced by Git SHA
`b5d80c28b2cd4eb459688e669185b3c88ba5218d`. This commit contains the cache
registry fix required by the benchmark. The later commit `123e9d5` changes
launcher diagnostics only and does not change the benchmark or final method.

The benchmark ran with PyTorch `2.7.1+cu126`, NumPy `2.2.6`, SciPy `1.15.3`,
OpenCV `4.13.0` and roma `1.5.6` on one `L1.gpu.xlarge` device with 24555 MiB
reported GPU memory. DUSt3R uses an image-size setting of 512. The five
Structured3D views preserve their original aspect ratio and were loaded as
`512 x 288`, so the report must not imply that those alignment images were
square.

## Stage1 support result

The evaluation used 80 pair records from eight independent Structured3D
scenes. These are sampled support-partition metrics, not full-image semantic
segmentation AP.

| Metric | Mean | Median |
|---|---:|---:|
| assignment rate | 0.998100 | 1.000000 |
| GT-labeled coverage | 0.998100 | 1.000000 |
| pairwise precision | 0.883814 | 0.902075 |
| pairwise recall | 0.629410 | 0.599966 |
| pairwise F1 | 0.724166 | 0.726905 |
| predicted purity | 0.928410 | 0.946061 |
| GT completeness | 0.726850 | 0.731812 |
| purity/completeness F1 | 0.809942 | 0.834746 |

The precision/recall gap shows that Stage1 is conservative: its retained
support is usually pure, but it misses part of each GT plane. This is
consistent with using the learned output as guidance for geometric consensus
rather than accepting it as the final plane partition.

## Model footprint

| Component | Parameters | Parameter MiB | Checkpoint MiB |
|---|---:|---:|---:|
| shared DUSt3R backbone | 571,171,208 | 2178.845 | 2179.151 |
| Stage1 plane-support head | 79,906,572 | 304.819 | 304.898 |
| Stage2 region-merge MLP | 203,521 | 0.776 | 0.783 |

The Stage1 head is about 14.0% of the DUSt3R parameter count. It should be
described as an added task head, not as a tiny head. Stage2 is genuinely small.
The project's lightweight claim rests on reusing one shared pointmap backbone,
keeping the learned downstream components bounded and using classical line
extraction and geometric consensus instead of training another reconstruction
foundation model.

## GPU latency and memory

| Component | Samples | P50 (ms) | P95 (ms) | Peak allocated (MiB) | Incremental peak (MiB) |
|---|---:|---:|---:|---:|---:|
| DUSt3R pair backbone, two images | 10 | 148.428 | 152.940 | 3084.664 | 575.407 |
| Stage1 support head, one image | 50 | 75.160 | 101.401 | 2959.156 | 361.899 |
| Stage2 MLP, 64 candidate pairs | 50 | 0.451 | 0.483 | 2598.386 | 1.125 |
| five-view inference and global alignment | 3 | 9123.057 | 11921.618 | 2745.809 | 546.402 |

The five-view figure includes 20 ordered image-pair inferences and 300 global
alignment iterations. The local RoPE2D implementation was not CUDA compiled,
so these numbers describe the actual project environment and should not be
presented as a hardware-independent speed limit.

## Archived final-batch stages

| Stage | Scenes | Mean (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|
| global RANSAC | 8 | 20.946 | 15.905 | 39.815 |
| learning-guided RANSAC | 8 | 19.172 | 16.631 | 32.101 |
| structural lines | 8 | 1.725 | 1.716 | 1.893 |
| support-record metrics | 8 | 2.534 | 2.483 | 2.962 |
| full-cache metrics | 8 | 3.904 | 3.895 | 4.522 |
| direct support | 8 | 52.160 | 56.119 | 81.668 |
| uncached direct-support alignment/export proxy | 6 | 64.161 | 57.742 | 82.082 |

The guided method passed the frozen quality path, not the efficiency path.
Its median stage time is slightly higher than RANSAC even though its mean is
lower. Scene structure changes the number and size of plane hypotheses, so no
universal acceleration claim is permitted.
