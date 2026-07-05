# Stage3 DUSt3R Global Alignment Task

Read:

- `AGENTS.md`
- `docs/codex_handoff/CURRENT_STATE.md`

Current task: redesign Stage2-to-Stage3 data flow so cross-view plane merge/refit operates in DUSt3R globally aligned coordinates.

Execute one phase at a time.

## Phase 1 only

Perform a read-only audit:

1. Locate the Stage2 NPZ writer and list the exact current NPZ fields.
2. Locate the Stage3 reader and current local merge/refit code.
3. Trace how RGB paths, view IDs, scene names, and pair groups currently propagate.
4. Trace mask resolution and pixel-coordinate conventions.
5. Locate the vendored/installed DUSt3R global-alignment implementation.
6. Confirm actual signatures for:
   - image loading;
   - `make_pairs`;
   - `inference`;
   - `global_aligner`;
   - `compute_global_alignment`;
   - `scene.get_pts3d()`;
   - confidence retrieval.
7. Identify missing metadata and compatibility risks.
8. Do not modify code yet.
9. Do not start training.
10. Do not tune plane merge thresholds.

Write the audit to:

`docs/codex_handoff/STAGE3_GLOBAL_ALIGNMENT_AUDIT.md`

At the end report:

- files inspected;
- current data-flow diagram;
- current NPZ schema;
- missing fields;
- coordinate-system risks;
- view-index risks;
- proposed file changes;
- focused test plan;
- git status and diff summary.