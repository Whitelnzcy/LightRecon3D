# Boundary-zero target compatibility diagnostic

Base HEADb8422c9. This is an investigation within the SAME learned directional
feature propagation route, not a new reconstruction method. No mainline promotion.

## Two-end agreement oracle test

Multiply original weights by (1+dot(unit_d_i,unit_d_j))/2; zero directions neutral.
Same800steps,seed42,grid1,bias2,loss,strict deterministic,TRAIN4/historicalDEV2.
Both variants use GT directions at training and evaluation: diagnostic only.

| Oracle arm | Total errors | Boundary errors | Damage |
|---|---:|---:|---:|
| Original | 224727 | 57896 | 14991 |
| Agreement | 230721 | 56994 | 17785 |

902 fewer boundary errors but5994 more total errors and2794 more damaged points.
Still worse boundary error than none800 (55926). Independent audit passed all4
scene/arm metrics. Scene199 comparison inspected, persistent ragged junctions.
Output: local_outputs/direction_agreement_oracle_20261024_s42.

## Explicit edge-weight audit

All valid right/down grid edges plus reverse edges, no self edges. Full GT labels
on predicted XYZ; original k4 boundary rule and frozen RGB boundary probabilities.
Reported weight shares are RAW, unnormalized and exclude self; not accuracy.

| Statistic | scene199 | scene190 |
|---|---:|---:|
| Directed cross-plane edges | 7178 | 13962 |
| Both endpoint directions nonzero, original targets | 664 | 1620 |
| Cross-edge original oracle mean weight | 0.79095 | 0.77310 |
| Same-plane original oracle mean weight | 0.92581 | 0.92902 |

90.75%/88.40% of cross-plane edges have at least one zero endpoint direction.
Original boundary supervision explicitly sets boundary directions to zero. This
creates a compatibility problem for same-resolution local edge propagation,
particularly two-end agreement. This is a measured limitation, not proof it is
the only reason for weak final accuracy, nor a criticism of the original paper's
different hierarchical upsampling architecture.

## Target completion diagnostic

Change ONLY GT boundary targets: point toward nearest nonboundary point of the
same GT instance in predicted XYZ. Preserve ALL nonboundary targets. Missing
same-instance interior remains zero; covered by unit tests. No extra parameters.

| Inward-completed oracle mean weight | scene199 | scene190 |
|---|---:|---:|
| Cross-plane | 0.42415 | 0.46578 |
| Same-plane | 0.92695 | 0.93088 |
| Cross-plane with two-end agreement | 0.17358 | 0.18001 |

All measured cross edges now have two nonzero targets. This changes the intended
operator behavior while preserving same-plane mean weights, but is NOT evidence
of better learned predictions. Oracle inward800 completed in
local_outputs/direction_inward_oracle_20261024_s42: total232596,boundary56312,
damage10421. Independently audited on2026-10-25. Versus boundary-zero oracle,
1584 fewer boundary errors and4570 fewer damages but7869 more total errors.
Still386 more boundary errors than none800. No deployable superiority claim.
Next paired none/learned with inward TRAIN supervision and no GT prediction is
running in local_outputs/direction_inward_learned_20261025_s42.

Sources: direction_edge_weights_20261024_v1 and
direction_edge_weights_inward_20261024_v1 under local_outputs. 19 direction tests
pass (toward own interior, preserve nonboundary, missing interior fallback,
agreement extremes/constant field, existing mapping-independent tests). All old
model defaults preserved; checkpoint schema3 records agreement/grid/bias.
