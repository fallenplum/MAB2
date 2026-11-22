# M60 max-uncertainty baseline correction

## Protocol

The former scorer routed `max_uncertainty` to the warm replay implementation. The scorer now has an explicit policy. It ranks legal unresolved row and cell audits using the current audit mass and cell weight magnitude, with deterministic tie-breaking. It does not use sealed labels for policy selection.

The replay covers the existing three-model, three-task interface-drift stress panel. It has 1,152 question-model units per density and is a same-input control, not independent confirmation.

## Results

| changed relay rows | warm exact set | explicit max-uncertainty exact set | warm tokens | explicit tokens |
|---:|---:|---:|---:|---:|
| 1 | 0.961806 | 0.961806 | 22.8828 | 22.8828 |
| 2 | 0.909722 | 0.909722 | 23.3030 | 23.5564 |
| 4 | 0.817708 | 0.817708 | 23.3056 | 23.5720 |

Full CTDR with the signature guard remains exact-set 1.0 at 170.2092, 307.5851 and 587.8151 tokens for densities 1, 2 and 4. The explicit maximum-uncertainty baseline therefore does not remove the observed same-input separation from Full CTDR.

## Interpretation

The correction closes a reproducibility and attribution defect. It does not establish an independent ACL-level result. The new baseline is almost identical to warm replay on this finite audit, which indicates that the previous positive separation was not caused by accidentally comparing CTDR against a different policy. It also shows that this heuristic has little additional value over warm replay under the present interface.

The evidence remains limited to exact recovery of a finite audit decision set and logical token accounting. Independent question cohorts, task-level answer quality, wall-clock measurements, and broader conditional-channel drift are still required before an ACL Main claim.

