# ACL Main readiness review for Pairwise Gap Certificate

## Current decision

Continue the research cycle. The new candidate closes the independent
algorithmic-cost gate and one real service gate, but the package is not yet
marked Main-ready.

## Evidence gates

| gate | status | evidence |
|---|---|---|
| Conditional soundness | pass under stated TV-radius and complete-mask assumptions | `PAIRWISE_GAP_THEORY.md` |
| Algorithmic replay | pass | 27 held-out model--task--density panels, action exact 1.0 |
| Fair answer-action baseline | pass | `PAIRWISE_GAP_CERTIFICATE_AGGREGATE.md` |
| Pooled quantile selection | pass | pooled (q=.9), 27/27 action exact |
| Split-conformal calibration | pass under exchangeability contract | fixed α=.1, 27/27 action exact, 24.9% weighted reduction |
| Synthetic drift stress | pass under supplied radii | 9 cells, 5,000 samples per cell |
| Hidden interface handling | partial pass | Qwen-0.5B transfer plus fail-closed unit tests |
| Qwen-0.5B service | pass for answer-action target | (q=.9) preserves action and reduces forwards and wall time |
| Qwen-14B service | diagnostic only | custom worker/service replay mismatch remains |
| Independent question cohort | partial | held-out residue is question-disjoint within repeated panels, not a new benchmark |
| Independent ARC validation cohort | pass for worker action-cost transfer | 128 cached ARC-Challenge questions, 24 held-out questions per density |
| Independent ARC service trace | pass for measured action-cost diagnostic | Qwen-0.5B single-forward relay-cost reduction at all densities |
| Full edge certificate | intentionally out of scope for low-cost route | edge exactness is reported as a diagnostic |

The paper must keep the service claim model-specific and must not present the
Pairwise Gap scheduler as a stand-alone novelty. Gap-directed refinement is
covered by prior work. The novelty boundary is graph-edit descendant analysis,
interface-fingerprint invalidation and fail-closed transport of paid
communication evidence.

