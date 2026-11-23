# Research cycle baseline and theory update

## Decision

The previous ACL Main gate is reopened for this cycle. The new evidence
supports a narrower graph-manifest certificate-transport claim. It does not
support describing transport radius or selective ranking as independent cost
innovations.

## M60 strong baseline matrix

The matrix contains 3,456 model--task--density--worker units. Every controller
uses the same old transcript, post-edit source, token prices, full-restart
post-edit budget and exact-set stopping target. Full CTDR uses only the
graph-manifest descendant rows as its affected-row input. The restricted oracle
refreshes changed relay rows offline.

| controller | exact set | action error | tokens |
| --- | ---: | ---: | ---: |
| stale | 0.9039 | 0.0428 | 0.00 |
| exact warm cache | 0.8877 | 0.0443 | 0.00 |
| reweighted old history | 0.9167 | 0.0391 | 22.73 |
| CTDR transport only | 0.9167 | 0.0391 | 22.73 |
| CTDR + selective ranking | 0.9167 | 0.0391 | 24.48 |
| Full CTDR | 1.0000 | 0.0000 | 230.91 |
| max uncertainty | 0.9167 | 0.0391 | 22.86 |
| periodic | 0.9167 | 0.0391 | 39.81 |
| random | 0.9167 | 0.0391 | 37.34 |
| full restart | 1.0000 | 0.0000 | 878.78 |
| restricted oracle row refresh | 1.0000 | 0.0000 | 21.20 |

Full CTDR removes the action error of warm, reweighted, uncertainty, periodic
and random controllers, while saving 73.7% of full-restart tokens. Its cost is
still higher than reweighted replay and the restricted oracle. The attributable
mechanism is graph-manifest changed-row revalidation. Transport-only and
selective-ranking ablations do not add independent benefit in this matrix.

## Theory closure

The companion theorem artifact supplies a safe-transport sufficient condition,
a tight two-point TV counterexample, a minimax necessity boundary under TV-only
information, a no-drift reduction to zero-query stale reuse and a proof that
selective ranking changes cost ordering without changing safety. The current
implementation's source radius is conservative. The tighter pairwise source
term is documented separately.

## Generalization attempt

Route A now includes short-protocol and semantic-interface drift panels. Route B
was attempted with a synthetic binary conditional-channel stress test over 500
instances per drift class. It confirms the certificate logic outside MCQA, but
it is not a real non-MCQA language-model evaluation. A real BoolQ/GSM8K task and
new checkpoint remain unvalidated and are excluded from the empirical claim.

## Reproduction

The clean CPU replay fixture exposes the complete nine-controller pipeline,
including the restricted oracle lower bound, and tests interval updates,
stopping rules and token accounting without private questions or weights.

## Release note

The source changes in this cycle were rebuilt with pdfTeX on the server into a six-page PDF. The source/PDF pair was rendered locally and checked visually before release.

