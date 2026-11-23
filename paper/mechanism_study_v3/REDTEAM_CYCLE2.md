# Cycle-2 ACL Main red-team review

## Verdict

**CONDITIONAL GO for a narrow graph-manifest certificate-transport paper.**
The new evidence materially closes the strongest-baseline attribution gap. The
release is still blocked until the ACL PDF is rebuilt from the current source.

## Evidence that now passes

### Fair M60 matrix

The public matrix covers 3,456 model--task--density--worker units. It keeps the
same old transcript, post-edit source, token prices, full-restart budget and
exact-set stopping target. The full CTDR mask is sourced from the graph manifest
descendant mapping, not row-level sealed metadata. The matrix reports stale,
exact warm cache, reweighted old history, CTDR transport only, CTDR plus
selective ranking, full CTDR, maximum uncertainty, periodic, random, full
restart and a restricted changed-row oracle.

The aggregate numbers are internally consistent with the manuscript. Full CTDR
has exact set 1.0000, zero action error and 230.91 tokens. Full restart also has
exact set 1.0000 at 878.78 tokens. Reweighted old history and transport only
both have exact set 0.9167 at 22.73 tokens. Selective ranking has the same exact
set at 24.48 tokens. The restricted oracle has exact set 1.0000 at 21.20
tokens.

This supports graph-manifest changed-row revalidation as an action-error
recovery mechanism. It does not support cost superiority over reweighted audit
or the restricted oracle. The manuscript correctly narrows the claim and keeps
transport and ranking as non-independent components in this matrix.

### Theory and clean replay

`THEORY_CLOSURE.md` supplies the sufficient safe-transport theorem, a tight TV
counterexample, the necessity boundary, no-drift degeneration to stale reuse,
and the separation between safety stopping and cost ordering. The clean replay
fixture exposes all nine requested controllers and passes its two tests without
private questions or weights.

### Generalization routes

Route A has short and semantic interface-drift stress panels. Route B is a
synthetic binary conditional-channel sanity check. The artifacts explicitly
exclude a real non-MCQA or new-checkpoint claim, which is scientifically
correct. This remains a scope limitation rather than a hidden positive result.

## Remaining blockers

1. `main.tex` was modified after the currently stored `main.pdf`. The current
   PDF therefore does not contain the M60 matrix or theory additions. A fresh
   successful PDF build and visual check are required before release.
2. The matrix uses graph-manifest descendant rows as a privileged method input.
   The paper must keep this input permission explicit and separate it from
   sealed row-level metadata. The public protocol now states this correctly.
3. The M60 units reuse the same question cohort across task, model and density
   factors. They are repeated factors, not independent datasets. The source
   already states this limitation.
4. The restricted oracle is not a global cell-level optimum. The current
   artifact labels it as a restricted offline oracle, which must remain.
5. Route B uses oracle synthetic radii. It is a falsifiable extension artifact
   and cannot widen the empirical language-model claim.

## Acceptance decision

After a successful source-matched PDF rebuild, the evidence supports an ACL
Main-level mechanism submission with the narrow claim that graph-manifest and
interface-aware changed-row revalidation can recover exact finite audit
decisions under graph edits at a substantial cost below full restart. A claim
of cost-optimal scheduling, universal service quality, broad non-MCQA
generalization or independent benefit from transport radius/ranking remains
unsupported.

