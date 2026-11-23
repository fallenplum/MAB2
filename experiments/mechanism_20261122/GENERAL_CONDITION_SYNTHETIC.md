# General conditional-channel extension

This route-B attempt uses a synthetic binary service action rather than a real language-model task. It tests whether the certificate logic survives outside the finite MCQA representation. The generated TV radii are oracle diagnostics, so the result is a mechanism sanity check and not an ACL service-quality result.

| drift type | method | action error | mean tokens | mean queries |
| --- | --- | ---: | ---: | ---: |
| source_only | stale | 0.0000 | 0.00 | 0.00 |
| source_only | warm_reweighted | 0.0000 | 0.00 | 0.00 |
| source_only | ctdr_transport | 0.0000 | 0.00 | 0.00 |
| source_only | fixed_reverse_recheck | 0.0000 | 0.00 | 0.00 |
| source_only | full_restart | 0.0000 | 42.13 | 4.00 |
| source_only | oracle_lower_bound | 0.0000 | 0.00 | 0.00 |
| conditional | stale | 0.4300 | 0.00 | 0.00 |
| conditional | warm_reweighted | 0.4300 | 0.00 | 0.00 |
| conditional | ctdr_transport | 0.0000 | 9.51 | 0.90 |
| conditional | fixed_reverse_recheck | 0.0000 | 9.51 | 0.90 |
| conditional | full_restart | 0.0000 | 41.81 | 4.00 |
| conditional | oracle_lower_bound | 0.0000 | 0.00 | 0.00 |
| interface | stale | 0.4860 | 0.00 | 0.00 |
| interface | warm_reweighted | 0.4860 | 0.00 | 0.00 |
| interface | ctdr_transport | 0.0000 | 9.36 | 0.91 |
| interface | fixed_reverse_recheck | 0.0000 | 9.36 | 0.91 |
| interface | full_restart | 0.0000 | 41.76 | 4.00 |
| interface | oracle_lower_bound | 0.0000 | 0.00 | 0.00 |
| mixed | stale | 0.4040 | 0.00 | 0.00 |
| mixed | warm_reweighted | 0.4040 | 0.00 | 0.00 |
| mixed | ctdr_transport | 0.0000 | 11.54 | 1.14 |
| mixed | fixed_reverse_recheck | 0.0000 | 14.20 | 1.37 |
| mixed | full_restart | 0.0000 | 42.22 | 4.00 |
| mixed | oracle_lower_bound | 0.0000 | 0.00 | 0.00 |

## Interpretation

Source-only edits have zero conditional radius and CTDR performs zero new queries, formally matching warm reuse. Conditional and interface drift cells require revalidation when the old action interval crosses the threshold. Full restart is an upper-cost exact reference. Random recheck is included as a non-adaptive control.

The experiment does not establish a new-task or new-checkpoint result. A real BoolQ, GSM8K, or other non-MCQA worker run was not claimed because the current server execution path was unavailable during this cycle. The route-B claim therefore remains unvalidated and the paper's empirical scope stays on graph-conditioned certificate transport.

