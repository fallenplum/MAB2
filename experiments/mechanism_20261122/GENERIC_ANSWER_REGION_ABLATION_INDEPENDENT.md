# Independent graph-conditioned ablation

The ten corrected independent cells keep the same cached transcript and
answer-region stopping rule. All four relay rows are exposed in calibration and
test. Only the uncertainty-radius source and row-priority rule change.

| policy | decision exact | answer-action exact | relay tokens | queries |
|---|---:|---:|---:|---:|
| graph-conditioned radii + source-aware priority | 0.9968 | 1.0000 | 162.29 | 1.184 |
| pooled radius + source-aware priority | 0.9968 | 1.0000 | 356.87 | 2.706 |
| pooled radius + uniform priority | 0.9968 | 1.0000 | 356.90 | 2.706 |

The graph-conditioned policy saves 54.5% relay tokens against the pooled-radius
control. The pooled control removes row-specific drift information, while the
uniform-priority control additionally removes token-aware ranking. The all-row
calibration protocol assumes interface invariance for rows with no observed
conditional drift. The hidden-interface panel supplies the corresponding
fingerprint gate and conservative fallback.

