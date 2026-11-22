# Interface-drift robustness panel

The GPU worker executed 54 model jobs covering two relay-interface variants, three graph-edit densities, three tasks and three fixed pretrained models. The two variants were a short protocol suffix and a semantic reviewer suffix. Each density contains 1,152 question-model units.

The tuple in the table is exact-set accuracy, action error, and mean incremental tokens.

| variant | density | stale | warm / explicit max-uncertainty | Full CTDR guard | token-aware guard | bounded shortcut | cold token |
|---|---:|---|---|---|---|---|---|
| short | 1 | 0.9592 / 0.0104 | 0.9861 / 0.0061 / 23.44 | 1.0000 / 0 / 145.27 | 1.0000 / 0 / 146.92 | 0.9297 / 0.0278 / 30.14 | 1.0000 / 0 / 940.80 |
| short | 2 | 0.9436 / 0.0156 | 0.9618 / 0.0113 / 23.47 | 1.0000 / 0 / 262.38 | 1.0000 / 0 / 263.69 | 0.9141 / 0.0347 / 55.33 | 1.0000 / 0 / 942.05 |
| short | 4 | 0.8976 / 0.0417 | 0.9097 / 0.0391 / 23.68 | 1.0000 / 0 / 502.10 | 1.0000 / 0 / 502.19 | 0.8767 / 0.0547 / 126.30 | 1.0000 / 0 / 941.25 |
| semantic | 1 | 0.9271 / 0.0226 | 0.9479 / 0.0165 / 22.83 | 1.0000 / 0 / 177.86 | 1.0000 / 0 / 178.64 | 0.9184 / 0.0347 / 34.27 | 1.0000 / 0 / 958.17 |
| semantic | 2 | 0.8507 / 0.0547 | 0.8585 / 0.0503 / 22.94 | 1.0000 / 0 / 325.69 | 1.0000 / 0 / 325.44 | 0.8793 / 0.0556 / 67.69 | 1.0000 / 0 / 966.53 |
| semantic | 4 | 0.7743 / 0.1068 | 0.7734 / 0.1068 / 23.28 | 1.0000 / 0 / 628.22 | 1.0000 / 0 / 628.14 | 0.8168 / 0.1016 / 157.52 | 1.0000 / 0 / 1019.84 |

The semantic suffix is a stronger same-input stress condition. Stale and warm replay degrade with edit density, while Full CTDR remains exact in the finite audit. This supports robustness of the certificate mechanism under two additional interface changes. It does not establish an independent task-quality result.

