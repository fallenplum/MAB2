# Corrected maximum-uncertainty replay on the density panels

The three-model fact-drift density panels were rescored after replacing the former `max_uncertainty` alias with the explicit audit-mass and cell-weight ranking policy.

| changed relay rows | warm exact set | explicit max-uncertainty exact set | warm tokens | explicit tokens | Full CTDR token-aware exact set |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.979167 | 0.979167 | 23.2161 | 23.3516 | 1.0 |
| 2 | 0.926215 | 0.926215 | 22.6727 | 22.8082 | 1.0 |
| 4 | 0.844618 | 0.844618 | 22.2977 | 22.4332 | 1.0 |

The correction changes the mean baseline cost slightly and leaves the exact-set comparison unchanged. The density panel is historical development evidence and is not an independent test cohort.

