# Phase 2 replay

The server runner is the local source file
`experiments/p2_multihop_20261124/p2_service_runner.py`. It supports missing
`changed_rows` and four-option SciQ fixtures, then adds strict-certificate
routes named `certificate_voi`, `certificate_random`,
`certificate_uncertainty`, and `certificate_periodic`.

Private server fixtures used for the confirmatory replay were

```text
/home/chen/mab2_conditional/code/qasc_interface_drift_rows2_INPUT.json
/home/chen/mab2_conditional/code/sciq_interface_drift_rows2_INPUT.json
/home/chen/mab2_conditional/real_labels/qasc.json
/home/chen/mab2_conditional/real_labels/sciq.json
```

The model paths were `/home/chen/KDBPT/KD/models/Qwen-0.5B` and
`/home/chen/KDBPT/KD/models/Qwen-14B`. The command uses
`MAB2_MICROBATCH=1`, `--limit 32`, `--offset 0`, and one GPU per process.
Sealed labels are passed only to the final scorer.

