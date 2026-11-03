# G1 elastic execution tools

These tools preserve the original three logical shards and 60000-observation design. They are an execution supplement, not an acquisition algorithm or a completed scientific result. See [the contract addendum](../../protocol/ELASTIC_EXECUTION_ADDENDUM.md).

The repository keeps reviewable source here. A deployed run copies the six execution helpers into `<root>/artifacts/private/performance/` and the observer into `<root>/artifacts/private/monitoring/`; its allocation freezes the deployed byte hashes separately from the original model/executor/scorer source manifest. `SOURCE_HASHES.json` explicitly distinguishes runtime bytes from repository UTF8 LF text.

The migration helper requires the original live v3 run, source and process identity checks, completed qualification receipts, and a pidfd capability check before pausing any owned process. It refuses existing migration receipts and allocations. It is not a generic restart command. Failed attempts are retained and require explicit artifact review before another handoff.

`elastic_worker.py` executes only uncommitted cells, with one atomic owner per question/seed unit. `elastic_supervisor.py` admits the qualified cards and independently qualifies a deferred card before adding it. `elastic_finalize.py` validates all execution, cost and ownership provenance; only explicit commit after all writers stop can freeze outputs and open development labels. `watch_g1.py` is a read-only 30-minute observer with separate operational and scientific completion states. Credentials are supplied only by the caller's runtime environment.

CPU contract tests run from this directory:

```powershell
python -m unittest -v test_watch_g1 test_watch_g1_elastic test_elastic_worker test_elastic_finalize test_elastic_scheduling test_elastic_supervisor test_pidfd_compat test_mirror_only test_mirror_terminal_verify
```

They use synthetic fixtures and the repository's original executor; they do not query GPUs, open real labels, or migrate processes. Engineering tests and qualification checks do not establish the G1 scientific gate or ACL submission readiness.


The running G1 now uses `mirror_only.py` as a read-only local attach layer with four bounded data connections and an independent control heartbeat. It never uploads or launches experiments. `mirror_terminal_verify.py` verifies every committed artifact and all 60000 execution references before terminal publication. Local control counters are distinct from verified downloaded artifacts. See [the mirror recovery record](../../protocol/MIRROR_RECOVERY_20261104.md).
