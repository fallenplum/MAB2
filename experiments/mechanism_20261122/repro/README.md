# M60 clean replay

This directory is a clean-checkout entry point for the CPU replay layer. The
public fixture is synthetic and contains no question text, model weights, or
private answer key. It verifies the worker schema, old-history reuse, interval
updates, stopping rules, and token accounting.

## Quick start

From a checkout of `fallenplum/MAB2`, run:

```text
cd experiments/mechanism_20261122/repro
python prepare_inputs.py
python run_replay.py --manifest fixture_manifest.json
python -m unittest discover -s tests -v
```

The pipeline has six explicit stages:

1. `prepare inputs` creates the deterministic fixture and manifest.
2. `run worker outputs` is represented by the fixture's `WORKER0.jsonl`.
3. `score old transcript` reconstructs the old audit history.
4. `run post-edit baselines` evaluates stale, reweighted warm history,
   maximum uncertainty, periodic and random policies.
5. `run CTDR` evaluates source transport and transport plus selective ranking.
6. `generate M60 table` writes `fixture_result.json` with exact-set agreement,
   action error, false drop, false keep, query count and token cost.

## Real M60 replay

The private run replaces `fixture/` and `fixture_manifest.json` with a frozen
panel manifest. The worker stage uses
`conditional_mcqa_worker.py` and requires the private model checkpoint. The
CPU scorer is `conditional_real_score.py`, which imports `ctdr.py`,
`blind_scheduler.py`, `conditional_schema.py` and `m60_schema.py`. The sealed
answer map is read only for reporting and stays outside a public checkout.

Each worker JSON object contains `full` keys `0..26`. Keys `0..10` are eleven
normalized probability rows, keys `11..26` are the sixteen terminal labels,
and `tokens` has twenty-seven prices. Every controller receives the same old
history, post-edit source, relay table and token vector. The scorer writes one
ledger entry per action and applies the same decision stopping rule.

The reported oracle lower bound is a declared zero-incremental-token
information bound. It assumes an oracle already knows the post-edit decision
and is not executable. It must not be interpreted as a learned baseline.

The CPU replay does not require model weights. Worker generation, sealed-key
scoring and any real service latency measurement remain private execution
steps. Pin the checkout to the release commit recorded in the accompanying
research audit before reproducing published numbers.

