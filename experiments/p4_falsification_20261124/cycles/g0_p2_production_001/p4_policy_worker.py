"""Single-step public-state policy subprocess used by the P4 dev runtime."""
from __future__ import annotations

import argparse
import json
import sys


def choose(state: dict) -> dict:
    allowed = {"actions", "graph", "predictions", "weights", "known_labels", "legal_actions", "costs", "budget_remaining", "certificate"}
    if set(state) != allowed:
        return {"type": "abstain", "reason": "PUBLIC_STATE_SCHEMA_MISMATCH"}
    cert = state["certificate"]
    if cert["status"] in {"CERTIFIED", "FULL_RESOLVED"}:
        return {"type": "stop", "action": cert["action"]}
    if cert["status"] == "INVALID":
        return {"type": "abstain", "reason": "INVALID_CERTIFICATE"}
    legal = list(state["legal_actions"])
    if not legal:
        return {"type": "abstain", "reason": "NO_LEGAL_ACTION"}
    node = min(legal, key=lambda item: (state["costs"][item], item))
    if state["costs"][node] > state["budget_remaining"]:
        return {"type": "abstain", "reason": "BUDGET_EXHAUSTED"}
    return {"type": "query", "node_id": node}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    state = json.loads(sys.stdin.read())
    json.dump(choose(state), sys.stdout, ensure_ascii=False, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
