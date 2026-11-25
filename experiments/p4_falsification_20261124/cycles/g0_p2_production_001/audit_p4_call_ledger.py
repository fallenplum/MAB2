from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STAGES = ("history", "online", "post_stop_audit")
HASH_FIELDS = ("input_digest", "output_digest", "raw_output_digest")


def audit(payload: dict) -> dict:
    errors: list[str] = []
    events_by_stage = payload.get("event_rows")
    counts = payload.get("actual_call_counts")
    tokens = payload.get("actual_token_counts")
    costs = payload.get("actual_cost_units")
    if not all(isinstance(value, dict) for value in (events_by_stage, counts, tokens, costs)):
        return {"status": "INVALID", "errors": ["MISSING_LEDGER_SECTIONS"]}
    seen: set[str] = set()
    for stage in STAGES:
        events = events_by_stage.get(stage)
        if not isinstance(events, list):
            errors.append(f"{stage}:MISSING_EVENT_LIST")
            continue
        if counts.get(stage) != len(events):
            errors.append(f"{stage}:CALL_COUNT_MISMATCH")
        stage_cost = 0
        stage_tokens = 0
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                errors.append(f"{stage}:{index}:MALFORMED_EVENT")
                continue
            call_id = event.get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                errors.append(f"{stage}:{index}:DUPLICATE_OR_EMPTY_CALL_ID")
            else:
                seen.add(call_id)
            if not isinstance(event.get("node_id"), str) or not event["node_id"]:
                errors.append(f"{stage}:{index}:MISSING_NODE_ID")
            input_tokens = event.get("input_tokens")
            paid_cost = event.get("paid_cost")
            if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens <= 0:
                errors.append(f"{stage}:{index}:INVALID_INPUT_TOKENS")
            if isinstance(paid_cost, bool) or not isinstance(paid_cost, int) or paid_cost <= 0:
                errors.append(f"{stage}:{index}:INVALID_PAID_COST")
            elif isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and paid_cost != input_tokens + 1:
                errors.append(f"{stage}:{index}:TOKEN_COST_DISAGREEMENT")
            stage_cost += paid_cost if isinstance(paid_cost, int) and not isinstance(paid_cost, bool) else 0
            stage_tokens += input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else 0
            for field in HASH_FIELDS:
                value = event.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                    errors.append(f"{stage}:{index}:INVALID_{field.upper()}")
            for field in ("model_version", "tool_version"):
                if not isinstance(event.get(field), str) or not event[field]:
                    errors.append(f"{stage}:{index}:MISSING_{field.upper()}")
            if event.get("parse_status") != "OK":
                errors.append(f"{stage}:{index}:NON_OK_PARSE_STATUS")
        if costs.get(stage) != stage_cost:
            errors.append(f"{stage}:COST_TOTAL_MISMATCH")
        if tokens.get(stage) != stage_tokens:
            errors.append(f"{stage}:TOKEN_TOTAL_MISMATCH")
    hook = payload.get("hook_calls", {})
    total_events = sum(len(events_by_stage.get(stage, [])) for stage in STAGES if isinstance(events_by_stage.get(stage), list))
    if hook.get("pre") != total_events or hook.get("post") != total_events:
        errors.append("HOOK_EVENT_TOTAL_MISMATCH")
    return {
        "status": "PASS_INDEPENDENT_EVENT_RECONCILIATION_DEV" if not errors else "INVALID",
        "errors": errors,
        "event_count": total_events,
        "unique_call_id_count": len(seen),
        "cost_total": sum(value for stage in STAGES if isinstance((value := costs.get(stage)), int) and not isinstance(value, bool)),
        "input_token_total": sum(value for stage in STAGES if isinstance((value := tokens.get(stage)), int) and not isinstance(value, bool)),
        "scope": "Independent code path over hook event rows; validates internal event/cost consistency, not OS isolation or external blind signoff.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    receipt = audit(payload)
    receipt["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
