"""Create a tiny public fixture for the M60 replay pipeline.

The fixture is synthetic and contains no questions, model weights, or sealed
research labels. It exercises the same 27-field worker schema used by the
private replay. Real runs replace this directory with a frozen panel and keep
the sealed answer map outside the public checkout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixture"


def row(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def full_record(index: int) -> dict[str, object]:
    old_source = row([7 - index, 1 + index, 1, 1])
    new_source = row([4, 3 + index, 2, 1])
    base = row([1, 1, 1, 1])
    old_relay = [row([8, 1, 1, 0]), row([7, 1, 1, 1]),
                 row([6, 2, 1, 1]), row([5, 2, 2, 1])]
    new_relay = [old_relay[0], row([8, 1, 1, 0]),
                 old_relay[2], old_relay[3]]
    values: dict[str, object] = {}
    for offset, probability in enumerate((old_source, new_source, base)):
        values[str(offset)] = probability
    for offset, probability in enumerate(old_relay, start=3):
        values[str(offset)] = probability
    for offset, probability in enumerate(new_relay, start=7):
        values[str(offset)] = probability
    values.update({str(11 + offset): 0 for offset in range(16)})
    return values


def build() -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    rows = []
    sealed = {}
    for index in range(4):
        key = hashlib.sha256(f"m60-public-fixture-{index}".encode()).hexdigest()
        rows.append({
            "key": key,
            "full": full_record(index),
            "tokens": [1] * 27,
            "old_relay_fingerprints": [f"row-{i}" for i in range(4)],
            "new_relay_fingerprints": [f"row-{i}-edit" if i == 1 else f"row-{i}" for i in range(4)],
        })
        sealed[key] = 0
    (FIXTURE / "WORKER0.jsonl").write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows),
        encoding="utf-8",
    )
    (FIXTURE / "SEALED.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
    manifest = {
        "panels": {"public_fixture": "fixture"},
        "workers": [0],
        "changed_rows": [1],
        "changed_rows_mode": "fingerprint",
        "output": "fixture_result.json",
        "graph_manifest": {
            "schema": "frozen_graph_edit_manifest_v1",
            "edited_edges": ["source_to_relay_1"],
            "edge_to_relay_descendants": {"source_to_relay_1": [1]},
            "relay_rows": [0, 1, 2, 3],
            "uses_answer_keys": False,
        },
    }
    (ROOT / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "manifest": str(ROOT / "fixture_manifest.json")}))


if __name__ == "__main__":
    build()

