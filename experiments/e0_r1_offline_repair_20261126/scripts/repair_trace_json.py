"""Repair only the known Windows-path JSON serialization defect in copies."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parents[1] / "experiments/e0_real_workflow_20261126/results"
OUT = ROOT / "private_repaired"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = [
    "E0_FULL_32.json",
    "E0_STRICT_GATE_COLD_32.json",
    "E0_STRICT_GATE_CANONICAL_32.json",
    "E0_FLAT_32.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if '"model": "' in line or '"knowledge_source": "' in line:
            prefix, rest = line.split('"', 2)[0:2], None
            # Only escape backslashes on the two known top-level path fields.
            marker = '"model": "' if '"model": "' in line else '"knowledge_source": "'
            start = line.index(marker) + len(marker)
            end = line.rfind('"')
            if end > start:
                line = line[:start] + line[start:end].replace('\\', '\\\\') + line[end:]
        lines.append(line)
    return ''.join(lines)


def main() -> None:
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        src = OLD / name
        dst = OUT / name
        raw = src.read_text(encoding="utf-8")
        repaired = repair(raw)
        dst.write_text(repaired, encoding="utf-8")
        parsed = json.loads(repaired)
        records.append({
            "file": name,
            "source_path": str(src),
            "source_sha256": sha(src),
            "repaired_path": str(dst),
            "repaired_sha256": sha(dst),
            "repair": "escape top-level model/knowledge_source Windows backslashes only",
            "parsed": True,
            "top_level_keys": sorted(parsed),
        })
    (ROOT / "results/RAW_TRACE_SERIALIZATION_REPAIR.json").write_text(
        json.dumps({"schema": "mab2_r1_raw_trace_repair_v1", "records": records, "fields_changed": ["serialization_only"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()

