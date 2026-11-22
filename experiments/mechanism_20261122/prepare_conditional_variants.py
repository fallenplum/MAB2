from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("E:/图结构演化/artifacts/private/mechanism_20261122")
BASES = ("m60_conditional_qasc", "m60_conditional_sciq", "m60_conditional_commonsense")
VARIANTS = {"rows1": (1,), "rows2": (1, 3), "rows4": (0, 1, 2, 3)}


def save_variant(base_name, variant_name, changed):
    source = ROOT / base_name
    target = ROOT / f"{base_name}_{variant_name}"
    rows = json.loads((source / "INPUT.json").read_bytes())
    labels = json.loads((source / "SEALED.json").read_bytes())
    edited = []
    for row in rows:
        old = row.get("second_fact", "")
        facts = [old] * 4
        for index in changed:
            facts[index] = ""
        updated = dict(row)
        updated["second_facts_new"] = facts
        updated["second_fact_new"] = facts[1]
        edited.append(updated)
    target.mkdir(parents=True, exist_ok=True)
    for name, value in (("INPUT.json", edited), ("SEALED.json", labels)):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        path = target / name
        if path.exists() and path.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite {path}")
        path.write_bytes(raw)
    receipt = {"schema": "conditional_relay_variant_v1", "base": base_name,
               "variant": variant_name, "changed_rows": list(changed),
               "questions": len(edited),
               "input_sha256": hashlib.sha256((target / "INPUT.json").read_bytes()).hexdigest(),
               "label_sha256": hashlib.sha256((target / "SEALED.json").read_bytes()).hexdigest(),
               "sealed": "private"}
    (target / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


for base in BASES:
    for name, changed in VARIANTS.items():
        save_variant(base, name, changed)
