from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("E:/图结构演化")
PRIVATE = ROOT / "artifacts/private/mechanism_20261122/m60_conditional_commonsense"
RAW = ROOT / "data/raw/m43-commonsense-train/train.parquet"


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def previous_stems():
    seen = set()
    for path in (ROOT / "artifacts/private").glob("m60_*/INPUT.json"):
        try:
            for row in json.loads(path.read_bytes()):
                if "stem" in row:
                    seen.add(norm(row["stem"]))
        except (OSError, json.JSONDecodeError):
            continue
    return seen


def build():
    seen = previous_stems()
    selected = []
    for row in sorted(pq.read_table(RAW).to_pylist(),
                      key=lambda value: digest("CTDR-CONDITIONAL-COMMONSENSE-20261122|" + value["id"])):
        stem = row["question"]
        if norm(stem) in seen:
            continue
        choices = row["choices"]
        options = list(choices["text"])
        answer_index = list(choices["label"]).index(row["answerKey"])
        if len(options) != 5 or len(set(map(norm, options))) != 5:
            continue
        drop = next(index for index in range(5) if index != answer_index)
        kept = [index for index in range(5) if index != drop]
        key = digest("CTDR-CONDITIONAL-COMMONSENSE-20261122|" + row["id"])
        ordered = sorted((options[index] for index in kept), key=lambda text: digest(key + "|" + norm(text)))
        correct = options[answer_index]
        selected.append({
            "key": key,
            "stem": stem,
            "options": ordered,
            "first_fact": "",
            "second_fact": "The question concerns the concept " + row["question_concept"] + ".",
            "second_fact_new": "",
            "control_fact": "",
            "answer": ordered.index(correct),
        })
        seen.add(norm(stem))
        if len(selected) == 128:
            break
    if len(selected) != 128:
        raise RuntimeError(f"only {len(selected)} new Commonsense rows")
    labels = {row["key"]: row.pop("answer") for row in selected}
    return selected, labels


def main():
    rows, labels = build()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for name, value in (("INPUT.json", rows), ("SEALED.json", labels)):
        target = PRIVATE / name
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite {target}")
        target.write_bytes(raw)
    receipt = {"schema": "conditional_relay_input_v1", "questions": len(rows),
               "input_sha256": hashlib.sha256((PRIVATE / "INPUT.json").read_bytes()).hexdigest(),
               "label_sha256": hashlib.sha256((PRIVATE / "SEALED.json").read_bytes()).hexdigest(),
               "edit": "delete second_fact only for advised A relay row", "sealed": "private"}
    (PRIVATE / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
