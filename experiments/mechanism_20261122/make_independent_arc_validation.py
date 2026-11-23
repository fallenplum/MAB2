"""Create the independent ARC-Challenge validation cohort used by MAB2.

The script accepts either a Hugging Face dataset name or a local Arrow/JSON
export. It keeps the first 128 four-choice records with A--D answer keys,
creates the same graph-edit metadata for rows1/2/4 and writes the answer map
separately from the controller input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_records(source: str) -> list[dict]:
    path = Path(source)
    if path.exists():
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))
    from datasets import load_dataset
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    return list(dataset)


def build(source: str, output_dir: Path, limit: int = 128) -> None:
    records = load_records(source)
    rows = []
    gold = {}
    for record in records:
        choices = record["choices"]
        texts = list(choices["text"])
        answer = str(record["answerKey"])
        if len(texts) != 4 or answer not in {"A", "B", "C", "D", "1", "2", "3", "4"}:
            continue
        key = hashlib.sha256((record["question"] + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()
        rows.append({
            "key": key,
            "stem": record["question"],
            "options": texts,
            "first_fact": "A first reviewer inspects the question and available hypotheses.",
            "second_fact": "A second reviewer compares the evidence against each hypothesis.",
            "control_fact": "A control reviewer checks the wording and answer choices.",
            "second_fact_new": "A newly added reviewer independently rechecks the evidence and restates the decision.",
        })
        gold[key] = "ABCD".index(answer) if answer in "ABCD" else int(answer) - 1
        if len(rows) == limit:
            break
    if len(rows) != limit:
        raise RuntimeError(f"expected {limit} four-choice records, found {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for density, changed in ((1, [1]), (2, [1, 2]), (4, [0, 1, 2, 3])):
        variant = []
        for row in rows:
            item = dict(row)
            facts = [row["second_fact"]] * 4
            for index in changed:
                facts[index] = row["second_fact_new"] + f" Variant {index}"
            item["second_facts_new"] = facts
            variant.append(item)
        (output_dir / f"arc_validation_rows{density}_INPUT.json").write_text(json.dumps(variant, ensure_ascii=False), encoding="utf-8")
    (output_dir / "arc_validation_gold.json").write_text(json.dumps(gold), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="ai2_arc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output_dir)

