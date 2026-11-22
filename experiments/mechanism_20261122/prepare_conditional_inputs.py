from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("E:/图结构演化")
PRIVATE = ROOT / "artifacts/private/mechanism_20261122"
QASC_RAW = ROOT / "data/raw/m56-qasc-train/train.parquet"
SCIQ_RAW = ROOT / "data/raw/m49-sciq-train/train.parquet"


def norm(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def previous(path_names):
    seen = set()
    for name in path_names:
        path = PRIVATE / name / "INPUT.json"
        if path.exists():
            for row in json.loads(path.read_bytes()):
                seen.add(norm(row["stem"]))
    return seen


def qasc():
    seen = previous(["m60_qasc_r5", "m60_qasc_b_r2", "m60_qasc", "m60_qasc_r2", "m60_qasc_r3", "m60_qasc_r4"])
    entries = []
    answers = {}
    for row in sorted(pq.read_table(QASC_RAW).to_pylist(), key=lambda r: digest("CTDR-CONDITIONAL-QASC-20261122|" + r["id"])):
        question = norm(row["question"])
        facts = [row["fact1"], row["fact2"]]
        choices = row["choices"]
        options = choices["text"]
        labels = choices["label"]
        if question in seen or not all(map(norm, facts)) or len(set(map(norm, options))) != 8:
            continue
        if labels != list("ABCDEFGH") or row["answerKey"] not in labels:
            continue
        key = digest("CTDR-CONDITIONAL-QASC-20261122|" + row["id"])
        ordered = sorted(options, key=lambda text: digest(key + "|" + norm(text)))
        correct = options[labels.index(row["answerKey"])]
        entries.append((key, row["question"], ordered, facts))
        answers[key] = ordered.index(correct)
        seen.add(question)
        if len(entries) == 128:
            break
    if len(entries) != 128:
        raise RuntimeError(f"only {len(entries)} new QASC rows")
    indices = sorted(range(len(entries)), key=lambda i: (len(entries[i][3][0].split()), entries[i][0]))
    rows = []
    for index, (key, stem, options, facts) in enumerate(entries):
        rows.append({"key": key, "stem": stem, "options": options,
                     "first_fact": facts[0], "second_fact": facts[1],
                     "second_fact_new": "", "control_fact": ""})
    for left, right in zip(indices[::2], indices[1::2]):
        rows[left]["control_fact"] = rows[right]["first_fact"]
        rows[right]["control_fact"] = rows[left]["first_fact"]
    return rows, {str(k): int(v) for k, v in answers.items()}


def sciq():
    seen = previous(["m60_sciq_r2", "m60_sciq_b", "m60_sciq", "m60_sciq_b_r2"])
    entries = []
    answers = {}
    for row in sorted(pq.read_table(SCIQ_RAW).to_pylist(), key=lambda r: digest("CTDR-CONDITIONAL-SCIQ-20261122|" + norm(r["question"]))):
        question = norm(row["question"])
        options = [row[k].strip() for k in ("correct_answer", "distractor1", "distractor2", "distractor3")]
        support = row["support"].strip()
        if question in seen or not support or len(support.split()) < 20 or len(support.split()) > 200:
            continue
        if len(set(map(norm, options))) != 4:
            continue
        key = digest("CTDR-CONDITIONAL-SCIQ-20261122|" + norm(row["question"]))
        ordered = sorted(options, key=lambda text: digest(key + "|" + norm(text)))
        entries.append((key, row["question"], ordered, support))
        answers[key] = ordered.index(options[0])
        seen.add(question)
        if len(entries) == 128:
            break
    if len(entries) != 128:
        raise RuntimeError(f"only {len(entries)} new SciQ rows")
    rows = [{"key": key, "stem": stem, "options": options,
             "first_fact": support, "second_fact": support,
             "second_fact_new": "", "control_fact": ""}
            for key, stem, options, support in entries]
    return rows, {str(k): int(v) for k, v in answers.items()}


def save(name, maker):
    folder = PRIVATE / name
    folder.mkdir(parents=True, exist_ok=True)
    rows, labels = maker()
    for filename, value in (("INPUT.json", rows), ("SEALED.json", labels)):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        destination = folder / filename
        if destination.exists() and destination.read_bytes() != raw:
            raise RuntimeError(f"refusing to overwrite {destination}")
        destination.write_bytes(raw)
    receipt = {"schema": "conditional_relay_input_v1", "questions": len(rows),
               "input_sha256": hashlib.sha256((folder / "INPUT.json").read_bytes()).hexdigest(),
               "label_sha256": hashlib.sha256((folder / "SEALED.json").read_bytes()).hexdigest(),
               "edit": "delete second_fact only for advised A relay row",
               "sealed": "private"}
    (folder / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(name, json.dumps(receipt, indent=2))


if __name__ == "__main__":
    save("m60_conditional_qasc", qasc)
    save("m60_conditional_sciq", sciq)
