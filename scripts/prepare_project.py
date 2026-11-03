"""Copy exact source assets into the declared root and create disjoint GSM splits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata
import zipfile
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--source-data", required=True)
    p.add_argument("--source-plan", required=True)
    args = p.parse_args()
    root, source = Path(args.root), Path(args.source_data)
    for part in ["protocol", "configs", "data/raw/gsm8k", "data/derived", "data/sealed", "models", "artifacts/audit", "results", "logs", "artifacts/private"]:
        (root / part).mkdir(parents=True, exist_ok=True)
    records, seen, counts = {}, {}, {}
    for split in ["train", "test"]:
        src = source / f"{split}.jsonl"
        dst = root / "data/raw/gsm8k" / src.name
        if dst.exists() and sha(dst) != sha(src):
            raise RuntimeError("Refuse overwrite of different raw data")
        if not dst.exists():
            shutil.copyfile(src, dst)
        values = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[split] = {"rows": len(values), "sha256": sha(dst)}
        unique = []
        for item in values:
            key = hashlib.sha256(normalized(item["question"]).encode()).hexdigest()
            if key in seen:
                if seen[key]["answer"] != item["answer"]:
                    raise ValueError("Duplicate question with conflicting gold")
                if split != seen[key]["source_split"]:
                    raise ValueError("Official train/test overlap")
                continue
            row = {"id": "gsm8k_" + key[:20], "question": item["question"], "answer": item["answer"], "task": "gsm8k", "source_split": split}
            seen[key] = row
            unique.append(row)
        records[split] = unique
    train = sorted(records["train"], key=lambda row: hashlib.sha256(("graph-credit-split-v1:" + row["id"]).encode()).hexdigest())
    groups = {"qualification": train[:16], "motivation": train[16:216], "calibration": train[216:616], "online_stream": train[616:], "final_test": records["test"]}
    manifest = []
    for split, rows in groups.items():
        dest = root / "data" / ("sealed" if split == "final_test" else "derived")
        lines(dest / f"{split}_questions.jsonl", [{**{k:v for k,v in r.items() if k not in {"answer", "source_split"}}, "split":split} for r in rows])
        lines(dest / f"{split}_labels.jsonl", [{"id":r["id"], "task":"gsm8k", "answer":r["answer"]} for r in rows])
        manifest += [{"id":r["id"], "split":split, "source_split":r["source_split"]} for r in rows]
    dump(root / "data/derived/split_manifest.json", {"version":"graph_credit_gsm8k_v1", "rule":"NFKC lowercase whitespace question dedup; SHA256 fixed ordering", "groups":{k:len(v) for k,v in groups.items()}, "assignments":manifest})
    plan = Path(args.source_plan)
    shutil.copyfile(plan, root / "protocol" / plan.name)
    w, m = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    with zipfile.ZipFile(plan) as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
        text = []
        for i, block in enumerate(doc.find(w + "body")):
            content = "".join(t.text or "" for t in block.iter() if t.tag in {w+"t",m+"t"})
            if content:
                text.append(f"[{i:04}] {content}")
        (root / "protocol/PLAN_EXTRACT.txt").write_text("\n".join(text), encoding="utf-8")
        structure = {"paragraphs":sum(1 for _ in doc.iter(w+"p")), "tables":sum(1 for _ in doc.iter(w+"tbl")), "math":sum(1 for _ in doc.iter(m+"oMath"))}
    dump(root / "protocol/source_manifest.json", {"plan_sha256":sha(plan), "plan_structure":structure, "data":counts})
    dump(root / "artifacts/audit/split_audit.json", {"status":"PASS", "groups":{k:len(v) for k,v in groups.items()}, "unique_ids":len(manifest), "cross_split_overlap":0, "official_test_for_tuning":False})
    print(json.dumps({"status":"PREPARED", "groups":{k:len(v) for k,v in groups.items()}, "structure":structure}))


if __name__ == "__main__":
    main()
