"""Supplementary provenance check using exact published identity, no numeric suffix matching."""
import gzip
import json
from pathlib import Path
from lineage import parent_id


def load_keys(path, explicit=False):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            key=parent_id(row.get("parent")) if explicit else row["task_id"]
            if key is None or key in out:raise ValueError("Invalid or duplicate published identity")
            out[key]=row["entry_point"]
    return out


def inspect(data):
    source=load_keys(data/"HumanEval.jsonl.gz")
    subtle=load_keys(data/"EvoEval_subtle.jsonl.gz",explicit=True)
    good=set(subtle)&set(source)
    records={}
    for kind in ("verbose","concise"):
        by_id=load_keys(data/f"EvoEval_{kind}.jsonl.gz")
        exact=set(by_id)&set(source)
        compatible={k for k in exact if by_id[k]==source[k]}
        good &= compatible
        records[kind]={"published_ids":len(by_id),"identical_full_task_ids":len(exact),
                       "same_entry_point_on_identical_ids":len(compatible)}
    return dict(variant_counts=records,four_version_one_to_one_parent_groups=len(good),
                comparison="Exact full source task_id and entry_point for both rephrasings; explicit parent for subtle.",
                labels_used=False,model_calls=0,GPU_hours=0,independent_model_effect_established=False)
