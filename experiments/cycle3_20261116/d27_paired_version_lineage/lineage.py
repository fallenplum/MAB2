"""Count published parent links without inspecting solutions, prompts or tests."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

VARIANTS=("subtle","verbose","concise")

def parent_id(value):
    if isinstance(value,list) and len(value)==1:
        value=value[0]
    if type(value) is int:
        return "HumanEval/"+str(value)
    if isinstance(value,str) and re.fullmatch(r"HumanEval/(0|[1-9][0-9]*)",value):
        return value
    return None

def run(data):
    files={}
    groups={}
    with gzip.open(data/"HumanEval.jsonl.gz","rt",encoding="utf-8") as f:
        source_ids=[json.loads(line)["task_id"] for line in f]
    assert len(source_ids)==len(set(source_ids))
    parents=set(source_ids)
    for v in VARIANTS:
        path=data/f"EvoEval_{v}.jsonl.gz"
        counts=Counter();shapes=Counter();child_ids=set();unmatched=0
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                row=json.loads(line)
                child=row["task_id"]
                if child in child_ids:raise ValueError("Duplicate child identifier")
                child_ids.add(child)
                raw=row.get("parent")
                shape=type(raw).__name__+(":"+",".join(sorted({type(x).__name__ for x in raw})) if isinstance(raw,list) else "")
                shapes[shape]+=1
                parent=parent_id(raw)
                if parent not in parents:unmatched+=1
                else:counts[parent]+=1
        groups[v]={p for p,n in counts.items() if n==1}
        files[v]=dict(rows=len(child_ids),unique_valid_parents=len(counts),unmatched=unmatched,
                      ambiguous_parent_groups=sum(n>1 for n in counts.values()),parent_value_shapes=dict(shapes),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(source_groups=len(parents),variants=files,one_to_one_all_three=len(set.intersection(*groups.values())),
                subtle_verbose_intersection=len(groups["subtle"]&groups["verbose"]),
                subtle_concise_intersection=len(groups["subtle"]&groups["concise"]),
                source_sha256=hashlib.sha256((data/"HumanEval.jsonl.gz").read_bytes()).hexdigest(),
                score_or_solution_used=False,model_calls=0,GPU_hours=0,
                limit="Identifier linkage only. Exposure, license scope for redistribution, pretraining contamination and decision effect are not established.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("data",type=Path);p.add_argument("out",type=Path);a=p.parse_args()
    with a.out.open("x",encoding="utf-8") as f:json.dump(run(a.data),f,indent=2)
