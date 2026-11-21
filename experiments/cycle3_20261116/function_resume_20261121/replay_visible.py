"""Replay the artificial qualification corpus on the local CPU runtime."""
import json
from pathlib import Path
from function_adapter import run_function

def main():
    checks=0
    programs=0
    mismatch=0
    seen=set()
    for mpath in sorted(Path("/input").glob("worker_*/unit_*/MATRIX.json")):
        matrix=json.loads(mpath.read_bytes())
        payload=json.loads((mpath.parent/"INPUT.json").read_bytes())["payload"]
        for row in matrix["conditions"]:
            for program in row["programs"]:
                key=(matrix["question_fingerprint"],program["program_sha256"])
                if key in seen:continue
                seen.add(key);programs+=1
                for (args,expected),old in zip(payload["visible_cases"],program["visible"]["cases"]):
                    new=run_function(program["code"],payload["function_name"],json.loads(args),json.loads(expected),"/work/cases")
                    fields=("status","returned","typed_return","passed")
                    checks+=1
                    mismatch+=int(any(new.get(f)!=old.get(f) for f in fields))
    if not checks:raise ValueError("Artificial corpus is empty")
    Path("/work/replay.json").write_text(json.dumps(dict(programs=programs,cases=checks,mismatches=mismatch,
        hidden_access=False,artificial_examples_only=True)),encoding="utf-8")

if __name__=="__main__":main()
