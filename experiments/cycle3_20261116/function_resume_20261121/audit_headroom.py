"""Recompute both gold upper bounds from raw branch requests and audited pilot returns."""
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def check(view, scores, diag):
    source=read(scores/"PRIVATE_DECISION_TABLE.json")
    table={(r["question"],r["seed"],r["condition"]):r for r in source["records"]}
    saved=read(diag/"PRIVATE_ORACLE_CHOICES.json")
    arms=("G1","G1_MINUS_EU","G1_MINUS_EV")
    controls=("SINGLE_Q","SINGLE_C","BOK3","BOK3_EARLY","REPAIR3")
    frontiers=("G1_OBSERVED_ORACLE","FREE_PRESELECTION_ORACLE")
    choices={(r["question"],r["seed"],r["frontier"],r["arm"]):r for r in saved["all_choices"]}
    assert len(choices)==288
    selected={(r["question"],r["seed"],r["frontier"]):r["arm"] for r in saved["records"]}
    rebuilt={f:[] for f in frontiers}
    summary=read(diag/"SUMMARY.json")
    for p in view.glob("worker_*/unit_*/MATRIX.json"):
        m=read(p);q=m["question_fingerprint"]
        for s in m["execution_seeds"]:
            rows={r["condition"]:r for r in m["conditions"] if r["execution_seed"]==s}
            for f in frontiers:
                options=[]
                for a in arms:
                    included={a}
                    if f==frontiers[0]:included.add("G1")
                    calls={}; programs={}
                    for arm in sorted(included):
                        for node in rows[arm]["nodes"].values():
                            g=node["generation"]
                            calls[node["request_fingerprint"]]=g["input_tokens"]+g["output_tokens"]
                        for prog in rows[arm]["programs"]:
                            programs[prog["program_sha256"]]=prog["visible"]["seconds"]
                    cost=sum(calls.values());cpu=sum(programs.values());y=table[q,s,a]["quality"]
                    u=y-.000005*cost-.0001*cpu
                    z=choices[q,s,f,a]
                    assert z["tokens"]==cost and math.isclose(z["visible_execution_seconds"],cpu,abs_tol=1e-9)
                    assert z["quality"]==y and math.isclose(z["utility"],u,abs_tol=1e-10)
                    options.append((a,y,u))
                winner=max(options,key=lambda x:x[2])
                assert selected[q,s,f]==winner[0]
                rebuilt[f].append((q,s,winner[1],winner[2]))
    passed={}
    for f,rows in rebuilt.items():
        assert len(rows)==48
        front=summary["frontiers"][f]
        values=[]
        for c in controls:
            comp=front["comparisons"][c]
            d=[(s,y-table[q,s,c]["quality"],u-table[q,s,c]["utility"]) for q,s,y,u in rows]
            quality=sum(v[1] for v in d)/48;utility=sum(v[2] for v in d)/48
            assert math.isclose(comp["quality_difference"],quality,abs_tol=1e-10)
            assert math.isclose(comp["utility_difference"],utility,abs_tol=1e-10)
            assert (comp["harmed"],comp["improved"])==(sum(v[1]<0 for v in d),sum(v[1]>0 for v in d))
            good_seed=True
            for s in sorted({v[0] for v in d}):
                effect=sum(v[2] for v in d if v[0]==s)/16
                assert math.isclose(comp["utility_by_seed"][str(s)],effect,abs_tol=1e-10)
                good_seed &= effect>=0
            values.append(quality>=-.02 and utility>=.02 and (c not in ("BOK3_EARLY","REPAIR3") or good_seed))
        passed[f]=all(values) and read(scores/"SUMMARY.json")["credit"]["nonzero_current_tasks"]>=4
        assert passed[f]==front["all_gates"]
    return dict(status="PASS_INDEPENDENT_D26_COST_AND_DECISION_RECONSTRUCTION",recomputed_choices=len(choices),
                verified_selections=len(selected),frontier_passes=passed,model_calls=0,GPU_hours=0)
