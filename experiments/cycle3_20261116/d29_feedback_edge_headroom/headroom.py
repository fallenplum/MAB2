"""Exploratory replay of the value of continuing a shared coder draft in two ways."""
import hashlib
import json
import statistics
from pathlib import Path


def read(p):return json.loads(p.read_bytes())


def compare(records, field, control):
    pairs=[(r,r[field],r[control]) for r in records]
    by_seed={}
    for s in sorted({r["seed"] for r,_,_ in pairs}):
        by_seed[str(s)]=statistics.mean(x["utility"]-y["utility"] for r,x,y in pairs if r["seed"]==s)
    return dict(quality_difference=statistics.mean(x["quality"]-y["quality"] for _,x,y in pairs),
                utility_difference=statistics.mean(x["utility"]-y["utility"] for _,x,y in pairs),
                improved=sum(x["quality"]>y["quality"] for _,x,y in pairs),
                harmed=sum(x["quality"]<y["quality"] for _,x,y in pairs),utility_by_seed=by_seed)


def evaluate(archive,scores):
    scored=read(scores/"SUMMARY.json")
    assert scored["status"]=="STOP_FORMAL10_APPLICABILITY_ROUTE"
    table={(r["question"],r["seed"],r["condition"]):r for r in read(scores/"PRIVATE_DECISION_TABLE.json")["records"]}
    records=[];eligible=set();support=set()
    for path in sorted(archive.glob("worker_*/unit_*/MATRIX.json")):
        m=read(path);q=m["question_fingerprint"]
        for s in m["execution_seeds"]:
            raw={v["condition"]:v for v in m["conditions"] if v["execution_seed"]==s}
            b,r=(raw[k] for k in ("BOK3","REPAIR3"))
            assert b["nodes"]["M"]["request_fingerprint"]==r["nodes"]["M"]["request_fingerprint"]
            a0=b["programs"][0];r0=r["programs"][0]
            assert a0["program_sha256"]==r0["program_sha256"] and a0["visible"]["cases"]==r0["visible"]["cases"]
            v=a0["visible"]["passed"];assert v in (0,1,2)
            choices={k:table[q,s,k] for k in ("BOK3_EARLY","REPAIR3","SINGLE_C")}
            if v<2:eligible.add(q)
            if choices["BOK3_EARLY"]["quality"]!=choices["REPAIR3"]["quality"]:support.add(q)
            # Only the first draft feedback enters the implementable decision.
            selected="REPAIR3" if v==1 else "BOK3_EARLY"
            oracle=max(("BOK3_EARLY","REPAIR3"),key=lambda k:choices[k]["utility"])
            records.append(dict(seed=s,question=q,visible=v,policy=choices[selected],oracle=choices[oracle],**choices))
    assert len(records)==48 and len({r['question'] for r in records})==16
    res={}
    for policy in ("policy","oracle"):
        controls={c:compare(records,policy,c) for c in ("BOK3_EARLY","REPAIR3","SINGLE_C")}
        gates=dict(mean_headroom=all(c["utility_difference"]>=.02 for c in controls.values()),
                   quality=all(c["quality_difference"]>=-.02 for c in controls.values()),
                   seed_sign=all(x>=0 for k,v in controls.items() if k!="SINGLE_C" for x in v["utility_by_seed"].values()),
                   informative_tasks=len(support)>=4)
        res[policy]=dict(quality=statistics.mean(r[policy]["quality"] for r in records),controls=controls,
                         gates=gates,passed=all(gates.values()))
    return dict(status="HEADROOM_ONLY" if res["oracle"]["passed"] else "NO_CERTIFIED_REPAIR_OR_RESAMPLE_HEADROOM",
                feasible_policy_pass=res["policy"]["passed"],complete_drafts=48,unique_tasks=16,
                discordant_quality_tasks=len(support),tasks_with_any_visible_failure=len(eligible),
                first_visible_counts={str(v):sum(r["visible"]==v for r in records) for v in (0,1,2)},outcomes=res,
                matched_initial_requests=True,divergent_continuation_seeds=True,graph_edge_causal_effect_identified=False,
                new_model_calls=0,GPU_hours=0,independent_confirmation=False,
                summary_sha256=hashlib.sha256((scores/"SUMMARY.json").read_bytes()).hexdigest())
