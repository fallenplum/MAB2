"""Blindly specified replay of executable program messages and an incoming edge deletion."""
import json
import hashlib
import math
import statistics

def read(p):return json.loads(p.read_bytes())

def replay(names,nodes,outcomes):
    choices=[];calls={};programs={}
    for key in names:
        if key not in nodes:raise ValueError("Missing reachable branch")
        node,prog=nodes[key];fp=node["request_fingerprint"]
        calls[fp]=node["generation"]
        programs[prog["program_sha256"]]=prog
        choices.append(prog)
        if prog["visible"]["passed"]==2:break
    best=max(choices,key=lambda p:p["visible"]["passed"])
    y=int(all(c["passed"] for c in outcomes[best["program_sha256"]]["cases"]))
    tokens=sum(g["input_tokens"]+g["output_tokens"] for g in calls.values())
    cpu=sum(p["visible"]["seconds"] for p in programs.values())
    return {"quality":y,"tokens":tokens,"utility":y-tokens/200000-cpu/10000}


def evaluate(view,scores,hidden):
    source=read(scores/"SUMMARY.json");assert source["status"]=="STOP_FORMAL10_APPLICABILITY_ROUTE"
    gold=read(hidden/"HIDDEN_RESULTS.json")["results"]
    reference={(r["question"],r["seed"],r["condition"]):r for r in read(scores/"PRIVATE_DECISION_TABLE.json")["records"]}
    options={"BOK3_EARLY":("C","B1","B2"),"REPAIR3":("C","T1","T2"),
             "C_Q_B1":("C","Q","B1"),"C_B1_Q":("C","B1","Q"),"C_Q":("C","Q")}
    items=[];active=set();changed=set();nonzero=0;shifts=0
    for f in sorted(view.glob("worker_*/unit_*/MATRIX.json")):
        m=read(f);q=m["question_fingerprint"]
        for s in m["execution_seeds"]:
            rows={r["condition"]:r for r in m["conditions"] if r["execution_seed"]==s}
            nodes={}
            for condition,alias in [("SINGLE_Q",{"M":"Q"}),("BOK3",{"M":"C","B1":"B1","B2":"B2"}),
                                    ("REPAIR3",{"M":"C","T1":"T1","T2":"T2"})]:
                r=rows[condition]
                for program in r["programs"]:
                    matches=[(k,n) for k,n in r["nodes"].items() if n["request_fingerprint"]==program["source_request_fingerprint"]]
                    assert len(matches)==1
                    k,n=matches[0];id_=alias[k]
                    if id_ in nodes: assert n["request_fingerprint"]==nodes[id_][0]["request_fingerprint"]
                    nodes[id_]=(n,program)
            v=nodes["C"][1]["visible"]["passed"]
            choices={k:replay(names,nodes,gold[q]) for k,names in options.items()}
            for k in ("BOK3_EARLY","REPAIR3"):
                r=reference[q,s,k]
                assert r["quality"]==choices[k]["quality"] and r["tokens"]==choices[k]["tokens"]
                assert math.isclose(r["utility"],choices[k]["utility"],abs_tol=1e-10)
            direct=reference[q,s,"SINGLE_C"]
            choices["SINGLE_C"]={k:direct[k] for k in ("quality","utility","tokens")}
            choices["D29_RULE"]=choices["REPAIR3" if v==1 else "BOK3_EARLY"]
            choices["FEASIBLE"]=choices["C_Q_B1" if v==0 else "REPAIR3"]
            choices["ORACLE"]=max((choices[k] for k in options),key=lambda r:r["utility"])
            g0=choices["C_B1_Q"]["quality"]-replay(("C","B1"),nodes,gold[q])["quality"]
            g1=choices["C_Q"]["quality"]-replay(("C",),nodes,gold[q])["quality"]
            if g1:active.add(q);nonzero+=1
            if g0!=g1:changed.add(q);shifts+=1
            items.append((s,choices))
    assert len(items)==48
    results={}
    for name in ("FEASIBLE","ORACLE"):
        controls={}
        for c in (*options,"D29_RULE","SINGLE_C"):
            pairs=[(s,r[name]["quality"]-r[c]["quality"],r[name]["utility"]-r[c]["utility"]) for s,r in items]
            controls[c]=dict(quality_difference=statistics.mean(x[1] for x in pairs),
                utility_difference=statistics.mean(x[2] for x in pairs),improved=sum(x[1]>0 for x in pairs),
                harmed=sum(x[1]<0 for x in pairs),utility_by_seed={str(s):statistics.mean(x[2] for x in pairs if x[0]==s) for s in sorted({x[0] for x in pairs})})
        gates=dict(utility=all(v["utility_difference"]>=.02 for v in controls.values()),
                   quality=all(v["quality_difference"]>=-.02 for v in controls.values()),
                   seed_consistency=all(d>=0 for c in ("BOK3_EARLY","REPAIR3","D29_RULE") for d in controls[c]["utility_by_seed"].values()),
                   current_credit_tasks=len(active)>=4,changed_credit_tasks=len(changed)>=4)
        results[name]=dict(quality=statistics.mean(r[name]["quality"] for _,r in items),controls=controls,
                           gates=gates,passed=all(gates.values()))
    return dict(status="CONSIDER_NEW_INDEPENDENT_PROTOCOL" if results["ORACLE"]["passed"] else "FREEZE_EXPOSED_APPS_POLICY_SEARCH",
                tasks=16,seeds=3,current_nonzero_tasks=len(active),shift_nonzero_tasks=len(changed),
                current_nonzero_events=nonzero,shift_nonzero_events=shifts,results=results,independent=False,
                new_GPU_hours=0,new_model_calls=0,score_sha256=hashlib.sha256((scores/"SUMMARY.json").read_bytes()).hexdigest())
