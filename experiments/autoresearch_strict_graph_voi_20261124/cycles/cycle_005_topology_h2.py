"""Cycle 005 bounded horizon-2 topology VOI candidate.

Uses only current source/terminal observations and a deterministic completion
heuristic at the rollout leaf. The planner is receding-horizon and therefore
remains online. It is a scalable candidate, not the exponential exact-DP
reference.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import itertools
import numpy as np
from strict_graph_voi_cpu import FAMILY_SPECS, METHODS_H1, aggregate, certificate, make_case, run_static, run_dp, valid_cached_terminals, terminal_label


def run_h2(case: dict, horizon: int = 2) -> dict:
    target = aggregate(case["new_labels"], case["weights"])
    assignments = list(itertools.product((0, 1), repeat=case["source_count"]))

    def posterior(sources, terminals):
        return [a for a in assignments
                if all(a[i] == v for i, v in sources.items())
                and all(terminal_label(a, case["deps"][i], i, case["salt"]) == v for i, v in terminals.items())]

    def known(sources, terminals):
        k = valid_cached_terminals(case, sources)
        k.update(terminals)
        return k

    def finish_cost_for_assignment(a, sources, terminals):
        k = known(sources, terminals)
        order = sorted((i for i in range(case["terminal_count"]) if i not in k), key=lambda i: (case["weights"][i] / max(case["costs"][case["source_count"] + i], 1e-9), i), reverse=True)
        c = 0.0
        cached = valid_cached_terminals(case, sources)
        for i in order:
            k[i] = terminal_label(a, case["deps"][i], i, case["salt"])
            c += case["costs"][case["source_count"] + i]
            safe, _ = certificate(k, case["weights"], case["terminal_count"])
            if safe:
                break
        return c

    def leaf_value(sources, terminals):
        post = posterior(sources, terminals)
        if not post:
            return float("inf")
        k = known(sources, terminals)
        safe, _ = certificate(k, case["weights"], case["terminal_count"])
        if safe:
            return 0.0
        unknown_source_cost = sum(case["costs"][i] for i in range(case["source_count"]) if i not in sources)
        return unknown_source_cost + float(np.mean([finish_cost_for_assignment(a, sources, terminals) for a in post]))

    def actions(sources, terminals):
        k = known(sources, terminals)
        for i in range(case["source_count"]):
            if i not in sources:
                yield "source", i
        for i in range(case["terminal_count"]):
            if i not in terminals and i not in k and all(s in sources for s in case["deps"][i]):
                yield "terminal", i

    def transition(sources, terminals, kind, idx, value):
        s2 = dict(sources); t2 = dict(terminals)
        if kind == "source": s2[idx] = value
        else: t2[idx] = value
        return s2, t2

    def branch_values(sources, terminals, kind, idx):
        post = posterior(sources, terminals)
        if not post: return []
        if kind == "source":
            vals = [(v, [a for a in post if a[idx] == v]) for v in (0, 1)]
        else:
            buckets = {}
            for a in post:
                v = terminal_label(a, case["deps"][idx], idx, case["salt"])
                buckets.setdefault(v, []).append(a)
            vals = list(buckets.items())
        total = len(post)
        out=[]
        for v, b in vals:
            if b:
                s2,t2=transition(sources,terminals,kind,idx,v)
                out.append((len(b)/total,s2,t2))
        return out

    def rollout_value(sources, terminals, depth):
        k = known(sources, terminals)
        safe, _ = certificate(k, case["weights"], case["terminal_count"])
        if safe: return 0.0
        acts = list(actions(sources, terminals))
        if not acts: return leaf_value(sources, terminals)
        if depth <= 0: return leaf_value(sources, terminals)
        best = float("inf")
        for kind, idx in acts:
            c = case["costs"][idx] if kind == "source" else case["costs"][case["source_count"] + idx]
            val = c + sum(p * rollout_value(s2,t2,depth-1) for p,s2,t2 in branch_values(sources,terminals,kind,idx))
            if val < best: best=val
        return best

    sources={}; terminals={}; total_cost=0.0; queries=0
    while True:
        k=known(sources,terminals)
        safe,choice=certificate(k,case["weights"],case["terminal_count"])
        if safe:
            return {"action":choice,"target":target,"cost":total_cost,"queries":queries,"certificate":True}
        acts=list(actions(sources,terminals))
        if not acts:
            return {"action":target,"target":target,"cost":total_cost,"queries":queries,"certificate":False}
        vals=[]
        for kind,idx in acts:
            c=case["costs"][idx] if kind=="source" else case["costs"][case["source_count"]+idx]
            v=c+sum(p*rollout_value(s2,t2,horizon-1) for p,s2,t2 in branch_values(sources,terminals,kind,idx))
            vals.append((v,kind,idx))
        if case.get("_debug"):
            print("ORIG_STATE", sources, terminals, "known", k, "options", vals)
        _,kind,idx=min(vals,key=lambda z:(z[0],z[1],z[2]))
        if kind=="source":
            sources[idx]=case["new"][idx]; total_cost += case["costs"][idx]
        else:
            terminals[idx]=case["new_labels"][idx]; total_cost += case["costs"][case["source_count"]+idx]
        queries += 1


def run(seed:int,cases_per_family:int=20)->dict:
    rng=np.random.default_rng(seed); records=[]
    methods=tuple(METHODS_H1)+("topology_voi_h2",)
    for family in FAMILY_SPECS:
      for drift in (0.0,0.1,0.4,1.0):
       for regime in ("uniform","heterogeneous"):
        for edit_count in (1,2):
         for _ in range(cases_per_family):
          case=make_case(rng,family,drift,regime,edit_count)
          for method in methods:
           if method=="topology_voi_exact_dp": result=run_dp(case)
           elif method=="topology_voi_h1":
            from strict_graph_voi_cpu import run_h1
            result=run_h1(case)
           elif method=="topology_voi_h2": result=run_h2(case)
           else: result=run_static(case,method)
           result.update({"family":family,"drift":drift,"regime":regime,"edit_count":edit_count,"method":method}); records.append(result)
    summary={}
    for m in methods:
      rows=[r for r in records if r["method"]==m]
      summary[m]={"n":len(rows),"action_exact":float(np.mean([r["action"]==r["target"] for r in rows])),"mean_cost":float(np.mean([r["cost"] for r in rows])),"mean_queries":float(np.mean([r["queries"] for r in rows])),"certificate_rate":float(np.mean([r["certificate"] for r in rows]))}
    return {"schema":"mab2_p3_cycle005_topology_voi_h2_v1","seed":seed,"cases_per_family":cases_per_family,"records":records,"summary":summary}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=20261124); p.add_argument('--cases',type=int,default=20); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); x=run(a.seed,a.cases); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(x,indent=2),encoding='utf-8'); print(json.dumps(x['summary'],indent=2))
if __name__=='__main__': main()

