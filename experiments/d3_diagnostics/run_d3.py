"""Run the locked final CPU screen on accepted, exposed D2 arrays."""
from __future__ import annotations
import time
WALL_START=time.time()
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
import threadpoolctl
from drift_diagnostics import (fit_selector,predict_credits,drift_scores,select_top2,
    credit_design,evaluate_selector,first_order_degeneracy,spending_gate)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def checked(path,expected):
    raw=path.read_bytes()
    if sha(raw)!=expected:raise ValueError("Input pin mismatch: "+path.name)
    return raw

def main(root,repo):
    root,repo=root.resolve(),repo.resolve()
    proto_path=repo/"protocol/autonomous_20261113/D3_PROTOCOL.json"
    proto_raw=proto_path.read_bytes();proto=json.loads(proto_raw)
    assert proto["schema"]=="mab2_d3_development_drift_protocol_v1"
    git=["git","-c","safe.directory="+repo.as_posix(),"-C",str(repo)]
    git_start=time.perf_counter()
    commit=subprocess.check_output(git+["rev-parse","HEAD"],text=True,timeout=10).strip()
    assert subprocess.check_output(git+["show","HEAD:protocol/autonomous_20261113/D3_PROTOCOL.json"],timeout=10)==proto_raw
    git_upper=(time.perf_counter()-git_start)*(os.cpu_count() or 1)
    base=root/"results/autonomous_20261113/d3_20261113"
    old_cpu=0.0
    if base.exists():
        for attempt in base.iterdir():
            p=attempt/("D3_RESULTS.json" if (attempt/"COMPLETE.json").exists() else "FAILED.json")
            if not p.exists():raise ValueError("Prior attempt cost/state unresolved")
            previous=json.loads(p.read_bytes());old_cpu+=previous["process_cpu_seconds"]+previous["git_cpu_upper_bound_seconds"]
    out=base/uuid.uuid4().hex[:12];(out/"private").mkdir(parents=True)
    source_paths=list(Path(__file__).parent.glob("*.py"))+[repo/"experiments/d2_diagnostics/diagnostics.py"]
    source_hash={p.relative_to(repo).as_posix():sha(p.read_bytes()) for p in source_paths}
    dump(out/"STARTED.json",{"time_unix":WALL_START,"commit":commit,"protocol_sha256":sha(proto_raw),
        "source_sha256":source_hash,"numpy":np.__version__,"scipy":scipy.__version__,
        "threadpools":threadpoolctl.threadpool_info(),"new_GPU_calls":0,"role":"exposed_development"})
    print(json.dumps({"state":"STARTED","path":str(out)},ensure_ascii=False),flush=True)
    try:
        pin=proto["input_bindings"];parent=root/"results/autonomous_20261113/d2_20261113"/pin["accepted_D2_attempt"]
        names={parent/"COMPLETE.json":pin["complete_sha256"],parent/"D2_RESULTS.json":pin["results_sha256"],
               parent/"private/inputs.npz":pin["private_inputs_npz_sha256"],
               parent/"private/parameters.npz":pin["private_parameters_npz_sha256"],
               root/"artifacts/private/autonomous_20261113/D2_INDEPENDENT_REAL_REPLAY.json":pin["independent_replay_sha256"]}
        raw={p:checked(p,h) for p,h in names.items()}
        assert json.loads(raw[parent/"COMPLETE.json"])["results_sha256"]==pin["results_sha256"]
        assert json.loads(raw[root/"artifacts/private/autonomous_20261113/D2_INDEPENDENT_REAL_REPLAY.json"])["status"]=="PASS_WITH_EXPLORATORY_LIMITATIONS"
        dump(out/"DERIVED_INPUT_ACCESS.json",{"event":"PINNED_ACCEPTED_D2_ARRAYS_BEFORE_DECODE","time_unix":time.time(),
            "original_gold_or_questions_read":False,"sealed_test_read":False,"role":"exposed_development"})
        with np.load(io.BytesIO(raw[parent/"private/inputs.npz"]),allow_pickle=False) as npz:a={k:npz[k] for k in npz.files}
        with np.load(io.BytesIO(raw[parent/"private/parameters.npz"]),allow_pickle=False) as npz:old_beta=npz["C2_R3"].copy()
        split=a["partition"];tr=split=="train";tu=split=="tune";ev=split=="test"
        assert len(set(a["question_ids"]))==400 and [int(m.sum()) for m in [tr,tu,ev]]==[200,80,120]
        assert np.all(tr.astype(int)+tu.astype(int)+ev.astype(int)==1)
        qrow=a["row_question"]
        assert np.array_equal(qrow[a["left"]],np.repeat(np.arange(400),24))
        assert np.array_equal(qrow[a["right"]],np.repeat(np.arange(400),24))
        target=(a["quality_by_seed"][a["right"]]-a["quality_by_seed"][a["left"]]).reshape(400,2,12,5)
        bits=a["bits"]
        models={o:fit_selector(bits[tr],target[tr],bits[tu],target[tu],order=o) for o in [1,2]}
        credits={f"graph_only_order{o}":predict_credits(m.beta,bits,order=o) for o,m in models.items()}
        assert old_beta.shape==(711,) and a["phi"].shape==(400,9)
        credits["fixed_old_C2_R3"]=np.einsum("qi,qaef,if->qae",a["phi"],credit_design(bits,2),old_beta.reshape(9,79))
        b0=target[tr].mean(axis=(0,1,3));credits["B0"]=np.broadcast_to(b0,(400,2,12)).copy()
        scores={n:drift_scores(v) for n,v in credits.items()};indices={n:select_top2(v) for n,v in scores.items()}
        assert np.all(scores["B0"]==0) and np.all(scores["graph_only_order1"]==0)
        degeneracy={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in first_order_degeneracy(models[1],target[tr]).items()}
        assert degeneracy["coefficient_identity_max_abs"]<1e-10
        np.savez_compressed(out/"private/FROZEN_SELECTIONS.npz",**{n+"_indices":v for n,v in indices.items()},
                            **{n+"_scores":v for n,v in scores.items()},**{f"beta_order{o}":m.beta for o,m in models.items()})
        dump(out/"NOMINATION_FROZEN.json",{"time_unix":time.time(),"candidate":"D3_graph_only_order2_R3","alpha":models[2].alpha,
             "parameter_and_selection_sha256":sha((out/"private/FROZEN_SELECTIONS.npz").read_bytes()),
             "no_fallback":True,"exposed_scoring_started":False,"protocol_sha256":sha(proto_raw)})
        summaries={};private_arrays={}
        for part,mask in [("tune",tu),("exposed",ev)]:
            summaries[part]={}
            for name,score in scores.items():
                scored=evaluate_selector(score[mask],target[mask])
                assert np.array_equal(scored["selected_indices"],indices[name][mask])
                summaries[part][name]=scored["summary"]
                for key,value in scored["per_question"].items():private_arrays[part+"_"+name+"_"+key]=value
        gate=spending_gate(summaries["exposed"]["graph_only_order2"],integrity_ok=True,nomination_frozen=True,candidate_name="D3_graph_only_order2_R3")
        np.savez_compressed(out/"private/SCORED_ARRAYS.npz",**private_arrays)
        for path,h in names.items():checked(path,h)
        assert sha(proto_path.read_bytes())==sha(proto_raw)
        assert source_hash=={p.relative_to(repo).as_posix():sha(p.read_bytes()) for p in source_paths}
        assert old_cpu+time.process_time()+git_upper<proto["authorization"]["max_D3_process_cpu_seconds"]
        results={"schema":"mab2_d3_results_v1","status":"DEVELOPMENT_SCREEN_COMPLETE","protocol_sha256":sha(proto_raw),
            "source_commit":commit,"source_sha256":source_hash,
            "models":{str(o):{"alpha":m.alpha,"alpha_trials":m.alpha_trials,"diagnostics":m.diagnostics} for o,m in models.items()},
            "first_order_degeneracy":degeneracy,"drift_results":summaries,"spending_gate":gate,
            "changed_context_questions":int(np.any(bits[:,0]!=bits[:,1],axis=1).sum()),
            "new_GPU_calls":0,"independent_confirmation":False,"future_service_or_full_budget_benefit":False,
            "process_cpu_seconds":time.process_time(),"git_cpu_upper_bound_seconds":git_upper,"git_cpu_seconds":None,
            "previous_attempts_cpu_upper_bound_seconds":old_cpu,"wall_seconds":time.time()-WALL_START}
        dump(out/"D3_RESULTS.json",results)
        dump(out/"COMPLETE.json",{"status":"DEVELOPMENT_SCREEN_COMPLETE","results_sha256":sha((out/"D3_RESULTS.json").read_bytes()),
            "nomination_sha256":sha((out/"NOMINATION_FROZEN.json").read_bytes()),"selection_sha256":sha((out/"private/FROZEN_SELECTIONS.npz").read_bytes())})
        print(json.dumps({"state":"DEVELOPMENT_SCREEN_COMPLETE","path":str(out),"gate":gate},ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(out/"FAILED.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),
            "process_cpu_seconds":time.process_time(),"git_cpu_upper_bound_seconds":git_upper,"new_GPU_calls":0})
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--repo",type=Path,required=True)
    a=p.parse_args()
    with threadpoolctl.threadpool_limits(4):main(a.root,a.repo)
