"""Bounded exploratory D2. Reads only accepted, already-exposed G2 evidence."""
from __future__ import annotations
import time
PROCESS_START_WALL = time.time()
import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
import numpy as np
import scipy
import threadpoolctl
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import diagnostics as diag

OLD = "results/g2/g2_v1_20261104"
PINS = {
    "execution_tools/g2/descriptive_appendix.py": "3c6eb81fd21cd320b218c814bc4e369cbaf1fcd854abf328c4e84c51260c14d1",
    "experiments/g2/predictor.py": "d6d93dc938ce7cff682a5457c59b779bed5668999619e5afb614773f756e8e03",
    "artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json": "2da9c7ac2f1a75901f56dc25119cfc900d4d5b65034976a62db4bb25b7f4aff6",
    OLD + "/frozen_plan.json": "90cbbe0e22ae1cad177e30706759a6e04341ca1a956f8faf62f80497360cdeb2",
}
LABEL_PINS = {"train": "0e1037ec2e29ef5983e2ec19e3f39636762c8d48cd7ad2b991e00854b1d6e139",
              "tune": "beba0bb851404b127e6ce393690df7ccc6589b60d78d592cceeea52ca1ae29d0",
              "test": "641e3864993f6e29feedc55ff44fde1b89780a7290c74b5083033a8e2f8590b5"}

def sha(raw): return hashlib.sha256(raw).hexdigest()

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def checked(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected: raise ValueError("Pinned input hash mismatch: " + path.name)
    return raw

def read_inputs(root, out):
    for name, pin in PINS.items(): checked(root/name, pin)
    helper = module(root/"execution_tools/g2/descriptive_appendix.py", "d2_original_helper")
    w = helper.Workspace(root)
    c = helper.source_context(w, OLD+"/frozen_plan.json")
    helper.prelabel_gate(w, c, "artifacts/audit/G2_INDEPENDENT_ACCEPTANCE.json")
    plan = c["plan"]
    feature_raw = checked(root/plan["features"]["path"], plan["features"]["sha256"])
    with np.load(io.BytesIO(feature_raw), allow_pickle=False) as f:
        phi = dict(zip(f["question_ids"].tolist(), f["phi"]))
    labels = {}
    for part in ["train", "tune", "test"]:
        path = root/plan["label_release"][part]
        with path.open("rb") as f:
            dump(out/f"LABEL_ACCESS_{part}.json", {"time_unix":time.time(),"event":"OPENED_BEFORE_READ",
                "role":"already_exposed_G2_development", "expected_sha256":LABEL_PINS[part]})
            raw = f.read()
        assert sha(raw) == LABEL_PINS[part]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert {r["id"] for r in rows} == {q for q,p in plan["partitions"].items() if p==part}
        assert len(rows) == len({r["id"] for r in rows})
        labels.update({r["id"]:r for r in rows})
    scorer = helper.original_scorer(w,c)
    quality = {}
    for line in c["observation_raw"].splitlines():
        row = json.loads(line)
        assert row["quality"] is None and row["labels_read"] is False and row["plan_sha256"] == c["plan_sha256"]
        q, seed, mask = row["question_id"], row["seed"], tuple(row["graph_mask"])
        assert row["partition"] == plan["partitions"][q] and seed in [2701,2702,2703,2704,2705]
        key = (q,mask,seed)
        assert key not in quality
        value = scorer(row["service_output"],labels[q],task="gsm8k")
        assert value in (0,1)
        quality[key] = value
    expected = {(u["question_id"],tuple(r["mask"]),u["seed"]) for u in plan["units"] for r in u["mask_requests"]}
    assert set(quality) == expected and len(quality)==51970
    qids = sorted(plan["partitions"])
    assert len(qids)==400 and set(phi)==set(qids)
    rows, y, row_q, masks, bits, left, right = [],[],[],[],[],[],[]
    index = {}
    for qi,q in enumerate(qids):
        for mask in sorted({tuple(r["mask"]) for r in plan["designs"][q]["mask_requests"]}):
            index[(q,mask)] = len(rows)
            rows.append((q,mask)); row_q.append(qi); masks.append(mask)
            y.append([quality[(q,mask,s)] for s in [2701,2702,2703,2704,2705]])
    for q in qids:
        anchors = {r["anchor_id"]:r["mask"] for r in plan["designs"][q]["anchors"]}
        bits.append([anchors[a] for a in sorted(anchors)])
        contrasts = sorted(plan["designs"][q]["contrasts"],key=lambda r:(r["anchor_id"],r["edge"]))
        assert len(contrasts)==24
        assert {(r["anchor_id"],r["edge"]) for r in contrasts} == {(a,e) for a in anchors for e in range(12)}
        for r in contrasts:
            m0,m1=tuple(plan["masks"][r["mask0_id"]]),tuple(plan["masks"][r["mask1_id"]])
            assert m0[r["edge"]]==0 and m1[r["edge"]]==1
            assert all(m0[e]==m1[e] for e in range(12) if e!=r["edge"])
            assert tuple(anchors[r["anchor_id"]]) in (m0,m1)
            left.append(index[(q,m0)])
            right.append(index[(q,m1)])
    arrays = {"phi":np.stack([phi[q] for q in qids]),"quality_by_seed":np.array(y,dtype=float),
              "row_question":np.array(row_q),"masks":np.array(masks),"bits":np.array(bits),
              "left":np.array(left),"right":np.array(right),"question_ids":np.array(qids),
              "partition":np.array([plan["partitions"][q] for q in qids])}
    np.savez_compressed(out/"private/inputs.npz",**arrays)
    bindings = {**PINS, **c["anchors"], plan["features"]["path"]:plan["features"]["sha256"]}
    dump(out/"INPUT_BINDINGS.json", {"sha256":bindings,"labels_sha256":LABEL_PINS,"private_arrays_sha256":sha((out/"private/inputs.npz").read_bytes()),"raw_scored":len(quality),"question_count":400})
    params = json.loads((root/(OLD+"/parameters.json")).read_bytes())
    return arrays, params, c["predictions"], bindings

def bootstrap(values):
    rng = np.random.default_rng(59173)
    sampled = []
    for start in range(0,10000,200):
        idx = rng.integers(0,len(values),size=(min(200,10000-start),len(values)))
        sampled.extend(values[idx].mean(axis=1))
    return {"estimate":float(np.mean(values)),"interval95":np.quantile(sampled,[.025,.975]).tolist(),
            "question_count":len(values),"samples":10000,"seed":59173,"exploratory":True,"multiplicity":"unadjusted"}

def action_values(pred, targets, bits):
    gains = (1-2*bits)*pred
    a = np.argmax(gains,axis=-1)
    best = np.take_along_axis(gains,a[...,None],axis=-1)[...,0]
    a = np.where(best>0,a,12)
    menu = np.concatenate(((1-2*bits)*targets.mean(axis=-1),np.zeros((*bits.shape[:2],1))),axis=-1)
    chosen = np.take_along_axis(menu,a[...,None],axis=-1)[...,0]
    return chosen.mean(axis=1),a,menu.mean(axis=-1).mean(axis=1)

def metrics(pred, targets, bits, with_ci=False):
    gain,actions,uniform = action_values(pred,targets,bits)
    mean = targets.mean(axis=-1)
    errors = pred[...,None]-targets
    risk = (errors.sum(axis=-1)**2-(errors**2).sum(axis=-1))/20
    cors = []
    for x,y in zip(pred.reshape(-1,12),mean.reshape(-1,12)):
        if np.ptp(x)>0 and np.ptp(y)>0: cors.append(float(spearmanr(x,y).statistic))
    positive = mean!=0
    selected_bits=np.take_along_axis(bits,np.minimum(actions,11)[...,None],axis=-1)[...,0]
    result = {"question_count":len(pred),"mean_action_gain":float(gain.mean()),"mean_credit_risk":float(risk.mean()),
              "mean_credit_mse":float(((pred-mean)**2).mean()),"mean_prediction":float(pred.mean()),
              "positive_prediction_fraction":float((pred>0).mean()),"prediction_rms":float(np.sqrt((pred**2).mean())),
              "stay_count":int((actions==12).sum()),"addition_count":int(((actions!=12)&(selected_bits==0)).sum()),
              "deletion_count":int(((actions!=12)&(selected_bits==1)).sum()),
              "nonzero_observed_mean":int(positive.sum()),"nonzero_sign_agreements":int(((np.sign(pred)==np.sign(mean))&positive).sum()),
              "rank_defined":len(cors),"rank_undefined":len(pred)*2-len(cors),
              "mean_defined_spearman":float(np.mean(cors)) if cors else None}
    if with_ci:
        result["gain_vs_stay"]=bootstrap(gain)
        result["gain_vs_uniform13"]=bootstrap(gain-uniform)
    return result,gain

def choose_nominee(results,protocol):
    eligible=protocol["candidate_nomination"]["eligible"]
    best=max(results[n]["tune"]["mean_action_gain"] for n in eligible)
    names=[n for n in eligible if results[n]["tune"]["mean_action_gain"]>=best-1e-12]
    mse=min(results[n]["tune"]["mean_credit_mse"] for n in names)
    names=[n for n in names if results[n]["tune"]["mean_credit_mse"]<=mse+1e-12]
    return sorted(names)[0]

def noise_report(targets,bits):
    mean=targets.mean(axis=-1)
    cross=(targets.sum(axis=-1)**2-(targets**2).sum(axis=-1))/20
    per_seed_variance=targets.var(axis=-1,ddof=1)
    menu=np.concatenate(((1-2*bits)[...,None]*targets,np.zeros((*bits.shape[:2],1,5))),axis=2)
    naive=menu.mean(axis=-1).max(axis=-1).mean(axis=1)
    folds=[]
    for s in range(5):
        train=np.delete(menu,s,axis=-1).mean(axis=-1)
        idx=np.argmax(train[...,:12],axis=-1)
        best=np.take_along_axis(train,idx[...,None],axis=-1)[...,0]
        idx=np.where(best>0,idx,12)
        held=np.take_along_axis(menu[...,s],idx[...,None],axis=-1)[...,0]
        folds.append(held.mean(axis=1))
    honest=np.stack(folds,axis=-1).mean(axis=-1)
    return {"coordinates":mean.size,"all_seed_zero":int((targets==0).all(axis=-1).sum()),
            "mean_zero":int((mean==0).sum()),"both_positive_negative_seed_differences":int(((targets>0).any(axis=-1)&(targets<0).any(axis=-1)).sum()),
            "mean_cross_seed_credit_squared":float(cross.mean()),"mean_seed_variance":float(per_seed_variance.mean()),
            "observed_mean_squared":float((mean**2).mean()),"noise_identity_max_abs":float(np.max(np.abs(mean**2-per_seed_variance/5-cross))),
            "naive_observed_menu_oracle":bootstrap(naive),"leave_one_seed_out_selection_gain":bootstrap(honest),
            "five_folds_independent":False,"true_oracle_claimed":False}

def execute(args):
    started,cpu=PROCESS_START_WALL,0.0
    repo=args.repo.resolve();root=args.root.resolve()
    protocol_path=repo/"protocol/autonomous_20261113/D2_PROTOCOL.json"
    protocol_raw=protocol_path.read_bytes();protocol=json.loads(protocol_raw)
    assert protocol["schema"]=="mab2_d2_exploratory_protocol_v1_1"
    git=["git","-c","safe.directory="+repo.as_posix(),"-C",str(repo)]
    git_wall_start=time.perf_counter()
    commit=subprocess.check_output(git+["rev-parse","HEAD"],text=True,timeout=10).strip()
    assert subprocess.check_output(git+["show","HEAD:protocol/autonomous_20261113/D2_PROTOCOL.json"],timeout=10)==protocol_raw
    git_wall_seconds=time.perf_counter()-git_wall_start
    git_cpu_upper_bound=git_wall_seconds*(os.cpu_count() or 1)
    out=root/"results/autonomous_20261113/d2_20261113"/uuid.uuid4().hex[:12]
    previous_cpu=0.0
    if out.parent.exists():
        for prior in out.parent.iterdir():
            if not prior.is_dir(): continue
            cost=prior/"FAILED.json"
            if (prior/"COMPLETE.json").exists(): cost=prior/"D2_RESULTS.json"
            if not cost.exists(): raise ValueError("Previous diagnostic attempt has unknown cost or unfinished state")
            prior_cost=json.loads(cost.read_bytes())
            previous_cpu+=prior_cost["process_cpu_seconds"]+prior_cost.get("git_cpu_upper_bound_seconds",0)
    assert previous_cpu+time.process_time()+git_cpu_upper_bound<4*3600
    (out/"private").mkdir(parents=True)
    sources={p.name:sha(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
    dump(out/"STARTED.json", {"time_unix":started,"protocol_sha256":sha(protocol_raw),"source_commit":commit,
                             "source_sha256":sources,"python":platform.python_version(),
                             "versions":{"numpy":np.__version__,"scipy":scipy.__version__,"threadpoolctl":threadpoolctl.__version__},
                             "threadpools":threadpoolctl.threadpool_info(),"scope":"posthoc_development","new_gpu_calls":0})
    print(json.dumps({"state":"STARTED","attempt":str(out)},ensure_ascii=False),flush=True)
    try:
        a,old,oldpred,bindings=read_inputs(root,out)
        qn=len(a["partition"]);rowq=a["row_question"];y=a["quality_by_seed"].mean(axis=-1)
        target=(a["quality_by_seed"][a["right"]]-a["quality_by_seed"][a["left"]]).reshape(qn,2,12,5)
        cq=np.repeat(np.arange(qn),24)
        tr=a["partition"]=="train";tu=a["partition"]=="tune";ev=a["partition"]=="test"
        assert [int(s.sum()) for s in [tr,tu,ev]]==[200,80,120]
        assert np.all(tr.astype(int)+tu.astype(int)+ev.astype(int)==1)
        assert set(a["question_ids"][tr])==set(old["train_question_ids"])
        assert set(a["question_ids"][tu])==set(old["tune_question_ids"])
        trainrows=tr[rowq]; trainc=tr[cq];tunec=tu[cq]
        prediction={"ZERO":np.zeros((qn,2,12))}
        b0=target[tr].mean(axis=(0,1,3))
        assert np.allclose(b0,old["B0"],rtol=0,atol=1e-14)
        prediction["B0"]=np.broadcast_to(b0,(qn,2,12)).copy()
        results={};params={};constant={};replay={};geometries=[]
        for qi in range(qn):
            inds=np.flatnonzero(rowq==qi);pos={int(i):j for j,i in enumerate(inds)}
            D=np.zeros((24,len(inds)))
            for k in range(24):
                at=qi*24+k;D[k,pos[int(a["left"][at])]]=-1;D[k,pos[int(a["right"][at])]]=1
            g=diag.contrast_graph_diagnostics(D)
            geometries.append(g)
        for order in [1,2]:
            graph=np.column_stack([np.ones(len(y)),a["masks"]])
            if order==2:
                graph=np.column_stack([graph,*[a["masks"][:,i]*a["masks"][:,j] for i,j in itertools.combinations(range(12),2)]])
            X=(a["phi"][rowq,:,None]*graph[:,None,:]).reshape(len(y),-1)
            DX=X[a["right"]]-X[a["left"]]
            for variant in ["R0","R1","R1b","R2","R3"]:
                assert previous_cpu+time.process_time()-cpu+git_cpu_upper_bound<4*3600
                name=f"C{order}_{variant}"
                p=diag.prepare_problem(X[trainrows],y[trainrows],rowq[trainrows],variant,d=graph.shape[1],
                    difference_X=DX[trainc],difference_y=target[tr].mean(axis=-1).ravel(),difference_question_ids=cq[trainc])
                fits=diag.fit_path(p)
                select=diag.select_alpha(p,DX[tunec],target[tu].mean(axis=-1).ravel(),cq[tunec],fits=fits)
                beta=select.best_fit.beta
                prediction[name]=(DX@beta).reshape(qn,2,12)
                params[name]=beta
                tune,_=metrics(prediction[name][tu],target[tu],a["bits"][tu])
                historical_alpha=old["models"][f"C{order}"]["alpha"]
                fixed=next(f for f in fits if f.alpha==historical_alpha)
                fixedpred=(DX@fixed.beta).reshape(qn,2,12)
                params[name+"_fixed_alpha"]=fixed.beta
                results[name]={"order":order,"variant":variant,"alpha":select.best_alpha,"alpha_trials":select.all_trials,
                               "tune":tune,"fixed_historical_alpha":historical_alpha,
                               "selected_fit_diagnostics":select.best_fit.diagnostics,"fixed_fit_diagnostics":fixed.diagnostics,
                               "fixed_alpha_tune":metrics(fixedpred[tu],target[tu],a["bits"][tu])[0]}
                # Record fixed-alpha evaluation only after nomination freeze below.
                prediction[name+"_fixed_alpha"]=fixedpred
                if variant=="R0":
                    assert select.best_alpha==historical_alpha
                    delta=float(np.max(np.abs(beta-np.array(old["models"][f"C{order}"]["coefficients"])) ))
                    assert delta<1e-7
                    qpos={q:i for i,q in enumerate(a["question_ids"].tolist())}
                    family={a:i for i,a in enumerate(sorted({p["anchor_id"] for p in oldpred}))}
                    diffs=[abs(prediction[name][qpos[r["question_id"]],family[r["anchor_id"]],r["edge"]]-r["predictions"][f"C{order}"]) for r in oldpred]
                    assert max(diffs)<1e-8
                    replay[name]={"alpha_match":True,"coefficient_max_abs":delta,"prediction_max_abs":max(diffs)}
                    dec=diag.constant_residual_decomposition(p,historical_alpha,credit_X=DX)
                    # Serialized numerical details only; per-coordinate arrays stay private.
                    constant[name]={k:v for k,v in dec.items() if not isinstance(v,np.ndarray)}
                    for k,v in dec.items():
                        if isinstance(v,np.ndarray): params[name+"_decomposition_"+k]=v
                print(json.dumps({"state":"MODEL_FIT","model":name,"alpha":select.best_alpha,"cpu_seconds":time.process_time()-cpu}),flush=True)
        for name in ["ZERO","B0"]:
            results[name]={"tune":metrics(prediction[name][tu],target[tu],a["bits"][tu])[0]}
        nominee=choose_nominee(results,protocol)
        np.savez_compressed(out/"private/parameters.npz",**params)
        dump(out/"NOMINATION_FROZEN.json", {"time_unix":time.time(),"nominee":nominee,"tune":results[nominee]["tune"],
            "parameter_sha256":sha((out/"private/parameters.npz").read_bytes()),"protocol_sha256":sha(protocol_raw),
            "fresh_independence_claimed":False,"fallback_allowed":False})
        gains={}
        for name in list(results):
            results[name]["exposed_evaluation"],gains[name]=metrics(prediction[name][ev],target[ev],a["bits"][ev],True)
            if name not in ["ZERO","B0"]:
                results[name]["fixed_alpha_exposed_evaluation"]=metrics(prediction[name+"_fixed_alpha"][ev],target[ev],a["bits"][ev],True)[0]
        for name in results: results[name]["exposed_evaluation"]["gain_vs_B0"]=bootstrap(gains[name]-gains["B0"])
        for name in constant:
            constant[name]["by_partition"]={}
            for part in ["train","tune","test"]:
                mask=a["partition"]==part
                components={k:params[name+"_decomposition_credit_"+k].reshape(qn,2,12)[mask] for k in ["direct","constant","residual"]}
                stats={k:{"mean":float(v.mean()),"rms":float(np.sqrt((v*v).mean())),"positive_fraction":float((v>0).mean())} for k,v in components.items()}
                stats["linear_prediction_reconstruction_max_abs"]=float(np.max(np.abs(components["direct"]-components["constant"]-components["residual"])))
                constant[name]["by_partition"][part]=stats
        uniform=action_values(prediction["ZERO"][ev],target[ev],a["bits"][ev])[2]
        qualify=(results[nominee]["tune"]["mean_action_gain"]>0 and gains[nominee].mean()>0
                 and gains[nominee].mean()>gains["B0"].mean() and gains[nominee].mean()>uniform.mean())
        np.savez_compressed(out/"private/predictions.npz",**prediction)
        for name,pin in bindings.items(): checked(root/name,pin)
        assert sha(protocol_path.read_bytes())==sha(protocol_raw)
        assert sources=={p.name:sha(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
        dump(out/"private/geometry_by_question.json",geometries)
        histogram={}
        for g in geometries:
            key=f"vertices={g['n_vertices']};rank={g['rank']};components={g['n_components']}"
            histogram[key]=histogram.get(key,0)+1
        summary={"schema":"mab2_d2_results_v1","status":"EXPLORATORY_COMPLETE","protocol_sha256":sha(protocol_raw),
                 "nominee":nominee,"nominee_passes_development_filter":bool(qualify),"fallback_used":False,
                 "models":results,"constant_response_control":constant,"baseline_replay":replay,
                 "geometry_support_histogram":histogram,"noise_by_partition":{part:noise_report(target[a["partition"]==part],a["bits"][a["partition"]==part]) for part in ["train","tune","test"]},
                 "new_gpu_calls":0,"independent_confirmation":False,"online_budget_claim":False,
                 "source_commit":commit,"source_sha256":sources,"started_unix":started,"finished_unix":time.time(),
                 "previous_attempts_cpu_seconds_upper_bound":previous_cpu,
                 "cpu_scope":"main-process CPU since process start includes threads; two Git subprocess CPU values unknown, bounded by command wall time times logical cores and charged conservatively",
                 "git_cpu_seconds":None,"git_wall_seconds":git_wall_seconds,"git_cpu_upper_bound_seconds":git_cpu_upper_bound,
                 "process_cpu_seconds":time.process_time()-cpu,"wall_seconds":time.time()-started}
        dump(out/"D2_RESULTS.json",summary)
        assert previous_cpu+time.process_time()-cpu+git_cpu_upper_bound<4*3600, "CPU diagnostic budget exceeded; no completion acceptance"
        dump(out/"COMPLETE.json",{"status":"EXPLORATORY_COMPLETE","results_sha256":sha((out/"D2_RESULTS.json").read_bytes()),
             "private_inputs_sha256":sha((out/"private/inputs.npz").read_bytes()),"nomination_sha256":sha((out/"NOMINATION_FROZEN.json").read_bytes())})
        print(json.dumps({"state":"EXPLORATORY_COMPLETE","attempt":str(out),"nominee":nominee,"development_filter":bool(qualify),"cpu_seconds":time.process_time()-cpu},ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(out/"FAILED.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc(),
             "time_unix":time.time(),"process_cpu_seconds":time.process_time()-cpu,
             "git_cpu_upper_bound_seconds":git_cpu_upper_bound,"new_gpu_calls":0})
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--repo",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=4): execute(args)
