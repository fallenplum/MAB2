"""Fixed Formal8 second-task gates. No fitting, bootstrap or retrospective graph selection."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np

def correlation(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None

def summarize(em,f1,tokens,conditions,seeds):
    assert em.shape==f1.shape==tokens.shape==(96,2,11)
    ix={v:i for i,v in enumerate(conditions)};u=em-5e-6*tokens
    c0=np.stack([em[:,:,ix['G0']]-em[:,:,ix['G0_MINUS_E'+e]] for e in ['U','V']],axis=2)
    c1=np.stack([em[:,:,ix['G1']]-em[:,:,ix['G1_MINUS_E'+e]] for e in ['U','V']],axis=2)
    shift=c1-c0;changed=shift!=0;nchanged=int(changed.any(axis=(1,2)).sum())
    rc=correlation(c1[:,0,:],c1[:,1,:]);rs=correlation(shift[:,0,:],shift[:,1,:])
    differences={b:em[:,:,ix['G1']]-em[:,:,ix[b]] for b in ['SINGLE_Q','FLAT_ANNOTATION']}
    gain=u[:,:,ix['G1']]-u[:,:,ix['SINGLE_Q']]
    gates=dict(complete_matrix=True,fixed_G1_quality=all(x.mean()>=-.01 and np.all(x.mean(axis=0)>=-.01) for x in differences.values()),credit_shift_support=bool(changed.mean()>=.03 and np.all(changed.mean(axis=(0,2))>=.02) and nchanged>=20),cross_seed_reliability=bool(rc is not None and rs is not None and rc>=.30 and rs>=.20),service_economic_signal=bool(gain.mean()>=.005 and np.all(gain.mean(axis=0)>0)))
    table={k:dict(EM=float(em[:,:,i].mean()),F1=float(f1[:,:,i].mean()),service_tokens=float(tokens[:,:,i].mean()),service_utility=float(u[:,:,i].mean()),EM_by_seed=em[:,:,i].mean(axis=0).tolist(),utility_by_seed=u[:,:,i].mean(axis=0).tolist()) for k,i in ix.items()}
    nominee=em[:,:,ix['G1']]
    paired={k:dict(EM_delta=float((nominee-em[:,:,i]).mean()),corrections=int(((nominee==1)&(em[:,:,i]==0)).sum()),destructions=int(((nominee==0)&(em[:,:,i]==1)).sum())) for k,i in ix.items()}
    return dict(gates=gates,admission=all(gates.values()),conditions=table,paired_G1_comparisons=paired,current_credit_seed_correlation=rc,credit_shift_seed_correlation=rs,credit_shift_fraction=float(changed.mean()),credit_shift_by_seed=changed.mean(axis=(0,2)).tolist(),credit_shift_by_edge=changed.mean(axis=(0,1)).tolist(),distinct_shift_questions=nchanged,service_utility_gain_vs_single_Q=float(gain.mean()),service_utility_gain_by_seed=gain.mean(axis=0).tolist(),seeds=seeds,independent_question_count=96,question_seed_count=192,confidence_intervals=False,bootstrap_runs=0,claim='Foundation applicability only; no active feedback efficacy, independent replication or novelty established',cost_limitation='Service utility excludes development and feedback acquisition; physical collection costs are reported separately',evidence_ID_equality_certificates_used=False)

def main():
    a=argparse.ArgumentParser();a.add_argument('--approval',required=True);a.add_argument('--approval-sha',required=True);a.add_argument('--output',required=True);x=a.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(x.approval)==x.approval_sha;approval=json.loads(Path(x.approval).read_bytes());assert approval['decision']=='APPROVED_FIXED_ANNOTATION_2WIKI_TRAIN_SCORING'
    for v in approval['pins'].values():assert sha(v['path'])==v['sha256']
    for path in reversed(approval['import_paths']):sys.path.insert(0,path)
    from alias_metrics import answer_metrics
    from annotation_operator import SCHEDULE
    arc=Path(approval['archive_root']);manifest=json.loads((arc/'ARCHIVE_MANIFEST.json').read_bytes())
    for name,h in manifest.items():
        f=(arc/name).resolve();assert f.is_relative_to(arc.resolve()) and sha(f)==h
    vault=json.loads(Path(approval['pins']['vault']['path']).read_bytes());replay=json.loads(Path(approval['pins']['replay']['path']).read_bytes());cost=json.loads(Path(approval['pins']['cost']['path']).read_bytes())
    assert replay['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY' and replay['completed_units']==96 and cost['actual_cost_closed']
    seeds=[26111671,26111672];em=np.zeros((96,2,11));f1=em.copy();tokens=em.copy();seen=set()
    for unit in sorted(arc.glob('worker_*/unit_*')):
        inp=json.loads((unit/'INPUT.json').read_bytes());mat=json.loads((unit/'MATRIX.json').read_bytes());i=inp['index'];assert i not in seen;seen.add(i)
        assert mat['status']=='COMPLETE_UNSCORED' and len(mat['conditions'])==22
        rows={(c['execution_seed'],c['condition']):c for c in mat['conditions']};assert len(rows)==22;gold=vault[inp['payload']['id']]
        for j,seed in enumerate(seeds):
            for k,name in enumerate(SCHEDULE):
                row=rows[seed,name];sc=answer_metrics(row['answer'],gold);em[i,j,k]=sc['em'];f1[i,j,k]=sc['f1'];tokens[i,j,k]=sum(c['input_tokens']+c['output_tokens'] for c in row['logical_service_cost_by_model'].values())
    assert seen==set(range(96));result=summarize(em,f1,tokens,SCHEDULE,seeds)
    inputs=[json.loads(v) for v in Path(approval['pins']['questions']['path']).read_bytes().splitlines()];result['type_breakdown']={}
    for group in sorted(set(v['type'] for v in inputs)):
        mask=np.array([v['type']==group for v in inputs]);result['type_breakdown'][group]=dict(questions=int(mask.sum()),conditions={name:dict(EM=float(em[mask,:,k].mean()),F1=float(f1[mask,:,k].mean()),service_tokens=float(tokens[mask,:,k].mean())) for k,name in enumerate(SCHEDULE)})
    result.update(task='2WikiMultihopQA corrected train, answer-only alias-aware EM/F1',archive_manifest_sha256=sha(arc/'ARCHIVE_MANIFEST.json'),approval_sha256=x.approval_sha,source_sha256=sha(__file__),complete_trace_replay=True,physical_GPU_hours=cost['actual_GPU_hours'],baseline_routes={'INDEPENDENT3_Q':{'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},'VOTE_Q3':{'M':'qwen','U':'qwen','V':'qwen'}})
    out=Path(x.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'PRIVATE_ARRAYS.npz',em=em,f1=f1,tokens=tokens);(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(admission=result['admission'],gates=result['gates'])))

if __name__=='__main__':main()
