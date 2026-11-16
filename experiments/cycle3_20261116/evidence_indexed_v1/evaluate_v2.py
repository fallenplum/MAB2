"""Frozen applicability gates; no fitting, CI, resampling or policy selection."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np

def summarize(em,f1,tokens,certified,certificate_violations,selection_valid,conditions,seeds):
    # q,seed,condition; certificates q,seed,time,edge; exact paired observations.
    assert em.shape==f1.shape==tokens.shape and em.ndim==3
    n,k,c=em.shape;assert n==96 and k==2 and c==11
    ix={v:i for i,v in enumerate(conditions)}
    c0=np.stack([em[:,:,ix['G0']]-em[:,:,ix['G0_MINUS_E'+e]] for e in ['U','V']],axis=2)
    c1=np.stack([em[:,:,ix['G1']]-em[:,:,ix['G1_MINUS_E'+e]] for e in ['U','V']],axis=2)
    shift=c0!=c1;changed_questions=int(np.any(shift,axis=(1,2)).sum())
    diffs={b:em[:,:,ix['G1']]-em[:,:,ix[b]] for b in ['SINGLE_Q','FLAT_EVIDENCE']}
    gates=dict(selector_valid=bool(np.mean(selection_valid)>=.98),
      fixed_G1_quality=all(float(x.mean())>=-.01 and all(float(v)>=-.01 for v in x.mean(axis=0)) for x in diffs.values()),
      credit_shift_support=bool(shift.mean()>=.03 and np.all(shift.mean(axis=(0,2))>=.02) and changed_questions>=20),
      certificate_coverage=bool(np.mean(certified)>=.10 and certificate_violations==0))
    table={}
    for name,i in ix.items():
        table[name]=dict(EM=float(em[:,:,i].mean()),F1=float(f1[:,:,i].mean()),service_tokens=float(tokens[:,:,i].mean()),
          service_utility=float((em[:,:,i]-5e-6*tokens[:,:,i]).mean()),
          EM_by_seed=em[:,:,i].mean(axis=0).tolist(),utility_by_seed=(em[:,:,i]-5e-6*tokens[:,:,i]).mean(axis=0).tolist())
    paired={}
    nominee=em[:,:,ix['G1']]
    for name,i in ix.items():
        paired[name]=dict(EM_delta=float((nominee-em[:,:,i]).mean()),corrections=int(((nominee==1)&(em[:,:,i]==0)).sum()),destructions=int(((nominee==0)&(em[:,:,i]==1)).sum()))
    return dict(gates=gates,admission=all(gates.values()),claim='Applicability only; no active-policy efficacy or independent replication established',
      conditions=table,paired_G1_comparisons=paired,credit_shift_fraction=float(shift.mean()),credit_shift_by_seed=shift.mean(axis=(0,2)).tolist(),
      credit_shift_by_edge=shift.mean(axis=(0,1)).tolist(),distinct_shift_questions=changed_questions,certificate_fraction=float(np.mean(certified)),
      certificate_violations=int(certificate_violations),selection_valid_rate=float(np.mean(selection_valid)),seeds=seeds,
      independent_question_count=n,question_seed_count=n*k,confidence_intervals=False,bootstrap_runs=0,
      cost_limitation='Service utilities exclude collection training cost and cannot establish net policy gain; full physical collection ledger reported separately')

def main():
    a=argparse.ArgumentParser();a.add_argument('--approval',required=True);a.add_argument('--approval-sha',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(args.approval)==args.approval_sha
    approval=json.loads(Path(args.approval).read_bytes());assert approval['decision']=='APPROVED_FIXED_EVIDENCE_TRAIN_SCORING'
    for item in approval['pins'].values():assert sha(item['path'])==item['sha256']
    for path in reversed(approval['import_paths']):sys.path.insert(0,path)
    from hotpot_support import answer_metrics
    from evidence_operator_v2 import SCHEDULE
    root=Path(approval['archive_root']);manifest=json.loads((root/'ARCHIVE_MANIFEST.json').read_bytes())
    for name,digest in manifest.items():
        path=(root/name).resolve();assert path.is_relative_to(root.resolve()) and sha(path)==digest
    vault=json.loads(Path(approval['pins']['vault']['path']).read_bytes())
    replay=json.loads(Path(approval['pins']['replay']['path']).read_bytes());assert replay['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY' and replay['completed_units']==96
    cost=json.loads(Path(approval['pins']['cost']['path']).read_bytes());assert cost['actual_cost_closed']
    seeds=[26111661,26111662];em=np.zeros((96,2,11));f1=em.copy();tokens=em.copy();cert=np.zeros((96,2,2,2),bool);valid=[];projections=[];rank_lengths=[];violations=0;seen=set()
    for unit in sorted(root.glob('worker_*/unit_*')):
        inp=json.loads((unit/'INPUT.json').read_bytes());mat=json.loads((unit/'MATRIX.json').read_bytes());i=inp['index'];assert i not in seen;seen.add(i)
        assert mat['status']=='COMPLETE_UNSCORED' and len(mat['conditions'])==22
        rows={(x['execution_seed'],x['condition']):x for x in mat['conditions']};assert len(rows)==22
        gold=vault[inp['payload']['id']]
        for j,seed in enumerate(seeds):
            for z,name in enumerate(SCHEDULE):
                row=rows[seed,name];score=answer_metrics(row['answer'],gold);em[i,j,z]=score['em'];f1[i,j,z]=score['f1']
                tokens[i,j,z]=sum(v['input_tokens']+v['output_tokens'] for v in row['logical_service_cost_by_model'].values())
                valid.extend(v['selection_valid'] for v in row['nodes'].values() if 'selection_valid' in v)
                projections.extend(v['budget_projection_applied'] for v in row['nodes'].values() if 'budget_projection_applied' in v)
                rank_lengths.extend(v['ranking_length'] for v in row['nodes'].values() if 'ranking_length' in v)
            for t in range(2):
                base=rows[seed,f'G{t}']
                for edge,e in enumerate(['U','V']):
                    ablated=rows[seed,f'G{t}_MINUS_E{e}'];same=base['terminal_evidence_ids']==ablated['terminal_evidence_ids'];cert[i,j,t,edge]=same
                    if same:violations+=int(base['nodes']['M']['request_fingerprint']!=ablated['nodes']['M']['request_fingerprint'] or base['answer']!=ablated['answer'])
    assert seen==set(range(96));result=summarize(em,f1,tokens,cert,violations,valid,SCHEDULE,seeds)
    inputs=[json.loads(s) for s in Path(approval['pins']['questions']['path']).read_bytes().splitlines()]
    result['budget_projection_fraction_logical']=float(np.mean(projections));result['mean_raw_ranking_length_logical']=float(np.mean(rank_lengths))
    result['baseline_routes']={'INDEPENDENT3_Q':{'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},'VOTE_Q3':{'M':'qwen','U':'qwen','V':'qwen'}}
    result['task']='HotpotQA train distractor';result['type_breakdown']={}
    for group in sorted(set(v['type'] for v in inputs)):
        mask=np.array([v['type']==group for v in inputs]);result['type_breakdown'][group]=dict(questions=int(mask.sum()),conditions={name:dict(EM=float(em[mask,:,z].mean()),F1=float(f1[mask,:,z].mean()),service_tokens=float(tokens[mask,:,z].mean())) for z,name in enumerate(SCHEDULE)})
    result.update(archive_manifest_sha256=sha(root/'ARCHIVE_MANIFEST.json'),approval_sha256=args.approval_sha,source_sha256=sha(__file__),complete_trace_replay=True,physical_GPU_hours=cost['actual_GPU_hours'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'PRIVATE_ARRAYS.npz',em=em,f1=f1,tokens=tokens,certificate=cert)
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'admission':result['admission'],'gates':result['gates']}))

if __name__=='__main__':main()
