"""Frozen exposed-development replay of simple abstention-aware note baselines."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np

LEXICON=('uncertain','insufficient','cannot determine',"can't determine",'cannot be determined','not enough','unknown','not provided','not mentioned','does not provide','do not provide','unable to determine')
POLICY_SEEDS=(261116131,261116132,261116133,261116134,261116135)

def abstains(text):
    normalized=' '.join(text.casefold().replace('\u2019',"'").split())
    return any(re.search(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',normalized) for phrase in LEXICON)

def build_masks(ids,seeds,notes):
    n=len(ids);assert len(set(ids))==n
    masks={'KEEP_ALL_NOTES':np.zeros((n,2),bool),'IGNORE_ALL_NOTES':np.ones((n,2),bool)}
    masks['DROP_ALL_IF_ANY_ABSTAINS']=np.array([[any(abstains(v['text']) for v in notes[i][j]) for j in range(2)] for i in range(n)])
    masks['DROP_ALL_IF_R_ABSTAINS']=np.array([[any(v['source']=='R' and abstains(v['text']) for v in notes[i][j]) for j in range(2)] for i in range(n)])
    for rule in ['ANY','R']:
        reference=masks[f'DROP_ALL_IF_{rule}_ABSTAINS']
        for policy_seed in POLICY_SEEDS:
            mask=np.zeros((n,2),bool)
            for j,seed in enumerate(seeds):
                order=sorted(range(n),key=lambda i:hashlib.sha256(json.dumps(['D13',policy_seed,ids[i],seed],separators=(',',':')).encode()).digest())
                mask[order[:int(reference[:,j].sum())],j]=True
            assert (mask.sum(axis=0)==reference.sum(axis=0)).all()
            masks[f'RANDOM_MATCH_{rule}_{policy_seed}']=mask
    return masks

def measure(mask,em,f1,tokens,prefix,ix):
    q=np.where(mask,em[:,:,ix['SINGLE_Q']],em[:,:,ix['G1']]);f=np.where(mask,f1[:,:,ix['SINGLE_Q']],f1[:,:,ix['G1']])
    cost=np.where(mask,prefix+tokens[:,:,ix['SINGLE_Q']],tokens[:,:,ix['G1']]);u=q-5e-6*cost;single=em[:,:,ix['SINGLE_Q']]
    result=dict(EM=float(q.mean()),F1=float(f.mean()),service_tokens=float(cost.mean()),service_utility=float(u.mean()),EM_by_seed=q.mean(axis=0).tolist(),utility_by_seed=u.mean(axis=0).tolist(),ignore_count=int(mask.sum()),ignore_count_by_seed=mask.sum(axis=0).tolist(),corrections_vs_direct_single=int(((q==1)&(single==0)).sum()),destructions_vs_direct_single=int(((q==0)&(single==1)).sum()))
    return result,u

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--formal',required=True);ap.add_argument('--d12',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();f=Path(a.formal);d12=Path(a.d12);out=Path(a.output);assert not out.exists()
    read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert read(f/'INDEPENDENT_SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    s=read(f/'scoring_v1/SUMMARY.json');d=read(d12/'SUMMARY.json');assert d['matched_controls']['G1']['matched']==192 and d['scoring_summary_sha256']==sha(f/'scoring_v1/SUMMARY.json')
    arrays=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz');em,f1,tokens=(arrays[k] for k in ['em','f1','tokens']);ix={n:i for i,n in enumerate(s['conditions'])};seeds=s['seeds']
    units=sorted((f/'monitor/archive').glob('worker_*/unit_*'),key=lambda p:read(p/'INPUT.json')['index']);assert len(units)==96
    ids=[];notes=[];prefix=np.zeros((96,2))
    for i,unit in enumerate(units):
        ids.append(read(unit/'INPUT.json')['payload']['id']);rows={(r['execution_seed'],r['condition']):r for r in read(unit/'MATRIX.json')['conditions']};notes.append([])
        for j,seed in enumerate(seeds):
            row=rows[seed,'G1'];notes[i].append(json.loads(row['nodes']['M']['request']['messages'][1]['content'])['evidence_notes'])
            prefix[i,j]=sum(v['generation']['input_tokens']+v['generation']['output_tokens'] for k,v in row['nodes'].items() if k!='M')
            final=row['nodes']['M']['generation'];assert prefix[i,j]+final['input_tokens']+final['output_tokens']==tokens[i,j,ix['G1']]
    masks=build_masks(ids,seeds,notes);results={};utility={}
    for name,mask in masks.items():results[name],utility[name]=measure(mask,em,f1,tokens,prefix,ix)
    gates={}
    for rule in ['ANY','R']:
        name=f'DROP_ALL_IF_{rule}_ABSTAINS';value=utility[name];random_keys=[f'RANDOM_MATCH_{rule}_{v}' for v in POLICY_SEEDS]
        random_mean=np.mean([utility[k] for k in random_keys],axis=0)
        delta_ignore=value-utility['IGNORE_ALL_NOTES'];delta_random=value-random_mean
        comparisons={control:dict(utility_delta=float((value-(em[:,:,ix[control]]-5e-6*tokens[:,:,ix[control]])).mean()),delta_by_seed=(value-(em[:,:,ix[control]]-5e-6*tokens[:,:,ix[control]])).mean(axis=0).tolist()) for control in ['SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q']}
        gates[name]=dict(prefix_signal_gate=bool(delta_ignore.mean()>=.005 and (delta_ignore.mean(axis=0)>0).all() and delta_random.mean()>=.005 and (delta_random.mean(axis=0)>0).all()),system_gate=bool(all(v['utility_delta']>=.005 and min(v['delta_by_seed'])>0 for v in comparisons.values())),delta_vs_always_ignore=float(delta_ignore.mean()),delta_vs_ignore_by_seed=delta_ignore.mean(axis=0).tolist(),delta_vs_matched_random_mean=float(delta_random.mean()),delta_vs_random_by_seed=delta_random.mean(axis=0).tolist(),strong_control_comparisons=comparisons)
    result=dict(status='EXPOSED_DEVELOPMENT_FIXED_RULE_REPLAY',questions=96,question_seed_observations=192,seeds=seeds,lexicon=list(LEXICON),random_policy_seeds=list(POLICY_SEEDS),policies=results,screening=gates,new_GPU_calls=0,formal_round_increment=0,formal_gates_unchanged=s['gates'],source_sha256=sha(__file__),scoring_summary_sha256=sha(f/'scoring_v1/SUMMARY.json'),confidence_intervals=False,bootstrap_runs=0,limits=['Every policy pays the original G1 prefix before deciding whether to retain its notes.','Random controls match exact rejection count separately for each generation seed, using no labels.','Five random policy seeds reuse the same96 question groups and are not independent experiments.','Rules and lexicon are frozen for this development screen; any later change is a new exploratory candidate.','This is replay of actual paired terminal generations, not independent deployment or evidence for a novel active revalidation mechanism.','Continuation requires the same rule to pass on both tasks; no task pooling, retrospective winner or old-gate rescue.'])
    out.mkdir();(out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'PRIVATE_MASKS.json').write_text(json.dumps({k:v.astype(int).tolist() for k,v in masks.items()})+'\n');print(json.dumps(dict(status=result['status'],screening=gates,new_GPU_calls=0)))

if __name__=='__main__':main()
