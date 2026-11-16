"""Frozen exploratory gate. Run only after full archive, cost and execution audit."""
import argparse,json,hashlib,re,string
from pathlib import Path
from collections import Counter
import numpy as np

CONDITIONS=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','FLAT','SINGLE_Q','SINGLE_M','BM25','DENSE','VOTE_Q3']
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def f1(pred,gold):
    a=norm(pred).split();b=norm(gold).split();n=sum((Counter(a)&Counter(b)).values())
    return 2*n/(len(a)+len(b)) if n else 0.
def reference(a):
    if a['unanswerable']:return 'Unanswerable','none'
    if a['extractive_spans']:return ', '.join(a['extractive_spans']),'extractive'
    if a['free_form_answer']:return a['free_form_answer'],'abstractive'
    if a['yes_no']:return 'Yes','boolean'
    if a['yes_no'] is not None:return 'No','boolean'
    raise ValueError('Malformed reference')
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
def gates(y,tokens,valid):
    assert y.shape==(16,2,12) and tokens.shape==y.shape and np.isfinite(y).all() and np.isfinite(tokens).all()
    u=y-5e-6*tokens;old=u[:,:,0,None]-u[:,:,[1,2]];new=u[:,:,3,None]-u[:,:,[4,5]];shift=new-old
    current_count=int(np.any(np.all(np.abs(new)>=.02,axis=1),axis=1).sum());shift_count=int(np.any(np.all(np.abs(shift)>=.02,axis=1),axis=1).sum())
    cr=corr(new[:,0],new[:,1]);sr=corr(shift[:,0],shift[:,1]);means=y.mean((0,1));um=u.mean((0,1))
    result=dict(selector_valid=float(np.mean(valid))>=.95,full_quality=all(means[3]>=means[i]-.02 for i in [7,8,11]),retrieval_flat_utility=all(um[3]>=um[i]+.01 for i in [6,9,10]),retrieval_flat_quality=all(means[3]>=means[i]-.02 for i in [6,9,10]),supported_credit_papers=current_count>=4,supported_shift_papers=shift_count>=4,credit_reliability=cr is not None and cr>=.30,shift_reliability=sr is not None and sr>=.20)
    return dict(gates={k:bool(v) for k,v in result.items()},passed=all(result.values()),current_credit_correlation=cr,shift_correlation=sr,current_supported_papers=current_count,shift_supported_papers=shift_count,selector_valid_fraction=float(np.mean(valid)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--vault',required=True);ap.add_argument('--vault-sha',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();run=Path(a.run);out=Path(a.output);assert not out.exists();read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    release=read(run/'SCORING_RELEASE.json');assert release['approved'] and release['completed_papers']==16 and release['gold_sha256']==a.vault_sha
    assert sha(Path(a.vault))==a.vault_sha;gold=read(Path(a.vault));archive=run/'archive';manifest=read(archive/'ARCHIVE_MANIFEST.json')
    for file,h in manifest.items():assert sha(archive/file)==h
    y=np.full((16,2,12),np.nan);em=y.copy();tokens=y.copy();valid=[];seen=set();type_scores={c:{} for c in CONDITIONS};private=[]
    for unit in sorted(archive.glob('worker_*/unit_*')):
        inp=read(unit/'INPUT.json');mat=read(unit/'MATRIX.json');i=inp['index'];assert i not in seen;seen.add(i);qid=inp['payload']['id'];refs=[reference(v) for v in gold[qid]];assert mat['status']=='COMPLETE_UNSCORED'
        for row in mat['conditions']:
            s=[26111691,26111692].index(row['execution_seed']);c=CONDITIONS.index(row['condition']);assert np.isnan(y[i,s,c]);pred=row['answer'];scores=[f1(pred,g) for g,typ in refs];idx=max(range(len(scores)),key=lambda j:scores[j]);y[i,s,c]=scores[idx];em[i,s,c]=max(float(norm(pred)==norm(g)) for g,typ in refs);typ=refs[idx][1];type_scores[row['condition']].setdefault(typ,[]).append(scores[idx]);tokens[i,s,c]=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for n in row['nodes'].values());valid.extend(n['selection_valid'] for n in row['nodes'].values() if 'selection_valid' in n);private.append(dict(index=i,seed=s,condition=row['condition'],f1=scores[idx],em=em[i,s,c],tokens=tokens[i,s,c]))
    assert seen==set(range(16));result=gates(y,tokens,valid);result.update(status='EXPLORATORY_PILOT_PASS' if result['passed'] else 'STOP_FORMAL9_APPLICABILITY_ROUTE',papers=16,seeds=2,terminal_answers=384,metric='official_Qasper_max_reference_token_F1',independent_confirmation=False,condition_results={c:dict(f1=float(y[:,:,j].mean()),em=float(em[:,:,j].mean()),utility=float((y[:,:,j]-5e-6*tokens[:,:,j]).mean()),tokens=float(tokens[:,:,j].mean()),f1_by_seed=y[:,:,j].mean(0).tolist(),f1_by_type={k:dict(mean=float(np.mean(v)),support=len(v)) for k,v in type_scores[c].items()}) for j,c in enumerate(CONDITIONS)},paired_G1={c:dict(mean_f1_delta=float((y[:,:,3]-y[:,:,j]).mean()),improved=int((y[:,:,3]>y[:,:,j]).sum()),harmed=int((y[:,:,3]<y[:,:,j]).sum()),equal=int((y[:,:,3]==y[:,:,j]).sum())) for j,c in enumerate(CONDITIONS)},archive_sha256=sha(archive/'ARCHIVE_MANIFEST.json'),source_sha256=sha(Path(__file__)))
    out.mkdir();np.savez_compressed(out/'PRIVATE_ARRAYS.npz',f1=y,em=em,tokens=tokens);(out/'PRIVATE_RECORDS.json').write_text(json.dumps(private),encoding='utf-8');(out/'SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
