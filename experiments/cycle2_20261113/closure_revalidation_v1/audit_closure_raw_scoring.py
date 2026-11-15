"""Second implementation of raw answer scoring and voting; standard library only.

Requires already released, completely scored archive. No scorer/operator
imports, model calls, confidence intervals, or private text in the receipt.
"""
from pathlib import Path
import argparse,hashlib,json,re,string,math
CONDITIONS=('G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','INDEPENDENT3_Q','SINGLE_Q','VOTE_Q3')
SEEDS=[26111571,26111572,26111573,26111574,26111575]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def normalize(s):return ' '.join(re.sub(r'\b(?:the|an|a)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def metrics(prediction,gold):
    a,b=normalize(prediction),normalize(gold);em=int(a==b)
    if a!=b and ({a,b}&{'yes','no','noanswer'}):return em,0.
    left,right=a.split(),b.split();words=set(left)&set(right)
    overlap=sum(min(left.count(w),right.count(w)) for w in words)
    return em,(2*overlap/(len(left)+len(right)) if overlap else 0.)
def vote(texts):
    keys=[normalize(x) for x in texts];counts=[sum(k==key for k in keys) for key in keys]
    return counts.index(max(counts))
def audit(approval_path,private_path,summary_path):
    approval=read(approval_path);assert approval['decision']=='APPROVED_FROZEN_CLOSURE_ICS_SCORING'
    for key in ['selection','archive_manifest','replay_receipt','cost_closure','vault']:
        assert sha(approval[key]['path'])==approval[key]['sha256']
    root=Path(approval['archive_root']);manifest=read(approval['archive_manifest']['path'])
    replay=read(approval['replay_receipt']['path']);assert replay['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY' and replay['completed_units']==640
    selection=read(approval['selection']['path']);gold=read(approval['vault']['path']);private=read(private_path);summary=read(summary_path)
    ids=[q['id'] for q in selection['questions']];assert len(ids)==len(set(ids))==640 and private['ids']==ids and set(gold)==set(ids)
    checks=0;scores={g:[0]*5 for g in CONDITIONS};service={g:[0]*5 for g in CONDITIONS};finish={'eos':0,'length':0};blank=0
    for i,q in enumerate(selection['questions']):
        files=list(root.glob(f'worker_*/unit_{i:03d}/MATRIX.json'));assert len(files)==1
        f=files[0];assert sha(f)==manifest[f.relative_to(root).as_posix()]
        m=read(f);assert m['status']=='COMPLETE_UNSCORED' and m['complete_terminal_conditions']==45
        assert [c['key'] for c in m['conditions']]==[f'{s}/{g}' for s in SEEDS for g in CONDITIONS]
        for c in m['conditions']:
            s=SEEDS.index(c['execution_seed']);g=c['condition'];node='M'
            if g=='VOTE_Q3':
                order=['M','U','V'];node=order[vote([c['nodes'][n]['generation']['text'] for n in order])]
                assert node==c['selected_node'];checks+=1
            record=c['nodes'][node]['generation'];answer=record['text'].strip();assert answer==c['answer'];checks+=1
            em,f1=metrics(answer,gold[q['id']]);assert em==private['em'][i][s][g] and math.isclose(f1,private['f1'][i][s][g],abs_tol=1e-12,rel_tol=1e-12);checks+=2
            scores[g][s]+=em
            if i%80>=16:service[g][s]+=em
            finish[record['finish_reason']]+=1;blank+=not bool(answer)
    for g in CONDITIONS:
        for s in range(5):
            assert math.isclose(scores[g][s]/640,summary['all_condition_seed_accuracy'][g][s],abs_tol=1e-12)
            assert math.isclose(service[g][s]/512,summary['service_condition_seed_accuracy'][g][s],abs_tol=1e-12);checks+=2
    assert summary['finish_counts']==finish and summary['blank_final_outputs']==blank;checks+=2
    return {'status':'PASS_INDEPENDENT_RAW_SCORING_AND_VOTING','checks':checks,'questions':640,'terminal_answers':28800,'vote_decisions':3200,
        'all_condition_correct_by_seed':scores,'service_condition_correct_by_seed':service,'blank_final_outputs':blank,'finish_counts':finish,
        'new_model_calls':0,'bootstrap_runs':0,'human_independent_review':False,'stdlib_only':True,
        'input_sha256':{'approval':sha(approval_path),'private':sha(private_path),'summary':sha(summary_path)},'source_sha256':sha(__file__)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--private',required=True);p.add_argument('--summary',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=audit(a.approval,a.private,a.summary);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'checks':result['checks'],'terminal_answers':result['terminal_answers']}))
if __name__=='__main__':main()
