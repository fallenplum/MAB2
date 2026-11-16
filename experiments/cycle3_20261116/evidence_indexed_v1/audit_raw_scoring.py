"""Raw-output scoring audit independent of evaluator, NumPy and operator imports."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,re,string,math

def normalize(text):
    text=text.lower().translate(str.maketrans('','',string.punctuation))
    return ' '.join(re.sub(r'\b(?:a|an|the)\b',' ',text).split())
def score(pred,gold):
    x,y=normalize(pred),normalize(gold);em=int(x==y)
    if x!=y and (x in {'yes','no','noanswer'} or y in {'yes','no','noanswer'}):return em,0.
    px,gy=x.split(),y.split();hit=sum((Counter(px)&Counter(gy)).values())
    return em,(2.*hit/(len(px)+len(gy)) if hit else 0.)

def audit(archive,vault,summary):
    rows=defaultdict(list);by_key={};selector=[];certs=[];violations=0;checks=0
    seeds=[26111661,26111662]
    for unit in sorted(archive.glob('worker_*/unit_*')):
        inp=json.loads((unit/'INPUT.json').read_bytes());m=json.loads((unit/'MATRIX.json').read_bytes());q=inp['index'];gold=vault[inp['payload']['id']]
        for c in m['conditions']:
            key=(q,c['execution_seed'],c['condition']);assert key not in by_key
            # Reconstruct final answer from actual node outputs rather than trusting row['answer'].
            if c['condition']=='VOTE_Q3':
                nodes=['M','U','V'];texts=[c['nodes'][n]['generation']['text'].strip() for n in nodes];counts=Counter(normalize(t) for t in texts);winner=min(range(3),key=lambda i:(-counts[normalize(texts[i])],i));answer=texts[winner]
            else:answer=c['nodes']['M']['generation']['text'].strip()
            assert answer==c['answer'];checks+=1
            em,f1=score(answer,gold);tok=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for n in c['nodes'].values())
            record=dict(q=q,seed=c['execution_seed'],em=em,f1=f1,tokens=tok,row=c)
            rows[c['condition']].append(record);by_key[key]=record
            selector.extend(n['selection_valid'] for n in c['nodes'].values() if 'selection_valid' in n)
    assert len(by_key)==96*2*11 and set(rows)==set(summary['conditions'])
    def equal(x,y):
        nonlocal checks
        assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(x,y);checks+=1
    for name,values in rows.items():
        assert len(values)==192
        s=summary['conditions'][name]
        equal(sum(v['em'] for v in values)/192,s['EM']);equal(sum(v['f1'] for v in values)/192,s['F1'])
        equal(sum(v['tokens'] for v in values)/192,s['service_tokens'])
        equal(sum(v['em']-5e-6*v['tokens'] for v in values)/192,s['service_utility'])
        for j,seed in enumerate(seeds):
            v=[v for v in values if v['seed']==seed];assert len(v)==96
            equal(sum(x['em'] for x in v)/96,s['EM_by_seed'][j]);equal(sum(x['em']-5e-6*x['tokens'] for x in v)/96,s['utility_by_seed'][j])
        delta=0;cor=0;des=0
        for v in values:
            g=by_key[v['q'],v['seed'],'G1']['em'];delta+=g-v['em'];cor+=g==1 and v['em']==0;des+=g==0 and v['em']==1
        z=summary['paired_G1_comparisons'][name];equal(delta/192,z['EM_delta']);assert cor==z['corrections'] and des==z['destructions'];checks+=2
    changed=[];changed_questions=set();seed_changes=Counter()
    for q in range(96):
        for seed in seeds:
            for edge in ['U','V']:
                credits=[]
                for t in range(2):
                    full=by_key[q,seed,f'G{t}'];abl=by_key[q,seed,f'G{t}_MINUS_E{edge}'];credits.append(full['em']-abl['em'])
                    same=full['row']['terminal_evidence_ids']==abl['row']['terminal_evidence_ids'];certs.append(same)
                    if same:violations+=int(full['row']['answer']!=abl['row']['answer'] or full['row']['nodes']['M']['request_fingerprint']!=abl['row']['nodes']['M']['request_fingerprint'])
                flag=credits[0]!=credits[1];changed.append(flag);seed_changes[seed]+=flag
                if flag:changed_questions.add(q)
    equal(sum(changed)/len(changed),summary['credit_shift_fraction']);equal(sum(certs)/len(certs),summary['certificate_fraction']);equal(sum(selector)/len(selector),summary['selection_valid_rate'])
    assert len(changed_questions)==summary['distinct_shift_questions'] and violations==summary['certificate_violations'];checks+=2
    quality=True
    for base in ['SINGLE_Q','FLAT_EVIDENCE']:
        for seed in seeds:
            gap=sum(by_key[q,seed,'G1']['em']-by_key[q,seed,base]['em'] for q in range(96));quality&=gap*100>=-96
    gates=dict(selector_valid=sum(selector)*100>=98*len(selector),fixed_G1_quality=quality,
       credit_shift_support=sum(changed)*100>=3*len(changed) and all(seed_changes[s]*100>=2*192 for s in seeds) and len(changed_questions)>=20,
       certificate_coverage=sum(certs)*10>=len(certs) and violations==0)
    assert gates==summary['gates'] and all(gates.values())==summary['admission'];checks+=2
    return dict(status='PASS_INDEPENDENT_RAW_SCORING_AND_GATES',checks=checks,raw_terminal_outputs=2112,question_groups=96,gates=gates,shared_evaluator_imported=False,shared_operator_imported=False,new_GPU_hours=0,bootstrap_runs=0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--vault',required=True);a.add_argument('--summary',required=True);a.add_argument('--output',required=True);x=a.parse_args()
    result=audit(Path(x.archive),json.loads(Path(x.vault).read_bytes()),json.loads(Path(x.summary).read_bytes()));out=Path(x.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
