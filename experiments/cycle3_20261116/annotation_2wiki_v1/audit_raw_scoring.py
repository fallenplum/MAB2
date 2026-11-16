"""Independent stdlib reconstruction of Formal8 alias-aware metrics and frozen gates."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,re,string,math,hashlib

def norm(t):return ' '.join(re.sub(r'\b(?:a|an|the)\b',' ',t.lower().translate(str.maketrans('','',string.punctuation))).split())
def single_score(pred,gold):
    p,g=norm(pred),norm(gold);em=int(p==g)
    if p!=g and (p in {'yes','no','noanswer'} or g in {'yes','no','noanswer'}):return em,0.
    ps,gs=p.split(),g.split();hit=sum((Counter(ps)&Counter(gs)).values());return em,2.*hit/(len(ps)+len(gs)) if hit else 0.
def score(pred,golds):
    assert isinstance(golds,list) and golds
    values=[single_score(pred,g) for g in golds]
    return max(v[0] for v in values),max(v[1] for v in values)

def mean(x):return math.fsum(x)/len(x)
def corr(a,b):
    aa=[x-mean(a) for x in a];bb=[x-mean(b) for x in b];va=math.fsum(x*x for x in aa);vb=math.fsum(x*x for x in bb)
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(va*vb) if va>0 and vb>0 else None

def audit(arc,vault,s,questions):
    rows=defaultdict(list);keys={};checks=0;seeds=[26111671,26111672]
    for unit in arc.glob('worker_*/unit_*'):
        inp=json.loads((unit/'INPUT.json').read_bytes());q=inp['index'];assert inp['payload']['id']==questions[q]['id'];gold=vault[inp['payload']['id']]
        mat=json.loads((unit/'MATRIX.json').read_bytes())
        for c in mat['conditions']:
            if c['condition']=='VOTE_Q3':
                texts=[c['nodes'][n]['generation']['text'].strip() for n in ['M','U','V']];counts=Counter(norm(t) for t in texts);best=min(range(3),key=lambda i:(-counts[norm(texts[i])],i));answer=texts[best]
            else:answer=c['nodes']['M']['generation']['text'].strip()
            assert answer==c['answer'];checks+=1;em,f1=score(answer,gold);tokens=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for n in c['nodes'].values())
            row=dict(q=q,seed=c['execution_seed'],em=em,f1=f1,tokens=tokens,utility=em-tokens/200000);key=(q,c['execution_seed'],c['condition']);assert key not in keys;keys[key]=row;rows[c['condition']].append(row)
    assert len(keys)==2112 and set(rows)==set(s['conditions'])
    def equal(a,b):
        nonlocal checks
        if a is None or b is None:assert a is b
        else:assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
        checks+=1
    for name,values in rows.items():
        assert len(values)==192;ref=s['conditions'][name]
        for to,src in [('EM','em'),('F1','f1'),('service_tokens','tokens'),('service_utility','utility')]:equal(mean([x[src] for x in values]),ref[to])
        for j,seed in enumerate(seeds):
            part=[v for v in values if v['seed']==seed];equal(mean([v['em'] for v in part]),ref['EM_by_seed'][j]);equal(mean([v['utility'] for v in part]),ref['utility_by_seed'][j])
        delta=[keys[v['q'],v['seed'],'G1']['em']-v['em'] for v in values];pair=s['paired_G1_comparisons'][name];equal(mean(delta),pair['EM_delta']);assert delta.count(1)==pair['corrections'] and delta.count(-1)==pair['destructions'];checks+=2
    c0=[[],[]];c1=[[],[]];shifts=[[],[]];byedge=Counter();changedq=set()
    for q in range(96):
        for j,seed in enumerate(seeds):
            for edge in ['U','V']:
                a=keys[q,seed,'G0']['em']-keys[q,seed,'G0_MINUS_E'+edge]['em'];b=keys[q,seed,'G1']['em']-keys[q,seed,'G1_MINUS_E'+edge]['em'];c0[j].append(a);c1[j].append(b);shifts[j].append(b-a)
                if a!=b:byedge[edge]+=1;changedq.add(q)
    counts=[sum(v!=0 for v in s0) for s0 in shifts];equal(sum(counts)/384,s['credit_shift_fraction'])
    for j in range(2):equal(counts[j]/192,s['credit_shift_by_seed'][j])
    for j,e in enumerate(['U','V']):equal(byedge[e]/192,s['credit_shift_by_edge'][j])
    assert len(changedq)==s['distinct_shift_questions'];checks+=1
    rc,rs=corr(*c1),corr(*shifts);equal(rc,s['current_credit_seed_correlation']);equal(rs,s['credit_shift_seed_correlation'])
    gains=[];gain_units=[];quality=True
    for j,seed in enumerate(seeds):
        units=[(keys[q,seed,'G1']['em']-keys[q,seed,'SINGLE_Q']['em'])*200000-(keys[q,seed,'G1']['tokens']-keys[q,seed,'SINGLE_Q']['tokens']) for q in range(96)];gain_units.append(sum(units));gains.extend(x/200000 for x in units);equal(sum(units)/(200000*96),s['service_utility_gain_by_seed'][j])
        for name in ['SINGLE_Q','FLAT_ANNOTATION']:quality&=sum(keys[q,seed,'G1']['em']-keys[q,seed,name]['em'] for q in range(96))*100>=-96
    equal(mean(gains),s['service_utility_gain_vs_single_Q'])
    gates=dict(complete_matrix=True,fixed_G1_quality=quality,credit_shift_support=sum(counts)*100>=3*384 and all(c*100>=2*192 for c in counts) and len(changedq)>=20,cross_seed_reliability=rc is not None and rs is not None and rc>=.30 and rs>=.20,service_economic_signal=sum(gain_units)>=192000 and all(x>0 for x in gain_units))
    assert gates==s['gates'] and all(gates.values())==s['admission'];checks+=2
    for group,ref in s['type_breakdown'].items():
        ids={i for i,q in enumerate(questions) if q['type']==group};assert len(ids)==ref['questions'];checks+=1
        for name,values in rows.items():
            part=[v for v in values if v['q'] in ids]
            for to,src in [('EM','em'),('F1','f1'),('service_tokens','tokens')]:equal(mean([v[src] for v in part]),ref['conditions'][name][to])
    return dict(status='PASS_INDEPENDENT_RAW_SCORING_AND_GATES',checks=checks,raw_terminal_outputs=2112,question_groups=96,gates=gates,shared_evaluator_imported=False,shared_operator_imported=False,new_GPU_hours=0,bootstrap_runs=0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--archive',required=True);a.add_argument('--vault',required=True);a.add_argument('--summary',required=True);a.add_argument('--questions',required=True);a.add_argument('--output',required=True);x=a.parse_args()
    result=audit(Path(x.archive),json.loads(Path(x.vault).read_bytes()),json.loads(Path(x.summary).read_bytes()),[json.loads(v) for v in Path(x.questions).read_bytes().splitlines()]);result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out=Path(x.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
