"""Independent stdlib arithmetic audit; no main statistics imports or resampling."""
import argparse,hashlib,json,math
from pathlib import Path
NAMES=['G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','INDEPENDENT3_M','SINGLE_M','SINGLE_Q']
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(selection,private,result):
    x=private['em'];assert len(x)==96 and all(len(q)==2 and all(len(s)==9 and set(s)<= {0,1} for s in q) for q in x)
    assert private['ids']==[r['id'] for r in selection['questions']]
    checks=0
    def eq(a,b):
        nonlocal checks
        assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b);checks+=1
    ds=[];old=[];new=[]
    for i,q in enumerate(x):
        ds.append([]);old.append([]);new.append([])
        for s in range(2):
            values=q[s];before=[values[0]-values[1],values[0]-values[2]];after=[values[3]-values[4],values[3]-values[5]]
            old[-1].append(before);new[-1].append(after)
            j=0 if selection['questions'][i]['removed_upstream']=='U' else 1
            ds[-1].append(after[j]-before[j]-after[1-j]+before[1-j])
    point=sum(sum(v) for v in ds)/192
    eq(point,result['primary_D'])
    seed_means=[sum(v[s] for v in ds)/96 for s in range(2)]
    for s in range(2):eq(seed_means[s],result['D_by_seed'][s])
    support=sum(sum(v)!=0 for v in ds);eq(support,result['nonzero_question_support'])
    for shock in ['U','V']:
        chosen=[i for i,r in enumerate(selection['questions']) if r['removed_upstream']==shock];assert len(chosen)==48
        eq(sum(sum(ds[i]) for i in chosen)/96,result['D_by_shock'][shock])
    for key,data in [('old',old),('new',new),('change',None)]:
        for s in range(2):
            for e in range(2):eq(sum((new[i][s][e]-old[i][s][e]) if data is None else data[i][s][e] for i in range(96))/96,result['credit_by_edge_seed'][key][s][e])
    for j,name in enumerate(NAMES):
        for s in range(2):eq(sum(q[s][j] for q in x)/96,result['accuracy_by_condition_seed'][name][s])
    for graph,gi in [('G0',0),('G1',3)]:
        for baseline,bi in [('INDEPENDENT3_M',6),('SINGLE_M',7),('SINGLE_Q',8)]:
            for s in range(2):
                actual=result['comparisons_by_seed'][graph+'_minus_'+baseline][s]
                good=sum(q[s][gi]==1 and q[s][bi]==0 for q in x);bad=sum(q[s][gi]==0 and q[s][bi]==1 for q in x)
                for key,v in {'accuracy_a':sum(q[s][gi] for q in x)/96,'accuracy_b':sum(q[s][bi] for q in x)/96,'paired_difference':(good-bad)/96,'corrections':good,'breakages':bad,'paired_observations':96}.items():eq(v,actual[key])
    retained=point>=.05 and min(seed_means)>0 and support>=20
    assert (result['screen_action']=='RETAIN_MECHANISM_FOR_NEW_CONTRACT_REVIEW')==retained;checks+=1
    assert result['bootstrap_runs']==0 and result['confidence_intervals_computed'] is False and result['method_success_claim'] is False
    return {'status':'PASS_INDEPENDENT_STDLIB_POINT_AUDIT','checked_equalities':checks,'primary_D':point,'nonzero_question_support':support,'independent_model_execution':False,'independent_human_reviewer':False,'bootstrap_runs':0,'new_model_calls':0}
def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--private',required=True);p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(read(a.selection),read(a.private),read(a.summary));result['source_sha256']=sha(__file__)
    result['inputs_sha256']={k:sha(getattr(a,k)) for k in ['selection','private','summary']}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
