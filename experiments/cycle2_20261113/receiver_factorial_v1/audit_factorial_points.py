"""Independent standard-library arithmetic replay; not independent human review."""
from pathlib import Path
import argparse,json,hashlib,math
BASE=('G0','G0_MINUS_EU','G0_MINUS_EV','G1','G1_MINUS_EU','G1_MINUS_EV','SINGLE_Q','SINGLE_M','INDEPENDENT3_M')
# Obtain serialized order by the frozen cost rows, and crosscheck expected set.
def audit(private,summary,selection):
    x=private['em'];t=private['tokens'];k=private['kernel_seconds'];rows=private['cost_rows']
    order=[r['condition'].split('/')[1] for r in rows[:29]]
    expected=set(BASE)|{g+'__'+v for v in ['Q_PARTIAL','M_FULL','Q_FULL'] for g in BASE[:6]}|{'INDEPENDENT3_Q','VOTE_Q3'}
    assert len(order)==len(set(order))==29 and set(order)==expected
    assert len(x)==len(t)==len(k)==96 and len(rows)==5568
    count=0
    def eq(a,b):
        nonlocal count
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b);count+=1
    def val(i,s,n):return x[i][s][order.index(n)]
    signals={}
    for n in order:
        j=order.index(n)
        for s in [0,1]:
            eq(sum(val(i,s,n) for i in range(96))/96,summary['accuracy_by_condition_seed'][n][s])
            eq(sum(t[i][s][j] for i in range(96)),summary['tokens_by_condition_seed'][n][s])
            eq(sum(k[i][s][j] for i in range(96)),summary['kernel_seconds_by_condition_seed'][n][s])
    for cell in ['M_PARTIAL','Q_PARTIAL','M_FULL','Q_FULL']:
        name=lambda g:g if cell=='M_PARTIAL' else g+'__'+cell
        ds=[]
        for i,q in enumerate(selection['questions']):
            aff='EU' if q['removed_upstream']=='U' else 'EV';other='EV' if aff=='EU' else 'EU';d=[]
            for s in [0,1]:
                credit=lambda g,e:val(i,s,name(g))-val(i,s,name(g+'_MINUS_'+e))
                d.append((credit('G1',aff)-credit('G0',aff))-(credit('G1',other)-credit('G0',other)))
            ds.append(d)
        byseed=[sum(v[s] for v in ds)/96 for s in [0,1]];mean=sum(sum(v) for v in ds)/192;support=sum(sum(v)!=0 for v in ds)
        signals[cell]=(mean,byseed,support);target=summary['credit_signals_by_cell'][cell];eq(mean,target['D']);eq(support,target['nonzero_question_support'])
        for a,b in zip(byseed,target['D_by_seed']):eq(a,b)
    nominee='G1__Q_FULL';comparisons={}
    for baseline in ['SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','INDEPENDENT3_M','VOTE_Q3']:
        target=summary['nominated_comparisons'][baseline];ds=[]
        for s in [0,1]:
            pairs=[(val(i,s,nominee),val(i,s,baseline)) for i in range(96)];diff=sum(a-b for a,b in pairs)/96;ds.append(diff)
            eq(diff,target['by_seed'][s]['difference']);eq(sum(a==1 and b==0 for a,b in pairs),target['by_seed'][s]['corrections']);eq(sum(a==0 and b==1 for a,b in pairs),target['by_seed'][s]['breakages'])
        support=sum(sum(val(i,s,nominee)-val(i,s,baseline) for s in [0,1])!=0 for i in range(96));eq(sum(ds)/2,target['pooled']['difference']);eq(support,target['nonzero_question_support']);comparisons[baseline]=(sum(ds)/2,ds,support)
    ni=order.index(nominee);bi=order.index('INDEPENDENT3_Q');total=lambda z,j:sum(z[i][s][j] for i in range(96) for s in [0,1]);tr=total(t,ni)/total(t,bi);kr=total(k,ni)/total(k,bi)
    eq(tr,summary['nominee_cost_ratios_vs_independent3_Q']['total_tokens']);eq(kr,summary['nominee_cost_ratios_vs_independent3_Q']['kernel_seconds'])
    q=comparisons['SINGLE_Q'];v=comparisons['VOTE_Q3'];ind=comparisons['INDEPENDENT3_Q'];sig=signals['Q_FULL']
    gates={'gain_over_single_Q':q[0]>=.02 and min(q[1])>0 and q[2]>=20,'gain_over_cheap_Q_vote':v[0]>=.02 and min(v[1])>0,'independent_aggregation_quality_screen':ind[0]>=-.01 and min(ind[1])>=-.03,'independent_aggregation_cost_screen':tr<=.85 and kr<=.9,'credit_signal_screen':sig[0]>=.05 and min(sig[1])>0 and sig[2]>=20}
    assert gates==summary['gates'];count+=5
    return {'status':'PASS_INDEPENDENT_STDLIB_POINT_ARITHMETIC','checks':count,'human_independent_peer_review':False,'GPU_hours':0,'bootstrap_runs':0,'gates':gates}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--private',required=True);ap.add_argument('--summary',required=True);ap.add_argument('--selection',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    paths={n:Path(getattr(a,n)) for n in ['private','summary','selection']};result=audit(*(json.loads(paths[n].read_bytes()) for n in ['private','summary','selection']))
    result['input_sha256']={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()}
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
