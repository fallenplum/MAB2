"""Independent scalar replay of the descriptive D11 decision summaries."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np


def verify(em,f1,tokens,shocks,names,summary):
    assert em.shape==f1.shape==tokens.shape==(len(shocks),2,11)
    assert len(set(names))==11 and set(shocks)<=set('UV')
    assert np.isfinite(em).all() and np.isfinite(f1).all() and np.isfinite(tokens).all()
    assert np.isin(em,[0,1]).all() and (tokens>=0).all()
    ix={name:i for i,name in enumerate(names)}
    strategies=['KEEP','DELETE_AFFECTED','DELETE_OTHER','STALE_OLD_GRAPH_ORACLE_TRANSFER','CURRENT_PERFECT_INFORMATION_ORACLE','ALTERNATE_SEED_ORACLE_TRANSFER']
    rows={k:[] for k in strategies};errors={'affected':[],'other':[]};changed=0
    def outcome(i,j,t,action):
        edge=shocks[i] if action==1 else ('V' if shocks[i]=='U' else 'U')
        name=f'G{t}' if action==0 else f'G{t}_MINUS_E{edge}'
        c=ix[name];q=float(em[i,j,c]);f=float(f1[i,j,c]);cost=float(tokens[i,j,c])
        return q,f,cost,q-5e-6*cost
    def best(i,j,t):
        return max(range(3),key=lambda a:outcome(i,j,t,a)[3])
    for i in range(len(shocks)):
        for j in range(2):
            old=best(i,j,0);current=best(i,j,1);changed+=old!=current
            actions=[0,1,2,old,current,best(i,1-j,1)]
            base=outcome(i,j,1,0)
            for k,a in zip(strategies,actions):rows[k].append((j,a,*outcome(i,j,1,a),base[0]))
            for key,a in [('affected',1),('other',2)]:
                og=outcome(i,j,0,a)[3]-outcome(i,j,0,0)[3]
                ng=outcome(i,j,1,a)[3]-base[3]
                wrong=(og>0)!=(ng>0)
                errors[key].append((j,wrong,(og>0 and ng<=0),(og<=0 and ng>0),abs(ng)*wrong,outcome(i,j,0,0)[0]-outcome(i,j,0,a)[0],base[0]-outcome(i,j,1,a)[0]))
    checks=0
    def eq(actual,expected):
        nonlocal checks
        assert math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-10),(actual,expected)
        checks+=1
    def mean(v):return math.fsum(v)/len(v)
    for k,values in rows.items():
        s=summary['strategies'][k]
        for name,pos in [('EM',2),('F1',3),('service_tokens',4),('service_utility',5)]:eq(s[name],mean([v[pos] for v in values]))
        for j in range(2):
            subset=[v for v in values if v[0]==j]
            eq(s['EM_by_seed'][j],mean([v[2] for v in subset]));eq(s['utility_by_seed'][j],mean([v[5] for v in subset]))
        for a in range(3):eq(s['action_counts'][a],sum(v[1]==a for v in values))
        eq(s['corrections_vs_keep'],sum(v[2]==1 and v[6]==0 for v in values))
        eq(s['destructions_vs_keep'],sum(v[2]==0 and v[6]==1 for v in values))
        assert s['available_without_feedback']==(k in strategies[:3]);checks+=1
    eq(summary['old_new_optimal_action_disagreements'],changed)
    for key,values in errors.items():
        s=summary['net_edge_direction'][key]
        for field,pos in [('stale_net_direction_errors',1),('wrong_delete',2),('wrong_keep',3)]:eq(s[field],sum(v[pos] for v in values))
        eq(s['mean_binary_decision_regret'],mean([v[4] for v in values]))
        for j in range(2):
            subset=[v for v in values if v[0]==j]
            eq(s['errors_by_seed'][j],sum(v[1] for v in subset));eq(s['regret_by_seed'][j],mean([v[4] for v in subset]))
        for field,pos in [('old_credit',5),('current_credit',6)]:
            for sign,condition in [('positive',lambda x:x>0),('zero',lambda x:x==0),('negative',lambda x:x<0)]:eq(s[field][sign],sum(condition(v[pos]) for v in values))
    oracle=mean([v[5] for v in rows['CURRENT_PERFECT_INFORMATION_ORACLE']])
    for k in strategies[:3]:eq(summary['oracle_headroom_vs_fixed'][k],oracle-mean([v[5] for v in rows[k]]))
    eq(summary['oracle_headroom_vs_single_Q'],oracle-float((em[:,:,ix['SINGLE_Q']]-5e-6*tokens[:,:,ix['SINGLE_Q']]).mean()))
    for a in strategies[:3]:
        for b in strategies[:3]:
            if a!=b:eq(summary['fixed_pairwise_utility_points'][a+' minus '+b],mean([v[5] for v in rows[a]])-mean([v[5] for v in rows[b]]))
    eq(summary['questions'],len(shocks));eq(summary['question_seed_count'],2*len(shocks))
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--formal',required=True);ap.add_argument('--diagnosis',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    f=Path(a.formal);d=Path(a.diagnosis);out=Path(a.output);assert not out.exists()
    read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert read(f/'INDEPENDENT_SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    s=read(f/'scoring_v1/SUMMARY.json');diag=read(d/'SUMMARY.json');arrays=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz')
    units=sorted((f/'monitor/archive').glob('worker_*/unit_*'),key=lambda p:read(p/'INPUT.json')['index'])
    assert len(units)==96 and [read(p/'INPUT.json')['index'] for p in units]==list(range(96))
    count=verify(arrays['em'],arrays['f1'],arrays['tokens'],[read(p/'INPUT.json')['shock'] for p in units],list(s['conditions']),diag)
    result=dict(status='PASS_INDEPENDENT_SCALAR_D11_DECISION_AUDIT',checks=count,questions=96,summary_sha256=sha(d/'SUMMARY.json'),scoring_summary_sha256=sha(f/'scoring_v1/SUMMARY.json'),source_sha256=sha(Path(__file__)),scope='Decision choices, means, directions and realized headroom; prerequisite-call cost ledger is outside this scalar audit',new_GPU_calls=0)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
