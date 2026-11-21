import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean

def analyze(path):
    with path.open(encoding='utf-8-sig', newline='') as f:rr=list(csv.DictReader(f))
    by=defaultdict(dict)
    for r in rr:by[int(r['episode']),int(r['seed_index']),r['condition']][r['method']]=r
    def metric(r):
        n=int(r['service_questions']);assert n==64
        em=int(r['correct'])/n;t=int(r['complete_tokens'])/n;u=em-.000005*t
        assert abs(u-float(r['utility']))<1e-10
        return (em,u,t)
    entries=[]
    for ep in range(8):
        for sd in range(5):
            changed=by[ep,sd,'change'];original=by[ep,sd,'no_change']; assert changed and original
            side=changed['DELETE_CHANGED_NO_INIT']['selected_condition'][-1];assert side in 'UV'
            out={}
            for label,g in [('current',changed),('placebo',original)]:
                candidates={row['selected_condition'][-1]:row for row in (g['DELETE_CHANGED_NO_INIT'],g['DELETE_OTHER_NO_INIT'])}
                assert set(candidates)=={'U','V'}
                for edge in 'UV':out[label+'_fixed_'+edge]=metric(candidates[edge])
                out[label+'_protect']=metric(candidates['V' if side=='U' else 'U']);out[label+'_swapped']=metric(candidates[side]);out[label+'_keep']=metric(g['KEEP_NO_INIT'])
            for baseline in ('SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q','closure_EVSI','random','stale'):out['current_'+baseline]=metric(changed[baseline])
            entries.append((ep,sd,side,out))
    def collect(rows):
        means={k:[mean(row[3][k][i] for row in rows) for i in range(3)] for k in rows[0][3]}
        did=[(means['current_protect'][i]-means['current_swapped'][i])-(means['placebo_protect'][i]-means['placebo_swapped'][i]) for i in range(3)]
        return dict(episode_seed_cells=len(rows),means=means,interaction_accuracy=did[0],interaction_utility=did[1],delta_vs_best_fixed_utility=means['current_protect'][1]-max(means[k][1] for k in ('current_fixed_U','current_fixed_V','current_keep')),delta_vs_single_utility=means['current_protect'][1]-means['current_SINGLE_Q'][1])
    global_result=collect(entries);sides={s:collect([v for v in entries if v[2]==s]) for s in 'UV'};seeds={str(s):collect([v for v in entries if v[1]==s]) for s in range(5)}
    passed=global_result['interaction_utility']>=.005 and global_result['delta_vs_best_fixed_utility']>=.005 and all(v['interaction_utility']>0 for v in sides.values()) and sum(v['interaction_utility']>0 for v in seeds.values())>=4 and global_result['delta_vs_single_utility']>=0
    return dict(status='STRUCTURAL_RULE_REQUIRES_FRESH_REPLICATION' if passed else 'STRUCTURAL_ATTRIBUTION_NOT_QUALIFIED',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=512,episode_count=8,seed_count=5,columns=['accuracy','utility','tokens_per_service_item'],overall=global_result,by_seed=seeds,by_shock=sides,by_episode={str(i):collect([r for r in entries if r[0]==i]) for i in range(8)},new_GPU_hours=0,model_calls=0,scope='Reanalysis of exposed Formal5 TRAIN service groups; no causal population or independent replication claim.')
