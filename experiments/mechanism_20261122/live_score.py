"""Verify each hardware decision against its independently executed exhaustive counterfactual."""
import json
from decimal import Decimal
from pathlib import Path
import numpy as np
from table_score import parse
POLICIES=('full','anchored','weighted')

def process(trial,y):
    metrics={}
    for method in POLICIES:
        r=trial['policies'][method];p=np.array(r['probs']);assert p.shape==(6,4)
        assert np.max(abs(p.sum(1)-1))<1e-10
        w=p[0,:,None]*(p[2:]-p[1]);k=abs(w).sum()/2
        assert abs(k-r['weighted'])<1e-9 and r['weighted']<=r['anchored']+1e-9<=r['diameter']+2e-9
        if r['skip']:
            assert len(r['results'])==0 and method!='full' and r[method]<.01
            decision=False
        else:
            assert len(r['results'])==16
            rewards=np.array([parse(t)==y for t in r['results']]).reshape(4,4)
            credit=np.sum(w*rewards)
            assert abs(credit)<=k+1e-8
            decision=bool(credit>.01)
        metrics[method]=dict(decision=decision,tokens=int(r['prefill']+r['decode']),prefill=r['prefill'],decode=r['decode'],seqs=r['seqs'],
                             wall_seconds=r['wall_seconds'],skip=r['skip'],diameter_would_skip=r['diameter']<.01-1e-10)
    return metrics


def evaluate(folder):
    inputs=json.loads((folder/'INPUT.json').read_bytes());labels=json.loads((folder/'SEALED.json').read_bytes())
    out={};passed=True
    for role in (0,1):
        rows=list(map(json.loads,(folder/f'WORKER{role}.jsonl').read_bytes().splitlines()))
        records={r['key']:r for r in rows}
        assert len(records)==len(rows)==len(inputs)==48 and set(records)==set(labels)
        for group in ('lower','upper'):
            panel=[records[r['key']] for r in inputs if r['task']==group]
            orders=[tuple(r['arm_order']) for r in panel]
            assert len(set(orders))==6 and all(orders.count(k)==4 for k in set(orders))
            vals=[process(r,Decimal(labels[r['key']])) for r in panel]
            def summarize(indices):
                if not indices:return None
                aggregate={m:{k:sum(float(vals[i][m][k]) for i in indices) for k in vals[0][m]} for m in POLICIES}
                sums=[aggregate[m]['wall_seconds'] for m in POLICIES]
                return dict(questions=len(indices),totals=aggregate,
                            weighted_vs_full=1-sums[2]/sums[0],weighted_vs_anchored=1-sums[2]/sums[1],
                            weighted_wall_wins=sum(vals[i]['weighted']['wall_seconds']<vals[i]['anchored']['wall_seconds'] for i in indices),
                            mismatches=sum(any(vals[i][m]['decision']!=vals[i]['full']['decision'] for m in POLICIES) for i in indices))
            allq=summarize(range(len(vals)));clean=summarize([i for i,r in enumerate(panel) if not r['other_process_observed']])
            ok=allq['weighted_vs_full']>=.15 and allq['weighted_vs_anchored']>=.05 and allq['mismatches']==0
            out[f'{group}_{role}']=dict(all_trials=allq,uncontended=clean,other_process_count=sum(r['other_process_observed'] for r in panel),
                                      passed=bool(ok));passed&=ok
    return dict(cells=out,passed=bool(passed))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    print(json.dumps(evaluate(parser.parse_args().folder),indent=2))
