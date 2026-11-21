"""Separate strong forced interventions from their frequency under actual sender policies."""
import json
from pathlib import Path
import numpy as np

def summary(s,alt,rows,base,read):
    delta=rows-base
    t=abs(delta).sum(1)/2
    effect=np.einsum('ab,ab->a',delta,read)
    assert np.all(abs(effect)<=t+1e-8)
    K=s@t;new=alt@t
    drift=(alt-s)@effect
    assert abs(drift)<=abs(alt-s).sum()/2*np.ptp(effect)+1e-8
    assert abs(new-K)<=abs(alt-s).sum()/2*np.ptp(t)+1e-8
    fire=abs(effect)>=.1
    return dict(natural_K=K,other_sender_K=new,cpt_diameter=max(abs(v-w).sum()/2 for v in [*rows,base] for w in [*rows,base]),
                maximal_single=t.max(),uniform_K=t.mean(),natural_abs_credit=abs(s@effect),
                forced_any=bool(max(abs(effect))>=.1),forced_and_certified=bool(any(fire) and K<=.01),
                natural_mass_strong=float(s@fire),observable_overlap=bool(t.max()>=.25 and K<=.01),
                source_replacement_decision_changed=bool((s@effect>.01)!=(alt@effect>.01)),
                source_replacement_certificate_changed=bool((K<=.01)!=(new<=.01)),
                difference_saturated_by_rare=float(s@t<=.01 and t.mean()>.01),
                alternative_actual_quality_change=drift,natural_cert=K<=.01,uniform_cert=t.mean()<=.01)


def compute(folder,kind):
    rows=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in (0,1)]
    assert all(set(w)==set(gold) and len(w)==len(rows) for w in ws)
    if kind=='numeric':
        from decimal import Decimal
        from table_score import parse
    out={}
    for task in sorted({r['task'] for r in rows}):
        for reader in (0,1):
            records=[];support=0
            for row in rows:
                if row['task']!=task:continue
                k=row['key'];pair=[];p=[np.asarray(w[k]['probs']) for w in ws]
                p=[v/v.sum(1,keepdims=True) for v in p]
                for a in (0,1):
                    m=1-a;n=p[a].shape[-1];s=p[a][0];alt=p[m][0]
                    if kind=='numeric':
                        base=p[m][1];branch=p[m][2:]
                        read=np.array([[int(parse(ws[reader][k]['final'][f'{i},{j}']['text'])==Decimal(gold[k])) for j in range(4)] for i in range(4)])
                    else:
                        mt=p[m][1:].reshape(n+1,n+1,n);base=mt[0,0];branch=mt[1:,0]
                        read=p[reader][1:].reshape(n+1,n+1,n)[1:,1:,gold[k]]
                    pair.append(summary(s,alt,branch,base,read))
                support+=any(v['forced_and_certified'] for v in pair);records+=pair
            stats={key:float(np.mean([r[key] for r in records])) for key in records[0]}
            stats.update(support_questions=support,question_role_pairs=len(records));out[f'{task}_{reader}']=stats
    return out
