"""Exact enumeration of a three-agent categorical relay under two decision readouts."""
import argparse
import json
from pathlib import Path
import numpy as np


def prob(rows):
    p=np.asarray(rows,dtype=float)
    assert p.shape==(26,4) and p.min()>=0 and abs(p.sum(1)-1).max()<2e-6
    return p/p.sum(1,keepdims=True)


def relay(left,middle,receiver,y):
    source=left[0]
    m=middle[1:].reshape(5,5,4)
    end=receiver[1:].reshape(5,5,4)
    change=m[1:,0,:]-m[0,0,:]
    kernels=[np.broadcast_to(end[0,1:,:],(4,4,4)),end[1:,1:,:]]
    shifts=[np.einsum('ab,abc->ac',change,r) for r in kernels]
    np.testing.assert_allclose([s.sum(1) for s in shifts],0,atol=1e-10)
    tv=[source@(abs(s).sum(1)/2) for s in shifts]
    credits=[source@s[:,y] for s in shifts]
    first_stage=source@(abs(change).sum(1)/2)
    assert max(tv)<=first_stage+1e-10
    readout_effect=source@np.einsum('ab,ab->a',change,(kernels[1]-kernels[0])[:,:,y])
    assert abs(readout_effect-(credits[1]-credits[0]))<1e-10
    def action(c):return 1 if c>.01 else -1 if c<-.01 else 0
    expected=[]
    for r in kernels:
        expected.extend([np.einsum('a,b,abc->c',source,m[0,0],r)[y],
                         np.einsum('a,ab,abc->c',source,m[1:,0],r)[y]])
    return dict(last_tv=tv[0],all_tv=tv[1],first_stage=first_stage,last_credit=credits[0],all_credit=credits[1],
                strict_reversal=action(credits[0])*action(credits[1])==-1,
                action_changed=action(credits[0])!=action(credits[1]),
                materially_changed=abs(readout_effect)>.01,abs_credit_change=abs(readout_effect),
                weaker=tv[1]<tv[0]-1e-9,stronger=tv[1]>tv[0]+1e-9,
                last_off=expected[0],last_on=expected[1],all_off=expected[2],all_on=expected[3])


def evaluate(inputs,gold,workers):
    output={}
    for task in sorted({r['task'] for r in inputs}):
        for reader in (0,1):
            scores=[]
            for item in inputs:
                if item['task']!=task:continue
                k=item['key'];p=[prob(w[k]['probs']) for w in workers]
                scores.append([relay(p[a],p[1-a],p[reader],gold[k]) for a in (0,1)])
            def average(rows):
                return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            stats=average([r for pair in scores for r in pair])
            stats['assignments']=[average([r[a] for r in scores]) for a in (0,1)]
            stats['questions']=len(scores)
            stats['support_questions']=sum(any(r['materially_changed'] for r in pair) for pair in scores)
            stats['attenuation']=1-stats['all_tv']/stats['last_tv'] if stats['last_tv']>0 else None
            stats['passed']=(stats['attenuation'] is not None and stats['attenuation']>=.2
                             and stats['materially_changed']>=.05 and stats['support_questions']>=3)
            output[f'{task}_{reader}']=stats
    return output


def score_folder(directory):
    inputs=json.loads((directory/'INPUT.json').read_bytes())
    gold=json.loads((directory/'SEALED.json').read_bytes())
    workers=[]
    for role in (0,1):
        rows=list(map(json.loads,(directory/f'WORKER{role}.jsonl').read_bytes().splitlines()))
        tab={r['key']:r for r in rows}
        assert len(tab)==len(rows)==len(inputs) and set(tab)==set(gold)=={r['key'] for r in inputs}
        workers.append(tab)
    return evaluate(inputs,gold,workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    print(json.dumps(score_folder(parser.parse_args().run_directory),indent=2))
