"""Exact finite-orbit layout interventions for two advisers and one receiver."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

POLICIES = ('independent','sources_shared','left_shared','right_shared','all_shared')


def normalize(x):
    x = np.asarray(x,dtype=float)
    assert x.shape == (4,26,4) and np.min(x)>=0 and np.max(abs(x.sum(-1)-1))<2e-6
    return x / x.sum(-1,keepdims=True)


def evaluate_item(left,right,receiver,y):
    source = [left[:,0],right[:,0]]
    table = receiver[:,1:,y].reshape(4,5,5)
    all_graphs = []
    keys = np.array(list(itertools.product(range(4),repeat=3)))
    for i,j,k in keys:
        p,q = source[0][i],source[1][j]
        t = table[k]
        all_graphs.append([t[0,0],p@t[1:,0],q@t[0,1:],p@t[1:,1:]@q])
    i,j,k = keys.T
    masks = [np.ones(64,dtype=bool),i==j,i==k,j==k,(i==j)&(j==k)]
    menu = {name:np.asarray(all_graphs)[mask].mean(0) for name,mask in zip(POLICIES,masks)}
    actual,base = menu['all_shared'],menu['independent']
    for edge,p,read in [(1,source[0],table[:,1:,0]),(2,source[1],table[:,0,1:])]:
        covariance = ((p-p.mean(0))*(read-read.mean(0))).sum(1).mean()
        assert abs(actual[edge]-base[edge]-covariance)<1e-10
    np.testing.assert_allclose(base, [table[:,0,0].mean(),source[0].mean(0)@table[:,1:,0].mean(0),
                                      table[:,0,1:].mean(0)@source[1].mean(0),
                                      source[0].mean(0)@table[:,1:,1:].mean(0)@source[1].mean(0)],atol=1e-10)
    assert np.ptp([m[0] for m in menu.values()])<1e-10
    regret = actual.max()-actual[base.argmax()]
    stats = dict(regret=regret,material_loss=regret>.01,changed=actual.argmax()!=base.argmax(),
                 full_minus_no_advice_shared=actual[3]-actual[0],
                 full_minus_no_advice_independent=base[3]-base[0])
    for edge in (1,2):
        stats[f'edge{edge}_sync_credit_change']=actual[edge]-base[edge]
        stats[f'edge{edge}_abs_sync_credit_change']=abs(actual[edge]-base[edge])
    for name,m in menu.items():
        stats[name+'_full_quality']=m[3]
        stats[name+'_shared_regret']=actual.max()-actual[m.argmax()]
    for j,p in enumerate(source):
        stats[f'self{j}_mean_probability']=p[:,y].mean()
        stats[f'self{j}_ensemble_probability_argmax_correct']=float(p.mean(0).argmax()==y)
        votes=np.bincount(p.argmax(1),minlength=4)
        stats[f'self{j}_vote_accuracy_uniform_ties']=float(votes[y]==max(votes))/sum(votes==max(votes))
    return stats


def evaluate(inputs,gold,workers):
    result={}
    for role in range(2):
        for task in sorted(set(row['task'] for row in inputs)):
            rows=[]
            for row in inputs:
                if row['task']!=task:continue
                k=row['key']
                p=[normalize(w[k]['probs']) for w in workers]
                rows.append(evaluate_item(*p,p[role],gold[k]))
            stats={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            stats.update(questions=len(rows),material_questions=sum(bool(r['material_loss']) for r in rows))
            stats['passed']=stats['regret']>.002 and stats['material_loss']>=.05
            result[f'{task}_{role}']=stats
    return result


def evaluate_folder(folder):
    read=lambda name:json.loads((folder/name).read_bytes())
    inputs,gold=read('INPUT.json'),read('SEALED.json')
    keys={r['key'] for r in inputs}
    assert keys==set(gold) and len(keys)==len(inputs)
    workers=[]
    for role in range(2):
        entries=[json.loads(s) for s in (folder/f'WORKER{role}.jsonl').read_bytes().splitlines()]
        records={r['key']:r for r in entries}
        assert set(records)==keys and len(entries)==len(keys)
        workers.append(records)
    return dict(cells=evaluate(inputs,gold,workers),
                exhaustive_input_tokens=[sum(np.asarray(r['input_tokens']).sum().item() for r in w.values()) for w in workers])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(evaluate_folder(args.run_directory),indent=2))
