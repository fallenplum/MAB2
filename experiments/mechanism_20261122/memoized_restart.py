"""Exact-key cache baseline, paid on demand in the current graph's cold order."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from blind_scheduler import Audit
from source_transport import cases, insert, run, truth
from qasc_score import Item,credit_rows


def examples(folder,phase):
    if phase=='m56':
        qs=json.loads((folder/'INPUT.json').read_bytes())
        ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(3)]
        for team in range(3):
            for r in credit_rows(ws,qs,team,1):
                it=Item(r['data']);cache=dict(r['history']);matched=0
                while it.next is not None:
                    action=it.next
                    price=it.price
                    it.take()
                    if action in cache:
                        assert np.array_equal(it.history[-1][1],cache[action]);matched+=price
                assert np.array_equal(it.direction,r['truth'])
                yield team,r['old_cost'],r['warm_sum'],r['new_source']+it.spent-matched
    else:
        for team,_,values in cases(folder,phase):
            s,new,b,m,l,rc,tc,prices,y,init,source,baseprice=values
            _,hist,old_cost=run(s,b,m,l,rc,tc,prices)
            warm,_,repair=run(new,b,m,l,rc,tc,prices,hist)
            cache=dict(hist)
            audit=Audit(new,b,rc,prices,'joint');memo=0
            while audit.decision() is None:
                i,j=audit.next()
                if (i,j) in cache:val=cache[i,j]
                else:
                    val=m[i] if j is None else l[i,j]
                    memo+=rc[i] if j is None else tc[i,j]
                insert(audit,(i,j),val)
            assert audit.decision()==warm.decision()==truth(new,b,m,l)
            yield team,init+old_cost,source+repair,source+memo


def score(folder,phase):
    groups=defaultdict(list)
    for team,*costs in examples(folder,phase):groups[f'team{team}'].append(costs)
    result={}
    for team,rows in groups.items():
        old,warm,memo=np.asarray(rows).T
        result[team]=dict(n=len(rows),strict_better=int(sum(warm<memo)),same=int(sum(warm==memo)),
            strict_worse=int(sum(warm>memo)),warm=float(warm.mean()),memo=float(memo.mean()),previous=float(old.mean()),
            increment_gain=1-warm.sum()/memo.sum(),total_gain=1-(old+warm).sum()/(old+memo).sum())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['m50','m54','m56'])
    p.add_argument('folder',type=Path);args=p.parse_args();print(json.dumps(score(args.folder,args.phase),indent=2))
