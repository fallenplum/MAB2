"""Restrict legal extractive targets to normalized substrings of the passage."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from floor_decomposition import decomposition
from source_transport import cases,compare,INVALID
from passage_score import normalized


def score(folder):
    qs=json.loads((folder/'INPUT.json').read_bytes())
    groups=defaultdict(list)
    for i,(team,_,v) in enumerate(cases(folder,'m54')):
        q=qs[i//3];context=normalized(' '.join(q['sections']))
        s,new,base,cond,labels,rc,tc,prices,y,init,fee,basefee=v
        assert y in context and team==i%3
        legal=np.array([[x!=INVALID and x in context for x in row] for row in labels])
        filtered=np.where(legal,labels,INVALID)
        floors=decomposition(s,base,cond,filtered,rc,tc,prices,y,init,.01)
        edit=compare(s,new,base,cond,filtered,rc,tc,prices,y,init,fee,basefee)
        groups[team].append(dict(span_rate=legal.mean(),excluded=np.mean((labels!=INVALID)&~legal),
                observed_gold=np.any(filtered==y),gold_cost=floors['gold'],hard_cost=floors['hardest'],
                blind_cost=floors['all_numeric'],**{k:edit[k] for k in ('changed','stable_but_needs_evidence','row_tv_skip','reweighted_skip','repair','restart_with_base')}))
    results={}
    for t,rows in groups.items():
        m={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        gap=m['blind_cost']-m['gold_cost']
        results[str(t)]=dict(**m,common_share=(m['blind_cost']-m['hard_cost'])/gap if gap>0 else None,n=len(rows))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    print(json.dumps(score(a.folder),indent=2))
