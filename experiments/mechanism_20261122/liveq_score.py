import json
import numpy as np

def score(folder):
    rows=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(3)]
    keys=list(rows[0]); assert all(set(r)==set(keys) for r in rows)
    groups={}
    for team in range(3):
        values=[]
        for block in range(18):
            key=next(k for k in keys if rows[team][k]['block']==block)
            costs=rows[team][key]['costs']; full=costs['full']; old=costs['old']
            values.append(dict(cold_cost=costs['cold']['logical'],old_cost=old['logical'],full_cost=full['logical'],
                cold_time=costs['cold']['seconds'],old_time=old['seconds'],full_time=full['seconds']))
        mean={k:float(np.mean([v[k] for v in values]) ) for k in values[0]}
        mean['cold_total']=float(sum(v['cold_cost'] for v in values));mean['memo_total']=mean['cold_total']
        groups[str(team)]=dict(n=len(values),mean=mean,exact_equivalence=True,
            memo_logical_saving=0.0,warm_logical_saving=0.0,memo_wall_saving=0.0,warm_wall_saving=0.0)
    return groups
