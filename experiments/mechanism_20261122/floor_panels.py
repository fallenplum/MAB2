import argparse,hashlib,json
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
from official_numeric_audit import original_functions
from floor_decomposition import decomposition

def cases(folder,phase,train,grader):
    qs=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    count=3 if phase=='m50' else 2
    ws=[{r['key']:r for r in map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in range(count)]
    assert len(qs)==len(gold) and all(set(w)==set(gold) for w in ws)
    if phase=='m46':
        units={}
        for obj in json.loads(train.read_bytes()).values():
            if not isinstance(obj.get('table'),str):continue
            title=obj.get('table_title') or '';txt=(title+'\n' if title else '')+obj['table']+'\n'+obj['question']
            units[hashlib.sha256(' '.join(txt.lower().split()).encode()).hexdigest()]=obj.get('unit') or ''
        extract,normalize=original_functions(grader)
        def canonical(text,unit):
            try:
                v=Decimal(normalize(extract(text,[],[]),unit))
                return str(v.normalize()) if v.is_finite() else 'INVALID'
            except (InvalidOperation,ValueError,ZeroDivisionError,IndexError):return 'INVALID'
    for reader in range(count):
        for q in qs:
            k=q['key'];y=str(Decimal(gold[k]).normalize())
            for a in ((0,1) if phase=='m46' else (reader,)):
                if phase=='m46':
                    mid=1-a;p=[np.asarray(w[k]['probs']) for w in ws];p=[v/v.sum(1,keepdims=True) for v in p]
                    s=p[a][0];base=p[mid][1];m=p[mid][2:];rc=np.asarray(ws[mid][k]['policy_inputs'][2:])+1
                    init=ws[a][k]['policy_inputs'][0]+ws[mid][k]['policy_inputs'][1]+2
                    terminal=[ws[reader][k]['final'][f'{i},{j}'] for i,j in np.ndindex(4,4)]
                    labels=np.reshape([canonical(v['text'],units[k]) for v in terminal],(4,4));limit=160
                else:
                    req=ws[reader][k];full=req['arms']['full'];pre=full['prefix'];s=np.array(pre['0']);base=np.array(pre['1'])
                    m=np.array([pre[str(i)] for i in range(2,6)]);rc=np.array(req['lengths'][0][2:])+1;init=sum(req['lengths'][0][:2])+2
                    terminal=[full['terminal'][str(i)] for i in range(16)];labels=np.reshape([v['label'] for v in terminal],(4,4));limit=512
                tc=np.reshape([v['prefill']+v['decode'] for v in terminal],(4,4));prices=np.reshape([v['prefill']+limit for v in terminal],(4,4))
                yield (q['task'],reader,k),(s,base,m,labels,rc,tc,prices,y,init)

def run(folder,phase,train,grader,thresholds=(.005,.01,.02)):
    groups={}
    for (task,reader,key),params in cases(folder,phase,train,grader):
        for tau in thresholds:
            r=decomposition(*params,tau)
            groups.setdefault(f'{task}_{reader}_{tau}',[]).append(r)
    out={}
    for group,rows in groups.items():
        agg={k:float(np.mean([v[k] for v in rows])) for k in rows[0]}
        gap=agg['all_numeric']-agg['gold'];agg['hardest_share_of_premium']=agg['hard_minus_gold']/gap if gap>0 else None
        agg['premium']=agg['all_numeric']/agg['gold']-1;agg['joint_to_minimum']=agg['joint']/agg['all_numeric']
        agg['instances']=len(rows)
        agg['by_reachability']={}
        for state in (False,True):
            subset=[v for v in rows if v['gold_reached']==state]
            if subset:agg['by_reachability'][str(state)]=dict(instances=len(subset),gold=float(np.mean([v['gold'] for v in subset])),
                    all_numeric=float(np.mean([v['all_numeric'] for v in subset])))
        out[group]=agg
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('m46','m50'));p.add_argument('folder',type=Path)
    p.add_argument('--author-train',type=Path);p.add_argument('--grader-files',type=Path)
    a=p.parse_args();print(json.dumps(run(a.folder,a.phase,a.author_train,a.grader_files),indent=2))
