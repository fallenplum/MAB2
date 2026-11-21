"""Secondary strict-versus-official rescoring and stronger branch ordering on the same records."""
import json
from decimal import Decimal
from pathlib import Path
import numpy as np
from official_numeric_audit import original_functions
from table_score import one,parse,schedule
from blind_scheduler import Audit

def cost_strong(s,base,means,obs,price_b,price_r,actual,method):
    engine=Audit(s,base,price_b,price_r,method);cost=0;contrib=0.
    while True:
        missing=sum(s[~engine.known])*(1-min(base));pending=np.where(engine.known[:,None]&~engine.seen,engine.w,0.)
        lo=contrib+np.minimum(0.,pending).sum()-missing;hi=contrib+np.maximum(0.,pending).sum()+missing
        if hi<.01-1e-12 or lo>.01+1e-12 or engine.seen.all():break
        i,j=engine.next()
        if j is None:engine.update((i,j),means[i]);cost+=price_b[i]
        else:contrib+=engine.w[i,j]*obs[i,j];engine.update((i,j),str(obs[i,j]));cost+=actual[i,j]
    assert (np.sum(s[:,None]*(means-base)*obs)>.01)==(lo>.01 if not engine.seen.all() else contrib>.01)
    return cost

def analyze(directory,source,external):
    import hashlib
    units={}
    for obj in json.loads(source.read_bytes()).values():
        if not isinstance(obj.get('table'),str):continue
        title=obj.get('table_title') or '';txt=(title+'\n' if title else '')+obj['table']+'\n'+obj['question']
        key=hashlib.sha256(' '.join(txt.lower().split()).encode()).hexdigest();units[key]=obj.get('unit') or ''
    gold=json.loads((directory/'SEALED.json').read_bytes());inputs=json.loads((directory/'INPUT.json').read_bytes())
    ws=[{r['key']:r for r in map(json.loads,(directory/f'WORKER{i}.jsonl').read_bytes().splitlines())} for i in (0,1)]
    extract,normalize=original_functions(external);out={}
    for task in ('lower','upper'):
        for reader in (0,1):
            for mode in ('registered','official'):
                acc=[]
                for row in inputs:
                    if row['task']!=task:continue
                    k=row['key'];p=[np.asarray(w[k]['probs']) for w in ws];p=[z/z.sum(axis=1,keepdims=True) for z in p]
                    costs=[np.asarray(w[k]['policy_inputs'])+1 for w in ws];final=ws[reader][k]['final']
                    vals=[];y=Decimal(gold[k])
                    for a in range(4):
                        for b in range(4):
                            txt=final[f'{a},{b}']['text']
                            if mode=='registered':v=parse(txt)==y
                            else:
                                try:v=normalize(extract(txt,[],[]),units[k])==normalize(gold[k],units[k])
                                except (ValueError,IndexError,ZeroDivisionError):v=False
                            vals.append(int(v))
                    obs=np.array(vals).reshape(4,4)
                    cp=np.array([[final[f'{a},{b}']['prefill']+160 for b in range(4)] for a in range(4)])
                    ct=np.array([[final[f'{a},{b}']['prefill']+final[f'{a},{b}']['decode'] for b in range(4)] for a in range(4)])
                    for a in (0,1):
                        m=1-a;s=p[a][0];base=p[m][1];means=p[m][2:];initial=costs[a][0]+costs[m][1];bp=costs[m][2:];rec={}
                        for method in ('joint','row_first'):
                            _,_,_,cost=schedule(s,base,bp,cp,lambda i:(means[i],bp[i]),lambda i,j:(obs[i,j],ct[i,j]),method)
                            rec[method]=int(initial+cost)
                        for method in ('nested','unpriced'):
                            rec[method]=int(initial+cost_strong(s,base,means,obs,bp,cp,ct,method))
                        acc.append(rec)
                out[f'{task}_{reader}_{mode}']={k:float(np.mean([r[k] for r in acc])) for k in acc[0]}
    return out
