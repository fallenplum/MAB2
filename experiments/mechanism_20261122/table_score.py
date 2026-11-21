"""Cold-start certified auditing using observed rows and terminal outcomes only."""
import argparse,hashlib,json,re
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np

POLICIES=('joint','eager','row_first','conditioning')
TAU=.01
NUMBER=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
FINAL=re.compile(r'(?:the\s+)?(?:(?:final\s+)?(?:answer|result)(?:\s+is)?\s*[:=]?\s*|####\s*)(?:\\boxed\{)?\s*[$]?\s*('+NUMBER+r')\s*\}?(?:\s*\.)?\s*$',re.I)
ONLY=re.compile(r'^\s*(?:\\boxed\{)?\s*[$]?\s*('+NUMBER+r')\s*\}?\.?\s*$')
def parse(text):
    lines=[line.replace('**','').replace('`','').strip().strip('$').strip() for line in text.splitlines() if line.strip()]
    if not lines:return None
    match=ONLY.fullmatch(lines[-1]) or FINAL.search(lines[-1])
    if not match:
        marked=[FINAL.search(v) for v in lines[-3:-1]]
        match=next((v for v in reversed(marked) if v),None)
    return Decimal(match.group(1).replace(',','')) if match else None


def schedule(s,base,row_price,term_price,fetch_row,fetch_term,method):
    n=len(s);known=np.zeros(n,dtype=bool);seen=np.zeros((n,n),dtype=bool)
    weights=np.zeros((n,n));contrib=0.;row_calls=term_calls=0;spent=0
    eps=1e-12
    while True:
        uncertain=np.where(known[:,None]&~seen,weights,0.)
        unobserved=s[~known].sum()*(1-min(base))
        lo=contrib+np.minimum(uncertain,0).sum()-unobserved
        hi=contrib+np.maximum(uncertain,0).sum()+unobserved
        finished=known.all() and seen.all()
        if (hi<TAU-eps or lo>TAU+eps or finished) and (method!='eager' or known.all()):
            return (contrib>TAU if finished else lo>TAU+eps),row_calls,term_calls,spent
        missing=np.flatnonzero(~known);cells=np.argwhere(known[:,None]&~seen)
        prices=term_price[tuple(cells.T)] if len(cells) else np.array([])
        cell_scores=abs(weights[tuple(cells.T)])/prices if len(cells) else np.array([])
        row_scores=2*s[missing]*(1-min(base))/row_price[missing] if len(missing) else np.array([])
        if method=='conditioning':
            unfinished=[i for i in np.argsort(-s,kind='stable') if not seen[i].all()]
            i=int(unfinished[0]);row_step=not known[i]
            if not row_step:j=int(np.flatnonzero(~seen[i])[0])
        else:
            row_step=bool(len(missing)) and (method in ('eager','row_first') or not len(cells) or max(row_scores)>=max(cell_scores))
            if row_step:i=int(missing[np.argmax(s[missing] if method in ('row_first','eager') else row_scores)])
            else:i,j=map(int,cells[np.argmax(cell_scores)])
        if row_step:
            values,cost=fetch_row(i);weights[i]=s[i]*(values-base);known[i]=True;row_calls+=1
        else:
            v,cost=fetch_term(i,j);contrib+=weights[i,j]*v;seen[i,j]=True;term_calls+=1
        spent+=cost


def one(data,receiver,source,y):
    middle=1-source;prefix=[np.asarray(w['probs'],float) for w in data]
    assert all(p.shape==(6,4) and p.min()>=0 and abs(p.sum(1)-1).max()<2e-6 for p in prefix)
    prefix=[p/p.sum(1,keepdims=True) for p in prefix]
    s=prefix[source][0];base=prefix[middle][1];m=prefix[middle][2:];w=s[:,None]*(m-base)
    prices=[np.asarray(r['policy_inputs'])+1 for r in data]
    initial=prices[source][0]+prices[middle][1]
    results=data[receiver]['final'];assert len(results)==20
    reads={k:parse(v['text']) for k,v in results.items()}
    value=np.array([[int(reads[f'{a},{b}']==y) for b in range(4)] for a in range(4)])
    new_price=np.array([[results[f'{a},{b}']['prefill']+160 for b in range(4)] for a in range(4)])
    real_price=np.array([[results[f'{a},{b}']['prefill']+results[f'{a},{b}']['decode'] for b in range(4)] for a in range(4)])
    last=np.array([int(reads[f'-1,{b}']==y) for b in range(4)])
    target=np.sum(w*value);old=np.sum(w*last[None,:])
    kcal=abs(w).sum()/2;assert abs(target)<=kcal+1e-10
    current_on=float(np.sum(s[:,None]*m*value));current_off=float(np.sum(s[:,None]*base*value))
    assert abs((current_on-current_off)-target)<1e-10
    out=dict(material=float(abs(target)>TAU),keep=float(target>TAU),stale_error=float((old>TAU)!=(target>TAU)),
             stale_regret=abs(target-TAU)*((old>TAU)!=(target>TAU)),always_keep_regret=max(0.,TAU-target),
             always_drop_regret=max(0.,target-TAU),quality_on=current_on,quality_off=current_off,
             certified_upstream=float(kcal<=TAU),last_credit=old,new_credit=target,
             unparsed=sum(x is None for x in reads.values())/20,truncated=sum(v['decode']==160 for v in results.values())/20,
             exhaustive_tokens=float(initial+sum(prices[middle][2:])+real_price.sum()),
             stale_audit_tokens=float(initial+sum(prices[middle][2:])+sum(results[f'-1,{b}']['prefill']+results[f'-1,{b}']['decode'] for b in range(4))),
             service_on_tokens=float(prices[source][0]+s@prices[middle][2:]+np.sum(s[:,None]*m*real_price)),
             service_off_tokens=float(prices[source][0]+prices[middle][1]+np.sum(s[:,None]*base*real_price)))
    for method in POLICIES:
        decision,rows,terms,spent=schedule(s,base,prices[middle][2:],new_price,
                                          lambda i:(m[i],prices[middle][2+i]),lambda i,j:(value[i,j],real_price[i,j]),method)
        assert bool(decision)==bool(target>TAU)
        out[method+'_rows']=rows;out[method+'_terms']=terms;out[method+'_tokens']=float(initial+spent)
    return out


def score(folder):
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    ws=[]
    for i in (0,1):
        rs=list(map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines()));by={v['key']:v for v in rs}
        assert len(by)==len(rs)==len(data) and set(by)==set(gold);ws.append(by)
    def mean(rr):return {k:float(np.mean([r[k] for r in rr])) for k in rr[0]}
    output={};support={};mechanism=True;algorithm=True
    for task in sorted({r['task'] for r in data}):
        material=set()
        qs=sorted((r for r in data if r['task']==task),key=lambda r:hashlib.sha256(('M46-block|'+r['key']).encode()).digest())
        for reader in (0,1):
            pairs=[]
            for q in qs:
                k=q['key'];batch=[one([v[k] for v in ws],reader,a,Decimal(gold[k])) for a in (0,1)]
                if any(r['material'] for r in batch):material.add(k)
                pairs.append(batch)
            stats=mean(sum(pairs,[]));stats['questions']=len(qs)
            stats['blocks']=[mean(sum(pairs[i:i+16],[])) for i in range(0,len(qs),16)]
            stats['roles']=[mean([r[a] for r in pairs]) for a in (0,1)]
            stats['saving']=1-stats['joint_tokens']/stats['exhaustive_tokens']
            stats['saving_vs_strongest']=1-stats['joint_tokens']/min(stats[m+'_tokens'] for m in POLICIES if m!='joint')
            mechanism &= stats['saving']>=.3;algorithm &= stats['saving_vs_strongest']>=.05
            output[f'{task}_{reader}']=stats
        support[task]=len(material);mechanism &= len(material)>=4
    return dict(cells=output,material_questions=support,mechanism_gate=bool(mechanism),algorithm_gate=bool(algorithm and mechanism))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_directory',type=Path)
    print(json.dumps(score(p.parse_args().run_directory),indent=2))
