"""Exposed development replay of homogeneous terminal aggregation on real cached model answers."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from sklearn.linear_model import LogisticRegression

MODELS=('Llama-3.1-8B-Instruct','gemma-2-9b-it','granite-3.3-8b-instruct','glm-4-9b-chat','internlm3-8b-instruct','Qwen3-8B')
SEEDS=(17,31,47)
PRICE=0.000005
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def collect(data):
    tables=[]; h={}; fingerprints=set()
    for model in MODELS:
        files=list((data/model).glob('*.json'));assert len(files)==1
        raw=files[0].read_bytes();j=json.loads(raw);assert j['split']=='valid'
        fingerprints.add(j['data_fingerprint']);h[model]=hashlib.sha256(raw).hexdigest();rr={x['index']:x for x in j['records']};assert len(rr)==len(j['records']);tables.append(rr)
    assert len(fingerprints)==1 and all(t.keys()==tables[0].keys() for t in tables)
    keys=sorted(tables[0]);qs=[]; gold=[];votes=[];tokens=[];price=[]
    for k in keys:
        allr=[t[k] for t in tables];q=allr[0]['origin_query'];y=allr[0]['ground_truth'];assert isinstance(q,str) and y in ('A','B')
        assert all(r['origin_query']==q and r['ground_truth']==y and r['prediction'] in ('A','B','') for r in allr)
        qs.append(q);gold.append(int(y=='A'));votes.append([{'A':1,'B':-1,'':0}[r['prediction']] for r in allr]);tokens.append([int(r['prompt_tokens'])+int(r['completion_tokens']) for r in allr]);price.append([float(r['cost']) for r in allr])
    return qs,np.array(gold),np.array(votes),np.array(tokens),np.array(price),h

def fold_groups(questions):
    parent=list(range(len(questions)));pairs=defaultdict(list)
    def find(i):
        while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(a,b):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    ws=[set(re.findall(r'\w+',q.casefold())) for q in questions]
    # Fill-in counterparts often share almost the whole sentence. Also join adjacent release indices as a conservative extra grouping.
    for i in range(1,len(questions),2):merge(i-1,i)
    for i,a in enumerate(ws):
        for j in range(i):
            b=ws[j]
            if len(a&b)>=.72*len(a|b):merge(i,j)
    groups=defaultdict(list)
    for i in range(len(ws)):groups[find(i)].append(i)
    folds=np.zeros(len(ws),int);counts=[0,0,0]
    for ids in sorted(groups.values(),key=lambda ids:(-len(ids),digest('MAB2-D33|'+min(questions[i] for i in ids)))):
        f=min(range(3),key=lambda j:counts[j]);folds[ids]=f;counts[f]+=len(ids)
    return folds,len(groups),max(map(len,groups.values()))

def vote(v,subset):
    x=v[:,subset];s=x.sum(axis=1);fallback=np.zeros(len(v),int)
    for col in x.T:fallback=np.where(fallback!=0,fallback,col)
    return np.where(s==0,fallback,np.sign(s))

def build(v,t,base):
    arms=[tuple(base)+tuple(a) for a in ((),(1,),(2,),(3,),(1,2,3))];return np.column_stack([vote(v,a) for a in arms]),np.column_stack([t[:,a].sum(axis=1) for a in arms]),arms

def stratify(new,old,p):
    cur=(new!=new[:,[-1]]).sum(axis=1); shifted=((new-new[:,[-1]])!=(old-old[:,[-1]])).sum(axis=1)
    return np.column_stack([cur>0,shifted>0,np.abs(p-.5)<.25]).astype(int),cur,shifted

def allocate(strata,scores,k):
    quota=np.ones(len(strata),int);assert len(strata)<=k<=sum(map(len,strata))
    while sum(quota)<k:
        i=max((i for i,d in enumerate(strata) if quota[i]<len(d)),key=lambda i:scores[i]/(quota[i]+1));quota[i]+=1
    return quota

def reward(answer,p):return np.where(answer==1,p,np.where(answer==-1,1-p,0))

def evaluate(data):
    qs,y,v,t,money,hashes=collect(data);folds,groups,largest=fold_groups(qs);old,ot,old_arms=build(v,t,[0]);new,nt,arms=build(v,t,[4,5]); rows=[]
    direct=np.column_stack([vote(v,[i]) for i in range(len(MODELS))]);ind=np.arange(len(qs));tcost=PRICE*t
    for fold in range(3):
        h=ind[folds==(fold+1)%3];pool=ind[folds==(fold+2)%3];test=ind[folds==fold]
        model=LogisticRegression(C=1.0,solver='lbfgs',max_iter=500).fit(np.column_stack([v[h],v[h]==0]),y[h]);p=model.predict_proba(np.column_stack([v[pool],v[pool]==0]))[:,1]
        bins,cur,shift=stratify(new[pool],old[pool],p);slices=[np.where((bins==key).all(axis=1))[0] for key in sorted(set(map(tuple,bins)))];unc=4*p*(1-p)
        proxy=reward(new[pool],p[:,None]);direct_d=proxy-proxy[:,[-1]];relative_cost=PRICE*(nt[pool]-nt[pool,[-1]][:,None]);basis=(direct_d-relative_cost).mean(axis=0)
        actual=reward(new[pool],y[pool,None]);residual=(actual-actual[:,[-1]])-direct_d
        result={};paired={};outcomes={};oracle=(reward(new[test],y[test,None])-PRICE*nt[test]).mean(axis=0)
        def observe(name,chosen,selected,overhead,actionset='new',paid_history=True):
            if actionset=='new':answers=new[test,chosen];use=arms[chosen];service=nt[test,chosen]
            else:answers=direct[test,chosen];use=(chosen,);service=t[test,chosen]
            labels=len(selected);correct=reward(answers,y[test]);base_tokens=int(t[h].sum()) if paid_history else 0
            outcomes[name]=correct
            result[name]=dict(action=int(chosen),accuracy=float(correct.mean()),service_tokens=int(service.sum()),history_tokens=base_tokens,pool_tokens=int(overhead),historical_labels=len(h) if paid_history else 0,new_labels=labels,
                             service_utility=float(correct.mean()-PRICE*service.mean()),full_utility=float(correct.mean()-PRICE*(service.sum()+overhead+base_tokens)/len(test)),service_recorded_cost=float(money[test][:,use].sum()),service_calls=len(use)*len(test))
        histacc=reward(new[h],y[h,None])-PRICE*nt[h];hist_single=reward(direct[h],y[h,None])-tcost[h]
        observe('historical_exact_replay',int(histacc.mean(0).argmax()),[],0);observe('historical_best_single',int(hist_single.mean(0).argmax()),[],0,'single');observe('single_Qwen3',5,[],0,'single',False)
        oldhist=reward(old[h],y[h,None])-PRICE*ot[h];observe('reuse_old_graph_action',int(oldhist.mean(0).argmax()),[],0)
        observe('current_model_no_labels',int(basis.argmax()),[],int(t[pool].sum()));observe('hindsight_fixed_action',int(oracle.argmax()),[],0,paid_history=False)
        for method in ('random','periodic','uncertainty','current_pivotal','graph_transport'):
            ws={'random':np.ones(len(p)),'periodic':np.ones(len(p)),'uncertainty':np.maximum(.01,unc),'current_pivotal':np.maximum(.01,np.sqrt(unc*cur)), 'graph_transport':np.maximum(.01,np.sqrt(unc*cur)*(1+shift/(new.shape[1]-1)))}[method]
            quota=allocate(slices,[float(ws[a].sum()) for a in slices],24)
            for seed in SEEDS:
                rng=np.random.default_rng(seed+1000*fold);estimate=basis.copy();selected=[]
                blocks=[(np.arange(len(pool)),24)] if method in ('random','periodic') else zip(slices,quota)
                for ids,k in blocks:
                    chosen=(ids[(np.arange(k)*len(ids)//k+seed)%len(ids)] if method=='periodic' else rng.choice(ids,k,replace=False));assert len(set(chosen))==k
                    selected.extend(chosen);estimate+=len(ids)/len(pool)*residual[chosen].mean(0)
                observe(method+'_'+str(seed),int(estimate.argmax()),selected,int(t[pool].sum()))
        for seed in SEEDS:
            k='graph_transport_'+str(seed);paired[k]={}
            for ctrl in ('random_'+str(seed),'periodic_'+str(seed),'current_pivotal_'+str(seed),'uncertainty_'+str(seed),'historical_exact_replay','historical_best_single','single_Qwen3'):
                delta=outcomes[k]-outcomes[ctrl];paired[k][ctrl]=dict(corrected=int((delta>0).sum()),damaged=int((delta<0).sum()),tied=int((delta==0).sum()))
        rows.append(dict(fold=fold,historical=len(h),pool=len(pool),service=len(test),pool_pivotal=int((cur>0).sum()),pool_strata=len(slices),historical_recorded_cost=float(money[h].sum()),pool_recorded_cost=float(money[pool].sum()),by_method=result,paired=paired))
    counts={k:sum(row['service'] for row in rows) for k in rows[0]['by_method']}; agg={k:{metric:sum(row['service']*row['by_method'][k][metric] for row in rows)/counts[k] for metric in ('accuracy','service_utility','full_utility')} for k in counts}
    d=[]; controls=('random','uncertainty','periodic','current_pivotal')
    for seed in SEEDS:
        k='graph_transport_'+str(seed);control=max(agg[m+'_'+str(seed)]['full_utility'] for m in controls); cheap=max(agg[m]['full_utility'] for m in ('historical_exact_replay','reuse_old_graph_action','historical_best_single','single_Qwen3','current_model_no_labels'))
        d.append({'seed':seed,'delta_vs_best_matched':agg[k]['full_utility']-control,'delta_vs_best_cheap':agg[k]['full_utility']-cheap})
    ok=all(r['delta_vs_best_matched']>=.01 and r['delta_vs_best_cheap']>=0 for r in d)
    return dict(status='VOTE_DEVELOPMENT_SIGNAL_ONLY' if ok else 'NO_QUALIFIED_VOTE_TRANSPORT_SIGNAL',model_files=hashes,questions=len(qs),duplicate_control_groups=groups,largest_group=largest,folds=rows,aggregate=agg,paired_deltas=d,token_price=PRICE,selected_labels_per_fold=24,seeds=list(SEEDS),new_GPU_hours=0,scope='exposed cached official validation set; independent upstream outputs; deterministic terminal graph; no measured new inference, no independent confirmation')
