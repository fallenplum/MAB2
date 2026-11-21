"""Assess sharp finite-channel envelopes and visible-state sufficiency on natural messages."""
import json
import numpy as np
from lineage_score import parse

GRID=(0.,.0025,.005,.01,.02,.05)
def discrete(probs,lengths,y,a,reader):
    n=probs[a].shape[-1];b=1-a;s=probs[a][0];mid=probs[b][1:].reshape(n+1,n+1,n)
    last=probs[reader][1:].reshape(n+1,n+1,n)
    primary=mid[1:,0];base=mid[0,0]
    w=s[:,None]*(primary-base);kw=np.abs(w).sum()/2;km=np.abs(w.sum(0)).sum()/2
    x=float(np.sum(w*last[0,1:,y]));z=float(np.sum(w*last[1:,1:,y]))
    assert km<=kw+1e-10 and abs(x)<=km+1e-10 and abs(z)<=kw+1e-10
    assert abs(w[w>0].sum()-kw)<1e-10
    kl=np.sum(primary*np.log(np.maximum(primary,1e-300)/np.maximum(base,1e-300)),axis=1)
    pin=float(s@np.sqrt(np.maximum(kl,0)/2))
    assert pin>=kw-1e-8
    mode=bool(np.all(primary.argmax(axis=1)==base.argmax()))
    qcost=[np.asarray(v)+1 for v in lengths]
    init=qcost[a][0]+qcost[b][1:].reshape(n+1,n+1)[:,0].sum()
    cost=qcost[reader][1:].reshape(n+1,n+1)[1:,1:].ravel()
    old_cost=qcost[reader][1:].reshape(n+1,n+1)[0,1:].sum()
    row=dict(kw=kw,km=km,actual_last_abs=abs(x),actual_all_abs=abs(z),mode_certified=mode,
             mode_unsafe=mode and abs(z)>.01,stale_decision_error=(x>.01)!=(z>.01),
             guarantee_lost=km<=.01<kw,lost_and_exceeded=km<=.01<abs(z),pinsker_certified=pin<=.01,
             all_certified=kw<=.01,initial_tokens=init,full_new_cost=sum(cost),full_new_calls=n*n)
    # Re-run the strong M44 baseline with its original signed keep/drop decision and cost ledger.
    from credit_certificate import query_order
    flat=w.ravel();prior=np.tile(last[0,1:,y],n);target=last[1:,1:,y].ravel()
    for policy in ('width_cost','directed'):
        calls,expense,decision=query_order(flat,prior,target.__getitem__,cost,policy,None)
        assert bool(decision)==(z>.01)
        row[policy+'_new_calls']=calls;row[policy+'_total_tokens']=init+old_cost+expense
    row['all_table_tokens']=init+old_cost+sum(cost)
    for threshold in GRID:
        row['last_cert_'+str(threshold)]=km<=threshold
        row['all_cert_'+str(threshold)]=kw<=threshold
    return row


def natural(row,y,n):
    q=dict(trials=0,valid=0,same_vote=0,same_vote_different_text=0,flips=0,corrections=0,harms=0,
           same_text=0,same_text_changed=0,same_text_final_string_changed=0)
    for i in range(2):
        m0,m1=row['middle'][0]['text'],row['middle'][i+1]['text']
        b0,b1=parse(m0,n),parse(m1,n)
        # Primary negative-control and falsification use the matched source for direct readout.
        end0=row['final'][f'{i},0']['text'];end1=row['final'][f'{i},{i+1}']['text']
        r0,r1=parse(end0,n),parse(end1,n)
        q['trials']+=1
        if m0==m1:
            q['same_text']+=1;q['same_text_changed']+=r0!=r1
            q['same_text_final_string_changed']+=end0!=end1
        if min(b0,b1,r0,r1)<0:continue
        q['valid']+=1
        if b0!=b1:continue
        q['same_vote']+=1
        if m0==m1:continue
        q['same_vote_different_text']+=1;q['flips']+=r0!=r1
        q['corrections']+=r0!=y and r1==y;q['harms']+=r0==y and r1!=y
    return q


def evaluate(folder,natural_dir):
    inputs=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    def read(d,i):
        vals=list(map(json.loads,(d/f'WORKER{i}.jsonl').read_bytes().splitlines()))
        tab={r['key']:r for r in vals};assert len(tab)==len(vals)==len(inputs) and set(tab)==set(gold)
        return tab
    ws=[read(folder,i) for i in (0,1)];ns=[read(natural_dir,i) for i in (0,1)]
    result={};nresult={};unions={}
    tasks=sorted({q['task'] for q in inputs})
    for task in tasks:
        by_role=[]
        for reader in (0,1):
            rows=[];native=[];positive=set()
            for q in inputs:
                if q['task']!=task:continue
                k=q['key'];p=[np.asarray(w[k]['probs']) for w in ws]
                assert all(np.max(abs(z.sum(1)-1))<2e-6 and z.min()>=0 for z in p)
                p=[z/z.sum(1,keepdims=True) for z in p];cost=[w[k]['input_tokens'] for w in ws]
                for a in (0,1):rows.append(discrete(p,cost,gold[k],a,reader))
                native.append(natural(ns[reader][k],gold[k],len(q['options'])))
                if native[-1]['flips']:positive.add(k)
            stats={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            stats['question_role_pairs']=len(rows);result[f'{task}_{reader}']=stats
            nresult[f'{task}_{reader}']={k:sum(r[k] for r in native) for k in native[0]}
            nresult[f'{task}_{reader}']['support_questions']=len(positive);by_role.append(positive)
        unions[task]=len(set.union(*by_role))
    return dict(finite=result,natural=nresult,distinct_natural_support=unions,
                exemption_passed=all(r['all_certified']>=.3 for r in result.values()),
                refinement_frequency_passed=all(r['guarantee_lost']>=.05 for r in result.values()),
                natural_existence_passed=all(n>=5 for n in unions.values()) and all(r['support_questions']>=1 for r in nresult.values()))
