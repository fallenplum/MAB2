"""Finite-budget counterfactual acquisition from a disjoint old-question panel."""
import hashlib,itertools,json
from collections import Counter
import numpy as np

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def folds(n):
    ix=sorted(range(n),key=lambda i:digest(f'D35-group-v1|{i}'));f=np.empty(n,int);f[ix]=np.arange(n)%4;return f
def predict_proxy(w,full,base,old_action,current):
    pred=np.clip(w[old_action]+w[base]-w[full],0,1)
    return np.where(np.asarray(current)==0,w[0],np.where(np.asarray(current)==base,w[base],pred))

def score_task(em,tok,shock,items):
    assert em.shape==tok.shape==(96,2,4,8);ss=sorted({r['source_seed'] for r in items});tt=sorted({r['terminal_seed'] for r in items},key=lambda x:(x not in ss,x));assert len(ss)==2 and len(tt)==4
    # The D16 source defines this non-numeric order as the original repeated pair followed by two new draws.
    assert tt==[26111671,26111672,261116161,261116162]
    lookup={(r['index'],ss.index(r['source_seed']),tt.index(r['terminal_seed']),r['mask']):r for r in items};assert len(lookup)==6144
    fold=folds(96);out=[];price=.000005;cap=800000
    for seed in (17,31,47):
        for k in range(4):
            test=np.where(fold==k)[0];pool=np.where(fold!=k)[0];panel={};records=[]
            def construct(i,h):
                b=7^(1<<h);s=int(digest(f'D36-src|{seed}|{i}'),16)%2;t=int(digest(f'D36-term|{seed}|{i}'),16)%4
                es=[p for p in range(3) if p!=h];actions=[b,b^(1<<es[0]),b^(1<<es[1]),0];old=[7,7^(1<<es[0]),7^(1<<es[1]),0]
                entries=[lookup[i,s,t,m] for m in range(8)];prefix=np.array([e['prefix_tokens'] for e in entries]);assert all(prefix[m]==sum(prefix[1<<e] for e in range(3) if m&(1<<e)) for m in range(8))
                terminal=tok[i,s,t]-prefix;assert np.all(terminal>0)
                ids=[e['job_id'] for e in entries];frozen=[0,3,5,6,7];oldset={ids[m]:terminal[m] for m in frozen};assert all(oldset[ids[m]]==terminal[m] for m in frozen)
                # All three old sources were purchased. Future counterfactual terminal calls only add exact uncached requests.
                init=int(prefix[7]+sum(oldset.values()));extra={ids[m]:terminal[m] for m in actions if ids[m] not in oldset};fee=int(sum(extra.values()))
                w=em[i,s,t];pred=predict_proxy(w,7,b,old,actions)
                # Price quotes are observable: use the old terminals with the corresponding original edits, plus exact current source-token counts.
                quote=np.array([tok[i,s,t,b],prefix[actions[1]]+terminal[old[1]],prefix[actions[2]]+terminal[old[2]],tok[i,s,t,0]])
                for j,m in enumerate(actions):
                    match=next((a for a in frozen if ids[a]==ids[m]),None)
                    if match is not None:
                        assert w[match]==w[m] and terminal[m]==terminal[match]
                        pred[j]=w[match];quote[j]=prefix[m]+terminal[match]
                proxy=pred-price*quote; obs=w[actions]-price*tok[i,s,t,actions]; prior=w[old]-price*tok[i,s,t,old]
                risk=float(abs(w[7]-w[b])+np.mean(abs(w[old]-w[7]))); disagreement=float(np.std(w[old]));
                return dict(i=i,proxy=proxy,obs=obs,prior=prior,risk=risk,spread=disagreement,initial=init,extra=fee,extra_calls=len(extra),initial_calls=len(oldset)+3,actions=actions)
            for h in range(3):
                candidates=sorted((int(i) for i in pool if shock[i]==h),key=lambda i:digest(f'D36-panel-v1|{i}'));assert len(candidates)>=4
                panel[h]=[construct(i,h) for i in candidates[:4]];records+=panel[h]
            init=sum(r['initial'] for r in records)
            names=('graph_transport','uniform','periodic','uncertainty','graph_no_correction','old_only','proxy_only','no_init_keep','no_init_single','fixed_delete_0','fixed_delete_1','fixed_delete_2')
            results={};answers={}
            for method in names:
                uses_init=method in names[:7];over=init if uses_init else 0;spent=0;asks=0;trm=0;actionmap={}
                rng=np.random.default_rng(seed+k*1000)
                for h,rr in panel.items():
                    act=rr[0]['actions'];base=act[0]
                    if not uses_init:
                        if method.startswith('fixed_delete_'):m=base&~(1<<int(method[-1]))
                        else:m=0 if method=='no_init_single' else base
                    elif method in ('old_only','proxy_only'):
                        col='prior' if method=='old_only' else 'proxy';m=act[int(np.mean([r[col] for r in rr],axis=0).argmax())]
                    else:
                        if method in ('graph_transport','graph_no_correction'):weights=np.array([1+r['risk'] for r in rr])
                        elif method=='uncertainty':weights=np.array([1+2*r['spread'] for r in rr])
                        else:weights=np.ones(4)
                        probs=weights/weights.sum();j=(seed+k+h)%4 if method=='periodic' else int(rng.choice(4,p=probs));r=rr[j];avg=np.mean([u['proxy'] for u in rr],axis=0)
                        corrected=avg+(r['obs']-r['proxy'])/(4*probs[j]);m=act[int((r['obs'] if method=='graph_no_correction' else corrected).argmax())]
                        spent+=r['extra'];trm+=r['extra_calls'];asks+=1
                    actionmap[h]=m
                mask=np.array([actionmap[int(shock[i])] for i in test]);er=em[test,:,:,mask];service_tokens=tok[test,:,:,mask].mean(axis=(1,2)).sum();total=over+spent+service_tokens;assert total<=cap
                answers[method]=er;results[method]=dict(EM=float(er.mean()),service_tokens=float(service_tokens),initial_tokens=over,probe_tokens=spent,full_tokens=float(total),unused_token_cap=float(cap-total),historical_questions=12 if uses_init else 0,rechecked_questions=asks,additional_gold_questions=0,initial_calls=sum(r['initial_calls'] for r in records) if uses_init else 0,uncached_probe_calls=trm,service_utility=float(er.mean()-price*service_tokens/len(test)),full_utility=float(er.mean()-price*total/len(test)),chosen_masks_by_shock=actionmap)
            pairs={m:{'corrected':int(((answers['graph_transport']==1)&(a==0)).sum()),'harmed':int(((answers['graph_transport']==0)&(a==1)).sum()),'ties':int((answers['graph_transport']==a).sum())} for m,a in answers.items()}
            out.append(dict(seed=seed,fold=k,service_questions=len(test),methods=results,paired=pairs))
    ag={};checks=[]
    for seed in (17,31,47):
        rr=[r for r in out if r['seed']==seed];ag[seed]={m:{v:float(np.mean([r['methods'][m][v] for r in rr])) for v in ('EM','full_utility','service_utility','full_tokens','probe_tokens')} for m in out[0]['methods']}
        vals=ag[seed];u=vals['graph_transport']['full_utility'];checks.append(dict(seed=seed,matched_delta=u-max(vals[m]['full_utility'] for m in ('uniform','periodic','uncertainty','graph_no_correction')),cheap_delta=u-max(vals[m]['full_utility'] for m in ('old_only','proxy_only','no_init_keep','no_init_single','fixed_delete_0','fixed_delete_1','fixed_delete_2'))))
    return dict(fold_results=out,summary=ag,comparisons=checks,passed=all(r['matched_delta']>=.005 and r['cheap_delta']>=0 for r in checks),token_cap_per_episode=cap,groups=96,service_groups_per_fold=24)
