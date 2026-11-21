"""Scalar audit using published decisions, the private request registry and the frozen outcome grid."""
import hashlib,json
from collections import Counter
import numpy as np

def tag(x):return int(hashlib.sha256(x.encode()).hexdigest(),16)
def audit(root,report):
    entpath=root/'d16_crossed_input_v1/ENTRIES.json';records=json.loads(entpath.read_bytes());raw=np.load(root/'d16_crossed_collection_v1/scoring_v1/PRIVATE_ARRAYS.npz');n=0;count=Counter()
    def check(a,b):
        nonlocal n
        assert abs(float(a)-float(b))<1e-7,(a,b);n+=1
    for task,settings in report['tasks'].items():
        e=raw[task+'_em'];tokens=raw[task+'_tokens'];shock=raw[task+'_shock'];ss=[26111671,26111672];ts=ss+[261116161,261116162]
        rec={(x['index'],ss.index(x['source_seed']),ts.index(x['terminal_seed']),x['mask']):x for x in records if x['task']==task};check(len(rec),6144)
        ordered=sorted(range(96),key=lambda q:tag('D35-group-v1|'+str(q)))
        for block in settings['fold_results']:
            seed=block['seed'];fold=block['fold'];service=ordered[fold::4];check(len(service),24);hist=[q for q in ordered if q not in service];panel={h:sorted([q for q in hist if shock[q]==h],key=lambda q:tag('D36-panel-v1|'+str(q)))[:4] for h in range(3)}
            all_out={}
            for method,observed in block['methods'].items():
                rng=np.random.default_rng(seed+1000*fold);init=extra=0;pairs=0;known=method in ['graph_transport','uniform','periodic','uncertainty','graph_no_correction','old_only','proxy_only'];pays=known and method not in ['old_only','proxy_only'];ind_masks={};ninit=nextra=0
                for h,ids in panel.items():
                    current=7^(1<<h);edges=[i for i in range(3) if i!=h];masks=[current]+[current^(1<<i) for i in edges]+[0];prior=[7]+[7^(1<<i) for i in edges]+[0];old_set=(0,3,5,6,7);oldvalues=[];signals=[];truth=[];extras=[];ecalls=[];weights=[]
                    for q in ids:
                        s=tag(f'D36-src|{seed}|{q}')%2;t=tag(f'D36-term|{seed}|{q}')%4;meta=[rec[q,s,t,m] for m in range(8)];v=e[q,s,t];tk=tokens[q,s,t];oldvalues.append([v[m]-.000005*tk[m] for m in prior]);truth.append([v[m]-.000005*tk[m] for m in masks]);seen={meta[m]['job_id']:m for m in old_set};cost=lambda m:tk[m]-meta[m]['prefix_tokens'];additional={meta[m]['job_id']:m for m in masks if meta[m]['job_id'] not in seen};ecalls.append(len(additional));extras.append(sum(cost(m) for m in additional.values()))
                        if known:init+=sum(meta[1<<i]['prefix_tokens'] for i in range(3))+sum(cost(m) for m in seen.values());ninit+=3+len(seen)
                        xs=[]
                        for m,pm in zip(masks,prior):
                            idx=seen.get(meta[m]['job_id']);correct=v[idx] if idx is not None else min(1,max(0,v[pm]+v[current]-v[7]));p=meta[m]['prefix_tokens']+cost(idx if idx is not None else pm);xs.append(correct-.000005*p)
                        signals.append(xs)
                        if method in ('graph_transport','graph_no_correction'):weights.append(1+abs(v[7]-v[current])+sum(abs(v[z]-v[7]) for z in prior)/4)
                        elif method=='uncertainty':weights.append(1+2*np.std([v[z] for z in prior]))
                        else:weights.append(1)
                    if pays:
                        pmf=np.array(weights)/sum(weights);j=(seed+fold+h)%4 if method=='periodic' else int(rng.choice(4,p=pmf));extra+=extras[j];nextra+=ecalls[j];pairs+=1
                        values=truth[j] if method=='graph_no_correction' else [sum(row[k] for row in signals)/4+(truth[j][k]-signals[j][k])/(4*pmf[j]) for k in range(4)];chosen=masks[int(np.argmax(values))]
                    elif known:chosen=masks[int(np.argmax(np.mean(oldvalues if method=='old_only' else signals,axis=0)))]
                    elif method=='no_init_keep':chosen=current
                    elif method=='no_init_single':chosen=0
                    else:chosen=current&~(1<<int(method[-1]))
                    check(chosen,observed['chosen_masks_by_shock'][str(h)]);ind_masks[h]=chosen
                final=[];costs=[]
                for q in service:
                    m=ind_masks[int(shock[q])];final.extend(e[q,:,:,m].flatten());costs.extend(tokens[q,:,:,m].flatten())
                all_out[method]=np.array(final);svc=sum(costs)/8;total=init+extra+svc;check(init,observed['initial_tokens']);check(extra,observed['probe_tokens']);check(ninit,observed['initial_calls']);check(nextra,observed['uncached_probe_calls']);check(pairs,observed['rechecked_questions']);check(svc,observed['service_tokens']);check(total,observed['full_tokens']);check(sum(final)/len(final),observed['EM']);check(sum(final)/len(final)-.000005*total/24,observed['full_utility']);count['policy_fold_rows']+=1
            for m,out in all_out.items():
                diff=all_out['graph_transport']-out;r=block['paired'][m];check(sum(diff>0),r['corrected']);check(sum(diff<0),r['harmed']);check(sum(diff==0),r['ties'])
    return dict(status='PASS_INDEPENDENT_SCALAR_REPLAY',checks=n,**count,scope='reconstructed purchases, choices, unique request charges, service accuracy and paired outcomes; no fit or independent new deployment')
