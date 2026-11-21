import itertools
import numpy as np
PERMS=list(itertools.permutations(range(3)));ORDERS=[o for n in range(4) for o in itertools.permutations(range(3),n)]
def score_question(p,y):
    tab=dict(zip(ORDERS,p[:,y]));records=[]
    for h in range(3):
        a,b=[v for v in range(3) if v!=h];dm=[];dn=[];outcomes=[]
        for order in PERMS:
            def erase(*ids):return tuple(i for i in order if i not in ids)
            full=tab[order];new=tab[erase(h)]
            keep_a=tab[erase(b)];keep_b=tab[erase(a)]
            old_margin=keep_a-keep_b;new_margin=tab[(a,)]-tab[(b,)]
            assert abs((full-keep_b)-(full-keep_a)-old_margin)<1e-12
            assert abs((new-tab[(b,)])-(new-tab[(a,)])-new_margin)<1e-12
            dm.append(old_margin);dn.append(new_margin);outcomes.append((keep_a,keep_b,full))
        dm=np.array(dm);dn=np.array(dn);assert np.ptp(dn)<1e-12
        rev=(abs(dm)>.01)&(abs(dn)>.01)&(dm*dn<0);same=(abs(dm)>.01)&(abs(dn)>.01)&(dm*dn>0)
        outcomes=np.array(outcomes);chosen=np.argmax(outcomes[0,:2]);regret=np.amax(outcomes[:,:2],axis=1)-outcomes[:,chosen]
        margin_ranges=np.ptp(outcomes[:,:2],axis=0)
        records.append(dict(ambiguous=bool(any(rev) and any(same)),always_flip=bool(all(rev)),never_observed_flip=bool(not any(rev)),remaining_boundary=bool(any(rev) and not any(same) and not all(rev)),canonical_flip=bool(rev[0]),canonical_worst_regret=float(max(regret)),canonical_mean_regret=float(np.mean(regret)),keep_menu_changed=len(set(np.argmax(outcomes,axis=1)))>1,full_order_range=float(np.ptp(outcomes[:,2])),deleted_order_range=float(max(margin_ranges)),margin_range=float(np.ptp(dm)),new_margin_range=float(np.ptp(dn)),canonical_tie=bool(abs(dm[0])<=.01)))
    return records
def evaluate(inputs,labels,workers):
    out={}
    for role,w in enumerate(workers):
        for task in ('arc','obqa'):
            grouped=[]
            for item in inputs:
                if item['task']!=task:continue
                k=item['key'];p=np.array(w[k]['probs']);assert p.shape==(16,4) and np.max(abs(p.sum(1)-1))<1e-6
                grouped.append(score_question(p/p.sum(axis=1,keepdims=True),labels[k]))
            flat=sum(grouped,[]);stats={k:np.mean([x[k] for x in flat]).item() for k in flat[0]}
            stats['support_questions']=sum(any(x['ambiguous'] for x in rows) for rows in grouped)
            stats['pairs']=len(flat);stats['questions']=len(grouped);stats['passed']=bool(stats['ambiguous']>=.1 and stats['support_questions']>=4 and stats['canonical_worst_regret']>.002);out[f'{task}_{role}']=stats
    return out
