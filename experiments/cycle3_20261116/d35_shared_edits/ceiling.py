"""Prospectively scoped upper bound for a service edit conditioned only on the graph shock."""
from collections import Counter
import hashlib
import json
import numpy as np

def fold_ids(n):
    order=sorted(range(n),key=lambda i:hashlib.sha256(f'D35-group-v1|{i}'.encode()).hexdigest());f=np.empty(n,int);f[order]=np.arange(n)%4;return f

def evaluate(em,tok,shock):
    assert em.shape==tok.shape==(96,2,4,8);n=len(em); idx=np.arange(n);base=7^(1<<shock)
    # Four fixed deletion policies include the no-op if their edge is already absent; fifth is single-solver fallback.
    masks=np.column_stack([base]+[base&~(1<<k) for k in range(3)]+[np.zeros(n,int)])
    assert not ((masks & base[:,None])!=masks).any();f=fold_ids(n)
    y=em.mean(axis=(1,2));tokens=tok.mean(axis=(1,2));q=y[idx[:,None],masks];price=tokens[idx[:,None],masks];u=q-5e-6*price
    oldmask=np.column_stack([np.full(n,7)]+[np.where(shock==k,7,7^(1<<k)) for k in range(3)]+[np.zeros(n,int)])
    oldu=(y-5e-6*tokens)[idx[:,None],oldmask]
    actions={k:np.zeros(n,int) for k in ('keep','no_note','old_global','old_by_shock','train_current_by_shock','hindsight_by_shock','hindsight_per_question')};actions['no_note'][:]=4
    actions['hindsight_per_question']=u.argmax(1);split=[]
    for fold in range(4):
        tr=f!=fold;ev=~tr;actions['old_global'][ev]=int(oldu[tr].mean(axis=0).argmax())
        for h in range(3):
            a=tr&(shock==h);b=ev&(shock==h);assert a.any() and b.any()
            actions['old_by_shock'][b]=int(oldu[a].mean(axis=0).argmax());actions['train_current_by_shock'][b]=int(u[a].mean(axis=0).argmax());actions['hindsight_by_shock'][b]=int(u[b].mean(axis=0).argmax())
        split.append({'groups':int(ev.sum()),'shock_counts':np.bincount(shock[ev],minlength=3).tolist()})
    out={}
    for name,act in actions.items():
        acc=q[idx,act];ut=u[idx,act];tt=price[idx,act];ref=actions['old_by_shock'];c=em[idx,:,:,masks[idx,act]];r=em[idx,:,:,masks[idx,ref]]
        out[name]={'EM':float(acc.mean()),'utility':float(ut.mean()),'service_tokens':float(tt.mean()),'delta_by_fold_vs_old_by_shock':[float((ut-u[idx,ref])[f==k].mean()) for k in range(4)],'corrections_vs_old_by_shock':int(((c==1)&(r==0)).sum()),'harms_vs_old_by_shock':int(((c==0)&(r==1)).sum()),'source_terminal_observations':c.size}
    current=out['hindsight_by_shock']['utility'];benchmark=max(out[k]['utility'] for k in ('keep','no_note','old_global','old_by_shock'))
    # Low-cost controls using train labels have intentionally waived charges on both sides here.
    return dict(folds=split,methods=out,oracle_delta_vs_strongest_cheap=current-benchmark,passed=bool(current-benchmark>=.005 and min(out['hindsight_by_shock']['delta_by_fold_vs_old_by_shock'])>0))
