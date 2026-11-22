"""Conservative certificate transport for source-only graph edits."""
import numpy as np
from _mab2_blind_scheduler import Audit

TAU=0.01


def insert(audit, action, value):
    audit.update(action, value)


def source_radius(old_source, new_source, audit, base, label_count=8):
    radius=np.zeros(label_count+1)
    for i in range(len(old_source)):
        d=abs(float(new_source[i])-float(old_source[i]))
        if audit.known[i]:
            radius+=d
        else:
            radius+=d
    return radius


def intervals(audit, label_count=8):
    lo=np.zeros(label_count+1);hi=np.zeros(label_count+1)
    missing=audit.source[~audit.known].sum()*(1-audit.base.min())
    pending=np.where(audit.known[:,None]&~audit.seen,audit.w,0.)
    low=float(np.minimum(pending,0.).sum()-missing)
    high=float(np.maximum(pending,0.).sum()+missing)
    for k in range(label_count):
        lo[k]=audit.observed.get(k,0.)+low;hi[k]=audit.observed.get(k,0.)+high
    lo[label_count]=low;hi[label_count]=high
    return lo,hi


def safe_transport(old_source,new_source,audit,base,tau=TAU,label_count=8):
    lo,hi=intervals(audit,label_count);rad=source_radius(old_source,new_source,audit,base,label_count)
    keep=(lo-rad>tau+1e-12);drop=(hi+rad<tau-1e-12)
    return bool(np.all(keep|drop)),keep,drop,rad


def truth(source,base,conditional,labels,tau=TAU):
    weights=np.asarray(source)[:,None]*(np.asarray(conditional)-np.asarray(base))
    labels=np.asarray(labels)
    return np.array([weights[labels==y].sum()>tau for y in range(int(np.max(labels))+1)])


def ctdr(old_source,new_source,base,conditional,labels,history,tau=TAU):
    label_count=int(np.max(labels))+1
    audit=Audit(new_source,base,np.ones(4),np.ones((4,4)),'joint',tau=tau)
    for i,j,value in history:
        insert(audit,(i,j),value)
    safe,keep,drop,radius=safe_transport(old_source,new_source,audit,base,tau,label_count)
    spent=0
    while not safe and audit.decision() is None:
        action=audit.next();i,j=action
        value=conditional[i] if j is None else labels[i,j]
        insert(audit,action,value);spent+=1
        safe,keep,drop,radius=safe_transport(old_source,new_source,audit,base,tau,label_count)
    return audit,spent,safe,keep,drop,radius
