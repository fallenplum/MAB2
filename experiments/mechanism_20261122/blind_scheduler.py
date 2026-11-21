"""Audit a relay once for all possible exact-match labels, without reference answers."""
import numpy as np


class Audit:
    def __init__(self,source,base,row_prices,terminal_prices,policy,tau=.01):
        self.source=np.asarray(source,dtype=float);self.base=np.asarray(base,dtype=float)
        self.n=len(source);self.row_prices=np.asarray(row_prices);self.terminal_prices=np.asarray(terminal_prices)
        self.policy=policy;self.tau=tau;self.known=np.zeros(self.n,bool);self.seen=np.zeros((self.n,self.n),bool)
        self.w=np.zeros_like(self.seen,dtype=float);self.observed={}

    def update(self,action,result):
        i,j=action
        if j is None:
            assert not self.known[i];self.known[i]=True;self.w[i]=self.source[i]*(np.asarray(result)-self.base)
        else:
            assert self.known[i] and not self.seen[i,j];self.seen[i,j]=True
            self.observed[result]=self.observed.get(result,0.)+self.w[i,j]

    def decision(self):
        missing=self.source[~self.known].sum()*(1-self.base.min())
        pending=np.where(self.known[:,None]&~self.seen,self.w,0.)
        low=np.minimum(pending,0.).sum()-missing;high=np.maximum(pending,0.).sum()+missing
        values=list(self.observed.values())
        if self.seen.all():return {y for y,v in self.observed.items() if v>self.tau}
        # Every as-yet unseen output has the same zero observed mass.
        if high>=self.tau-1e-12:return None
        if any(not (v+high<self.tau-1e-12 or v+low>self.tau+1e-12) for v in values):return None
        return {y for y,v in self.observed.items() if v+low>self.tau+1e-12}

    def next(self):
        rows=np.flatnonzero(~self.known);cells=np.argwhere(self.known[:,None]&~self.seen)
        if self.policy=='row_first' and len(rows):return int(rows[np.argmax(self.source[rows])]),None
        if self.policy=='nested':
            i=next(int(i) for i in np.argsort(-self.source,kind='stable') if not self.seen[i].all())
            if not self.known[i]:return i,None
            eligible=np.flatnonzero(~self.seen[i]);rank=abs(self.w[i,eligible])/self.terminal_prices[i,eligible]
            return i,int(eligible[np.argmax(rank)])
        rwidth=2*self.source[rows]*(1-self.base.min())
        cwidth=abs(self.w[tuple(cells.T)]) if len(cells) else np.array([])
        if self.policy!='unpriced':
            rwidth=rwidth/self.row_prices[rows]
            if len(cells):cwidth=cwidth/self.terminal_prices[tuple(cells.T)]
        if len(rows) and (not len(cells) or max(rwidth)>=max(cwidth)):return int(rows[np.argmax(rwidth)]),None
        return tuple(map(int,cells[np.argmax(cwidth)]))


def exhaustive(source,base,conditional,labels,tau=.01):
    w=np.asarray(source)[:,None]*(np.asarray(conditional)-np.asarray(base))
    mass={}
    for i,j in np.ndindex(w.shape):
        label=labels[i][j];mass[label]=mass.get(label,0.)+w[i,j]
    return {y for y,v in mass.items() if v>tau}
