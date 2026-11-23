"""Bit-mask acceleration of the bounded horizon-2 Graph-VOI policy."""
from __future__ import annotations
import itertools
from functools import lru_cache
from typing import Any
import numpy as np
from strict_graph_voi_cpu import aggregate, certificate, valid_cached_terminals

def run_h2_fast(case: dict[str, Any], horizon: int = 2) -> dict[str, Any]:
    ns = case["source_count"]; nt = case["terminal_count"]
    assignments = list(itertools.product((0,1), repeat=ns)); all_mask=(1<<len(assignments))-1
    source_masks=[[0,0] for _ in range(ns)]
    for ai,assignment in enumerate(assignments):
        for source,value in enumerate(assignment): source_masks[source][value]|=1<<ai
    def label(assignment,terminal): return int((sum((i+1)*assignment[i] for i in case["deps"][terminal])+terminal+case["salt"])%3)
    label_masks=[[0,0,0] for _ in range(nt)]
    for ai,assignment in enumerate(assignments):
        bit=1<<ai
        for terminal in range(nt): label_masks[terminal][label(assignment,terminal)] |= bit
    label_orders=[]
    for terminal in range(nt):
        seen=[]
        for assignment in assignments:
            value=label(assignment,terminal)
            if value not in seen: seen.append(value)
        label_orders.append(seen)
    def possible(src,term):
        mask=all_mask
        for source,value in enumerate(src):
            if value>=0: mask &= source_masks[source][value]
        for terminal,value in enumerate(term):
            if value>=0: mask &= label_masks[terminal][value]
        return mask
    @lru_cache(None)
    def cached(src):
        observed={i:v for i,v in enumerate(src) if v>=0}; return tuple(valid_cached_terminals(case,observed).items())
    def known(src,term):
        result=dict(cached(src)); result.update({i:value for i,value in enumerate(term) if value>=0}); return result
    def branches(mask,kind,index):
        if kind=='source': values=(0,1); masks=source_masks[index]
        else: values=label_orders[index]; masks=label_masks[index]
        return [(value,mask&masks[value]) for value in values if mask&masks[value]]
    def finish_cost(assignment,src,term):
        current=known(src,term); safe,_=certificate(current,case['weights'],nt)
        if safe:return 0.0
        cached_cells=set(valid_cached_terminals(case,{i:v for i,v in enumerate(src) if v>=0}))
        order=sorted((i for i in range(nt) if i not in current),key=lambda i:(case['weights'][i]/max(case['costs'][ns+i],1e-9),i),reverse=True)
        cost=0.0
        for terminal in order:
            current[terminal]=label(assignment,terminal)
            if terminal not in cached_cells: cost+=float(case['costs'][ns+terminal])
            safe,_=certificate(current,case['weights'],nt)
            if safe:break
        return cost
    def leaf_value(src,term):
        current=known(src,term);safe,_=certificate(current,case['weights'],nt)
        if safe:return 0.0
        mask=possible(src,term)
        if not mask:return float('inf')
        source_cost=sum(float(case['costs'][i]) for i,value in enumerate(src) if value<0)
        total=mask.bit_count();completion=[]
        for ai,assignment in enumerate(assignments):
            if mask&(1<<ai): completion.append(finish_cost(assignment,src,term))
        return source_cost+float(np.mean(completion))
    @lru_cache(None)
    def rollout(src,term,depth):
        current=known(src,term);safe,_=certificate(current,case['weights'],nt)
        if safe:return 0.0
        if depth<=0:return leaf_value(src,term)
        mask=possible(src,term)
        if not mask:return float('inf')
        best=float('inf')
        for source in range(ns):
            if src[source]>=0:continue
            value=float(case['costs'][source]); total=mask.bit_count()
            for bit_value,branch_mask in branches(mask,'source',source):
                src_next=list(src);src_next[source]=bit_value;value+=branch_mask.bit_count()/total*rollout(tuple(src_next),term,depth-1)
            best=min(best,value)
        for terminal in range(nt):
            if term[terminal]>=0 or terminal in current or not all(src[i]>=0 for i in case['deps'][terminal]):continue
            value=float(case['costs'][ns+terminal]); total=mask.bit_count()
            for label_value,branch_mask in branches(mask,'terminal',terminal):
                term_next=list(term);term_next[terminal]=label_value;value+=branch_mask.bit_count()/total*rollout(src,tuple(term_next),depth-1)
            best=min(best,value)
        return best
    target=aggregate(case['new_labels'],case['weights']);src=tuple([-1]*ns);term=tuple([-1]*nt);cost=0.;queries=0
    while True:
        current=known(src,term);safe,choice=certificate(current,case['weights'],nt)
        if safe:return {'action':choice,'target':target,'cost':cost,'queries':queries,'certificate':True}
        mask=possible(src,term)
        if not mask:return {'action':target,'target':target,'cost':cost,'queries':queries,'certificate':False}
        total=mask.bit_count();options=[]
        for source in range(ns):
            if src[source]>=0:continue
            value=float(case['costs'][source])
            for bit_value,branch_mask in branches(mask,'source',source):
                src_next=list(src);src_next[source]=bit_value;value+=branch_mask.bit_count()/total*rollout(tuple(src_next),term,horizon-1)
            options.append((value,'source',source))
        for terminal in range(nt):
            if term[terminal]>=0 or terminal in current or not all(src[i]>=0 for i in case['deps'][terminal]):continue
            value=float(case['costs'][ns+terminal])
            for label_value,branch_mask in branches(mask,'terminal',terminal):
                term_next=list(term);term_next[terminal]=label_value;value+=branch_mask.bit_count()/total*rollout(src,tuple(term_next),horizon-1)
            options.append((value,'terminal',terminal))
        if not options:return {'action':target,'target':target,'cost':cost,'queries':queries,'certificate':False}
        _,kind,index=min(options,key=lambda item:(item[0],item[1],item[2]))
        if kind=='source':
            src_next=list(src);src_next[index]=case['new'][index];src=tuple(src_next);cost+=float(case['costs'][index])
        else:
            term_next=list(term);term_next[index]=case['new_labels'][index];term=tuple(term_next);cost+=float(case['costs'][ns+index])
        queries+=1

