"""Frozen retrieval-only experiment. Rankers see question/paragraph text only."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import time

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def years(s):return [int(x) for x in re.findall(r'\b(?:1[0-9]{3}|20[0-2][0-9])\b',s)]
def tokens(s):return re.findall(r'[\w]+',s.casefold())

def slices(paras,context):
    result=[]; start=0
    for entry in paras:
        text=entry['text']; pos=context.find(text,start)
        if pos<0 or not text.strip():return None
        result.append((pos,pos+len(text)));start=pos+len(text)
    return result

def evidence(row):
    ranges=slices(row['paragraphs'],row['context']); hits=set()
    if not ranges:return hits
    for start,end in zip(row['from'],row['end']):
        if row['context'][start:end].casefold().strip() not in {a.casefold().strip() for a in row['targets']}:continue
        hits.update(i for i,(a,b) in enumerate(ranges) if a<=start<end<=b)
    return hits

def pairs(rows):
    groups=defaultdict(list)
    for r in rows:
        group=r['idx'].rsplit('#',1)[0]; ys=years(r['question'])
        if ys:groups[group].append((min(ys),max(ys),r))
    options=defaultdict(list)
    for group,rr in groups.items():
        for lo,hi,a in rr:
            for low,high,b in rr:
                if hi>=low or tuple(x['text'] for x in a['paragraphs'])!=tuple(x['text'] for x in b['paragraphs']):continue
                options[group.rsplit('#',1)[0]].append((low-hi,digest(a['idx']+'|'+b['idx']),a,b))
    # This first article was used for schema and offset qualification.
    options.pop(rows[0]['idx'].split('#')[0],None)
    ordered=sorted(options,key=lambda k:digest('MAB2-D32-article-v1|'+k)); selected=[]
    for k in ordered[:64]:
        _,_,a,b=min(options[k],key=lambda x:x[:2]);selected.append((digest(k),a,b))
    return selected,len(ordered)

def rank_scores(question,texts):
    query=set(tokens(question)); qt=years(question); assert qt
    doc=[Counter(tokens(p)) for p in texts]; avg=sum(map(lambda v:sum(v.values()),doc))/len(doc)
    df=Counter(w for d in doc for w in d if w in query);n=len(texts)
    lex=[];dt=[]
    for d,p in zip(doc,texts):
        size=sum(d.values()); v=0.0
        for w in sorted(query):
            if d[w]:v+=math.log(1+(n-df[w]+0.5)/(df[w]+0.5))*d[w]*2.2/(d[w]+1.2*(.25+.75*size/avg))
        lex.append(v); py=years(p)
        dt.append(sum(1/(1+min(abs(q-y) for y in py)) for q in qt)/len(qt) if py else 0.)
    def norm(xs):
        m=max(xs);return [x/m if m else 0. for x in xs]
    lx=norm(lex);tm=norm(dt);joint=[l*t for l,t in zip(lx,tm)]
    systems={'LEX':lx,'DATE':tm,'JOINT':joint,'old_drop':lx,'old_keep':[l+t for l,t in zip(lx,tm)],
             'new_drop':[l+j for l,j in zip(lx,joint)],'new_keep':[l+j+t for l,j,t in zip(lx,joint,tm)]}
    pred={k:max(range(n),key=lambda i:score[i]) for k,score in systems.items()}
    s=sorted(systems['new_keep'],reverse=True);return pred,s[0]-s[1] if n>1 else 0.

def run(rows):
    start=time.perf_counter();selected,eligible=pairs(rows);obs=[];unscorable=0
    for family,a,b in selected:
        p1,m1=rank_scores(a['question'],[v['text'] for v in a['paragraphs']]);p2,m2=rank_scores(b['question'],[v['text'] for v in b['paragraphs']]);e1,e2=evidence(a),evidence(b)
        if not e1 or not e2:unscorable+=1;continue
        c=lambda p,g,s:int(p[g+'_keep'] in s)-int(p[g+'_drop'] in s)
        co=c(p1,'old',e1); c01=c(p1,'new',e1);c10=c(p2,'old',e2);cc=c(p2,'new',e2); keep=co>=0
        stale=int(p2['new_keep' if keep else 'new_drop'] in e2);best=max(int(p2['new_keep'] in e2),int(p2['new_drop'] in e2))
        # Graph-specific score is computed without any newer-query evidence labels.
        changed=sum(p2['old_'+t]!=p2['new_'+t] for t in ('keep','drop'))
        transferred=sum(p1['old_'+t]!=p2['new_'+t] for t in ('keep','drop'))
        obs.append(dict(key=family,scope=p2['new_keep']!=p2['new_drop'],g=changed,d=transferred,margin=m2,
                        old=co,matched_new=c01,matched_old=c10,current=cc,stale=stale,best=best,
                        static={k:int(p2[k] in e2) for k in p2},tie=digest('D32-tie|'+family)))
    valid=[i for i,r in enumerate(obs) if r['scope']];budget=min(len(valid),math.ceil(len(obs)/4));stale=sum(r['stale'] for r in obs)
    orders={'graph_change':lambda r:(-r['g'],-r['d'],r['margin'],r['tie']), 'time_change':lambda r:(-r['d'],r['margin'],r['tie']), 'uncertainty':lambda r:(r['margin'],r['tie'])}
    for seed in (17,31,47):orders['random_'+str(seed)]=lambda r,s=seed:(digest(str(s)+r['key']),)
    orders['periodic']=None; out={}
    for method,key in orders.items():
        if method=='periodic':
            pool=sorted(valid,key=lambda i:obs[i]['tie']);picked=[pool[k*len(pool)//budget] for k in range(budget)]
        else:picked=sorted(valid,key=lambda i:key(obs[i]))[:budget]
        assert len(set(picked))==budget
        gain=sum(obs[i]['best']-obs[i]['stale'] for i in picked)
        out[method]={'correct':stale+gain,'repairs':gain,'pair_acquisitions':budget}
    stat={k:sum(r['static'][k] for r in obs) for k in ('LEX','DATE','JOINT','new_keep','new_drop','old_keep','old_drop')}
    oracle=sum(r['best'] for r in obs); top=out['graph_change']['correct'];ctrl=max(v['correct'] for k,v in out.items() if k!='graph_change')
    shift=sum(r['current']!=r['matched_old'] for r in obs);joint=sum((r['current']-r['matched_old'])!=(r['matched_new']-r['old']) for r in obs)
    passed=bool(len(obs)>=32 and shift>=4 and top-ctrl>=2 and top>=max(stat.values()) and oracle-stale>=4)
    return dict(status='RETRIEVAL_SCREEN_PASSED_ONLY' if passed else 'NO_QUALIFIED_TEMPORAL_RETRIEVAL_MECHANISM',eligible_articles=eligible,selected_articles=len(selected),scorable_articles=len(obs),unscorable_articles=unscorable,scope_disagreement=len(valid),pair_budget=budget,
                old_paid_pairs=len(obs),static_correct=stat,stale_correct=stale,full_hindsight_correct=oracle,true_repairable_tasks=oracle-stale,
                topology_only_credit_changes_current_time=shift,time_only_credit_changes_old_topology=sum(r['matched_old']!=r['old'] for r in obs),interaction_count=joint,
                policies=out,CPU_wall_seconds=time.perf_counter()-start,GPU_hours=0,model_calls=0,unit='one family per article, annotated supporting paragraph at rank 1; no answer generation',independent_confirmation=False)
