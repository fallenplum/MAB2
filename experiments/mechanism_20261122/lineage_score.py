"""Paired source-lineage contrasts. The question is the unit of analysis."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

START=re.compile(r"^\s*(?:(?:my\s+(?:final\s+)?(?:choice|answer)|(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|option|choice))\s*(?:is\s*)?[:=\-]?\s*)?(?:option\s+)?[\(\[\*\s]*([A-E])(?=$|[\s\]\)\.,:;!\-*])",re.I)
def parse(text,n):
    match=START.match(text)
    if match and match.group(1).upper()=='A' and match.start(1)==len(text)-len(text.lstrip()) and re.match(r"\s+[A-Za-z]",text[match.end():]) and not re.match(r"\s+(?:because|is|since|as)\b",text[match.end():],re.I):
        return -1
    return ord(match.group(1).upper())-65 if match and match.group(1).upper() in 'ABCDE'[:n] else -1


def question(row,y,n):
    src=row['source'];mid=row['middle'];end=row['final']
    assert len(src)==2 and len(mid)==3 and len(end)==9
    def value(i,b):return parse(end[f'{i},{b}']['text'],n)
    result={'pairs':2,'supported':0,'change_same':0,'change_cross':0,'change_last':0,
            'last_support':0,'same_on_ok':0,'cross_on_ok':0,'same_off_ok':0,'cross_off_ok':0,
            'last_off_ok':0,'last_on_ok':0,'equal_sender_text':int(src[0]['text']==src[1]['text']),
            'unparsed':sum(parse(v['text'],n)<0 for v in end.values())}
    for name in ('same_on','same_off','cross_on','cross_off','last_on','last_off','collected'):
        for c in ('calls','prefill','decode'):result[f'{name}_{c}']=0
    def count(name,events):
        result[name+'_calls']+=len(events)
        for c in ('prefill','decode'):result[name+'_'+c]+=sum(r[c] for r in events)
    count('collected',[*src,*mid,*end.values()])
    for i in range(2):
        j=1-i
        a,b,c,d=value(i,i+1),value(i,0),value(j,i+1),value(j,0)
        if min(a,b,c,d)>=0:
            result['supported']+=1;result['change_same']+=a!=b;result['change_cross']+=c!=d
        e,f=value(-1,i+1),value(-1,0)
        if min(e,f)>=0:result['last_support']+=1;result['change_last']+=e!=f
        for name,choice in [('same_on',a),('same_off',b),('cross_on',c),('cross_off',d),('last_on',e),('last_off',f)]:
            result[name+'_ok']+=choice==y
        count('same_on',[src[i],mid[i+1],end[f'{i},{i+1}']])
        count('cross_on',[src[i],src[j],mid[i+1],end[f'{j},{i+1}']])
        count('same_off',[src[i],mid[0],end[f'{i},0']])
        count('cross_off',[src[j],mid[0],end[f'{j},0']])
        count('last_on',[src[i],mid[i+1],end[f'-1,{i+1}']])
        count('last_off',[mid[0],end['-1,0']])
    assert result['cross_off_ok']==result['same_off_ok']
    result['credit_changed_q']=int(result['same_on_ok']!=result['cross_on_ok'])
    result['credit_up_q']=int(result['same_on_ok']>result['cross_on_ok'])
    result['credit_down_q']=int(result['same_on_ok']<result['cross_on_ok'])
    return result


def summarize(rows):
    s={k:int(sum(r[k] for r in rows)) for k in rows[0]}
    div=lambda x,y: x/y if y else None
    return dict(questions=len(rows),counts=s,support=s['supported']/s['pairs'],
                D=div(s['change_cross']-s['change_same'],s['supported']),
                same_change=div(s['change_same'],s['supported']),cross_change=div(s['change_cross'],s['supported']),
                last_change=div(s['change_last'],s['last_support']),
                credit_changed_fraction=s['credit_changed_q']/len(rows),
                signed_quality={k:s[k+'_ok']/s['pairs'] for k in ('same_on','same_off','cross_on','cross_off','last_on','last_off')},
                unparsed=s['unparsed']/(len(rows)*9))


def score(folder):
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    assert len(data)==len(gold)==256
    output={};tasks=sorted({r['task'] for r in data});changed={task:[] for task in tasks}
    for role in (0,1):
        raw=list(map(json.loads,(folder/f'WORKER{role}.jsonl').read_bytes().splitlines()))
        records={r['key']:r for r in raw};assert len(raw)==len(records)==256 and set(records)==set(gold)
        for task in tasks:
            items=sorted([r for r in data if r['task']==task],key=lambda r:hashlib.sha256(('M43-block|'+r['key']).encode()).digest())
            assert len(items)==128
            rows=[question(records[r['key']],gold[r['key']],len(r['options'])) for r in items]
            changed[task].append({r['key'] for r,o in zip(items,rows) if o['credit_changed_q']})
            result=summarize(rows)
            result['blocks']=[{k:v for k,v in summarize(rows[i:i+32]).items() if k!='counts'} for i in range(0,128,32)]
            output[f'{task}_{role}']=result
    passed=True
    for task in tasks:
        both=[output[f'{task}_{i}'] for i in (0,1)]
        passed &= all(r['support']>=.9 and r['D'] is not None and r['D']>0 for r in both)
        passed &= all(r['D'] is not None for r in both) and np.mean([r['D'] for r in both])>=.02
    unique={task:dict(either=len(sets[0]|sets[1]),both=len(sets[0]&sets[1]),n=128) for task,sets in changed.items()}
    return dict(cells=output,transmission_gate=bool(passed),distinct_changed_questions=unique,
                quality_support_gate=all(s['either']/s['n']>=.05 for s in unique.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_folder',type=Path)
    print(json.dumps(score(p.parse_args().run_folder),indent=2))
