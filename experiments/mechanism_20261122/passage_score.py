import argparse,hashlib,json,re,string
from pathlib import Path
import numpy as np
from floor_decomposition import decomposition
from matched_stopping import stopping

def normalized(text):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in text.lower() if c not in string.punctuation)).split())
def mean(rows):return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
def aggregate(rows):
    totals=mean(rows);diff=totals['all_numeric']-totals['gold']
    totals.update(premium=totals['all_numeric']/totals['gold']-1,common_transcript_share=totals['simultaneous_extra']/diff if diff>0 else 0.,
                  joint_to_floor=totals['joint']/totals['all_numeric'],n=len(rows))
    return totals

def score(folder):
    qs=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes())
    ws=[]
    for i in range(3):
        rows=list(map(json.loads,(folder/f'WORKER{i}.jsonl').read_bytes().splitlines()));w={r['key']:r for r in rows}
        assert len(rows)==len(w)==len(qs)==len(gold)==128 and set(w)==set(gold);ws.append(w)
    results={}
    for a,b,r in [(0,1,2),(1,2,0),(2,0,1)]:
        by={tau:[] for tau in (.005,.01,.02)};reach=[];quality=[]
        for row in sorted(qs,key=lambda d:hashlib.sha256(('M54-block|'+d['key']).encode()).digest()):
            k=row['key'];data=[v[k] for v in ws];p=[np.asarray(v['probs']) for v in data]
            assert all(z.shape==(6,4) and abs(z.sum(1)-1).max()<2e-6 for z in p)
            p=[z/z.sum(1,keepdims=True) for z in p]
            source=p[a][0];base=p[b][1];m=p[b][2:];tcells=data[r]['terminal']
            raw=[normalized(v['text']) for v in tcells];labels=np.reshape([v if v not in ('','noevidence') else 'INVALID' for v in raw],(4,4))
            goldv=gold[k];assert goldv not in ('','noevidence','INVALID')
            costs=np.reshape([v['prefill']+v['decode'] for v in tcells],(4,4))
            prices=np.reshape([v['prefill']+32 for v in tcells],(4,4));rowcost=np.asarray(data[b]['prefix_tokens'][2:])+1
            init=data[a]['prefix_tokens'][0]+data[b]['prefix_tokens'][1]+2
            w=source[:,None]*(m-base);actual=labels==goldv
            credit=float(np.sum(w*actual));reach.append(bool(np.any(actual)))
            cats=sorted(set(labels.flat)-{'INVALID'})+['__UNSEEN__'];resp=(labels[:,:,None]==np.array(cats)[None,None,:]).astype(float)
            stopping_stats=stopping(source,base,m,resp,rowcost,costs,prices,cats.index(goldv) if goldv in cats else len(cats)-1,init)
            quality.append(dict(material=abs(credit)>.01,correct_on=np.sum(source[:,None]*m*actual),
              correct_off=np.sum(source[:,None]*base*actual),returned_evidence=np.mean(labels!='INVALID'),
              exact_span_in_provided_text=np.mean([bool(raw[i*4+j]) and raw[i*4+j] in normalized(row['sections'][i]+' '+row['sections'][j]) for i,j in np.ndindex(4,4)]),
              exhausted=np.mean([v['decode']==32 for v in tcells]),credit=credit,**stopping_stats))
            for tau in by:by[tau].append(decomposition(source,base,m,labels,rowcost,costs,prices,goldv,init,tau))
        primary=aggregate(by[.01]);service=mean(quality);n=sum(z['material'] for z in quality)
        results[f'{a}_{b}_{r}']=dict(thresholds={str(t):aggregate(rows) for t,rows in by.items()},service=service,
                blocks=[aggregate(by[.01][i:i+32]) for i in range(0,128,32)],material_questions=int(n),
                reached={str(state):aggregate([v for v,ok in zip(by[.01],reach) if state==ok]) for state in (False,True) if state in reach},
                gates=dict(premium=primary['premium']>=.15,compatibility=primary['common_transcript_share']>.5,
                           scheduler=primary['joint_to_floor']<=1.1,service=n>=8))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);print(json.dumps(score(p.parse_args().folder),indent=2))
