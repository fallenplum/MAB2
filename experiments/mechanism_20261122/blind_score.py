import argparse,json
from decimal import Decimal
from pathlib import Path
import numpy as np
from blind_scheduler import Audit,exhaustive
from table_score import parse
METHODS=('full','joint','row_first')

def totals(arm,lengths):
    pre=arm['prefix'];terminal=arm['terminal']
    return sum(lengths[0][int(i)]+1 for i in pre)+sum(v['prefill']+v['decode'] for v in terminal.values())


def replay(full,lengths,method):
    pre=full['prefix'];ter=full['terminal']
    aud=Audit(pre['0'],pre['1'],np.asarray(lengths[0][2:])+1,np.reshape(lengths[1],(4,4))+512,method)
    spent=lengths[0][0]+lengths[0][1]+2
    if method=='eager':
        for i in range(4):aud.update((i,None),pre[str(i+2)]);spent+=lengths[0][i+2]+1
    while aud.decision() is None:
        i,j=aud.next()
        if j is None:
            aud.update((i,j),pre[str(i+2)]);spent+=lengths[0][i+2]+1
        else:
            v=ter[str(i*4+j)];aud.update((i,j),v['label']);spent+=v['prefill']+v['decode']
    assert aud.decision()==set(full['keep'])
    return spent


def item(row,gold):
    y=str(Decimal(gold).normalize());arms=row['arms'];full=arms['full'];lengths=row['lengths']
    s=np.asarray(full['prefix']['0']);base=np.asarray(full['prefix']['1']);m=np.asarray([full['prefix'][str(i)] for i in range(2,6)])
    ter=full['terminal'];labels=np.reshape([ter[str(i)]['label'] for i in range(16)],(4,4))
    truth=exhaustive(s,base,m,labels)
    assert truth==set(full['keep'])
    actual=labels==y;credit=np.sum(s[:,None]*(m-base)*actual)
    assert (y in truth)==(credit>.01)
    stats=dict(official_parseable=np.mean(labels!='INVALID'),below_cap=np.mean([v['decode']<512 for v in ter.values()]),
               strict_parseable=np.mean([parse(v['text']) is not None for v in ter.values()]),full_positive_label_count=len(truth-{'INVALID'}),
               gold_keep=y in truth,material=abs(credit)>.01,full_quality_on=np.sum(s[:,None]*m*actual),full_quality_off=np.sum(s[:,None]*base*actual))
    for name,a in arms.items():
        kept=set(a['keep']);stats[name+'_tokens']=totals(a,lengths)
        stats[name+'_all_mismatch']=kept!=truth;stats[name+'_numeric_set_mismatch']=(kept-{'INVALID'})!=(truth-{'INVALID'})
        stats[name+'_gold_mismatch']=(y in kept)!=(y in truth)
        stats[name+'_conditional_label_mismatch']=sum(v['label']!=ter[i]['label'] for i,v in a['terminal'].items())
        stats[name+'_terminal_calls']=len(a['terminal']);stats[name+'_prefix_calls']=len(a['prefix'])
    for method in ('eager','nested','unpriced'):stats[method+'_tokens']=replay(full,lengths,method)
    return stats


def summarize(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}


def score(folder):
    data=json.loads((folder/'INPUT.json').read_bytes());gold=json.loads((folder/'SEALED.json').read_bytes());group={q['key']:q['task'] for q in data}
    output={}
    for role in range(3):
        rows=list(map(json.loads,(folder/f'WORKER{role}.jsonl').read_bytes().splitlines()))
        assert {r['key'] for r in rows}==set(gold) and len(rows)==48
        results=[item(r,gold[r['key']]) for r in rows]
        agg=summarize(results);gates={};blocks=[];costs={m:{} for m in METHODS}
        for b in range(6):
            group_rows=[(r,s) for r,s in zip(rows,results) if r['block']==b]
            assert len(group_rows)==8
            ct=[r['costs'] for r,_ in group_rows if r['costs']];assert len(ct)==1;ct=ct[0]
            for method in METHODS:
                assert sum(s[method+'_tokens'] for _,s in group_rows)==ct[method]['logical_prefill']+ct[method]['logical_decode']
                for k,v in ct[method].items():costs[method][k]=costs[method].get(k,0)+v
            bs=summarize([s for _,s in group_rows])
            blocks.append(dict(wall={m:ct[m]['wall_seconds'] for m in METHODS},order=group_rows[0][0]['order'],
                               foreign=any(r['foreign'] for r,_ in group_rows),logical_joint=bs['joint_tokens'],logical_row_first=bs['row_first_tokens']))
        saving=1-agg['joint_tokens']/min(agg[m+'_tokens'] for m in ('row_first','eager','nested','unpriced'))
        speed=1-costs['joint']['wall_seconds']/costs['row_first']['wall_seconds']
        qualified=agg['official_parseable']>=.9 and agg['below_cap']>=.8 and agg['joint_all_mismatch']==0
        gates.update(qualification=qualified,logical=qualified and saving>=.05,hardware=qualified and speed>=.05)
        output[str(role)]=dict(summary=agg,grades={t:summarize([s for s,r in zip(results,rows) if group[r['key']]==t]) for t in ('lower','upper')},
                               blocks=blocks,costs=costs,saving_vs_strongest=saving,speed_vs_row_first=speed,gates=gates)
    return dict(models=output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    print(json.dumps(score(p.parse_args().folder),indent=2))
