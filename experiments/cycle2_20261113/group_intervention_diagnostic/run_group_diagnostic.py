"""Historical G1/G2 only; source-pinned scalar extraction and full-face replay."""
import argparse,collections,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from group_math import summarize_faces
from inventory_metadata import faces_for_masks

G1AUDIT='e52e02e9c11ee722f43b5e9621c13b06bbb8aeb858480a8930213729cbf1f745'
G1PLAN='85fc48c682694154ca47d060c0d6227584d374d3bf19112e4aae54e94c0f56a9'
D2PIN='7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def g2_rows(path):
    if sha(path)!=D2PIN:raise ValueError('D2 input pin mismatch')
    with np.load(path,allow_pickle=False) as z:
        masks=z['masks'];qs=z['row_question'];quality=z['quality_by_seed'];parts=z['partition']
        if masks.shape!=(10394,12) or quality.shape!=(10394,5) or len(parts)!=400:raise ValueError('D2 shape mismatch')
        rows=[]
        for q in range(400):
            indices=np.where(qs==q)[0];lookup={tuple(masks[i]):i for i in indices}
            if len(lookup)!=len(indices):raise ValueError('Duplicate G2 masks')
            faces,_=faces_for_masks(masks[indices])
            for f in faces:rows.append({'q':q,'pair':f['edges'],'background':f['background'],'partition':str(parts[q]),'quality':quality[[lookup[w] for w in f['worlds']]]})
    return rows,{'source_sha256':D2PIN,'questions':400,'seeds':[2701,2702,2703,2704,2705]}

def g1_rows(root):
    auditpath=root/'artifacts/audit/INDEPENDENT_G1_ACCEPTANCE.json'
    if sha(auditpath)!=G1AUDIT:raise ValueError('G1 acceptance pin mismatch')
    audit=json.loads(auditpath.read_bytes())
    if audit['status']!='ACCEPTED_WITH_DOCUMENTED_LIMITATIONS' or audit['score_replay']['mismatches']!=0:raise ValueError('G1 acceptance failed')
    rel='results/pilot/motivation_v3_20261103/shard_000/frozen_plan.json'
    if sha(root/rel)!=G1PLAN:raise ValueError('G1 plan mismatch')
    plan=json.loads((root/rel).read_bytes());qids=sorted(plan['question_ids']);seeds=plan['seeds']
    if len(qids)!=200 or seeds!=[1701,1702,1703,1704,1705]:raise ValueError('Unexpected G1 population')
    qi={q:i for i,q in enumerate(qids)};si={s:i for i,s in enumerate(seeds)}
    obsrels=sorted(k for k in audit['input_sha256'] if k.startswith('results/pilot/motivation_v3_20261103/shard_') and k.endswith('/observations.jsonl'))
    if len(obsrels)!=3:raise ValueError('Expected three accepted G1 observation shards')
    cells={};orig_groups={};sources={};seenrows=0;duplicate=0
    for rel in obsrels:
        path=root/rel;pin=audit['input_sha256'][rel]
        if sha(path)!=pin:raise ValueError('G1 observation pin mismatch')
        sources[rel]=pin
        with path.open(encoding='utf8') as f:
            for line in f:
                r=json.loads(line);seenrows+=1
                q=qi[r['question_id']];s=si[r['seed']];mask=tuple(r['graph_mask'])
                if r['protocol_hash']!=plan['protocol_hash'] or len(mask)!=12 or any(x not in (0,1) for x in mask):raise ValueError('G1 row lineage mismatch')
                quality=r['quality'];cost=r['logical_service_cost']
                if quality not in (0,1):raise ValueError('Nonbinary G1 quality')
                token=cost['input_tokens']+cost['output_tokens'];calls=cost['calls']
                if not isinstance(token,int) or not isinstance(calls,int) or token<0 or calls<0:raise ValueError('Bad G1 logical cost')
                key=(q,mask,s);value=(float(quality),token,calls)
                if key in cells:
                    duplicate+=1
                    if cells[key]!=value:raise ValueError('Same-q/seed/mask has conflicting quality or logical costs; do not choose a replicate')
                cells[key]=value
                group=(q,r['context_id'],r['edge_e'],r['edge_f'])
                orig_groups.setdefault(group,{})[(r['cell'],s)]=mask
    if seenrows!=60000 or len(orig_groups)!=3000:raise ValueError('G1 frozen coverage mismatch')
    byq=collections.defaultdict(set)
    for q,mask,s in cells:byq[q].add(mask)
    rows=[];corners=collections.Counter()
    for q in range(200):
        faces,counts=faces_for_masks(byq[q]);corners.update(counts)
        for f in faces:
            y=[];tokens=[];calls=[]
            for w in f['worlds']:
                if any((q,w,s) not in cells for s in range(5)):raise ValueError('Incomplete G1 seed/world block')
                x=[cells[(q,w,s)] for s in range(5)];y.append([v[0] for v in x]);tokens.append([v[1] for v in x]);calls.append([v[2] for v in x])
            rows.append({'q':q,'pair':f['edges'],'background':f['background'],'quality':np.array(y),'tokens':np.array(tokens),'calls':np.array(calls)})
    declared=[]
    rowlookup={(r['q'],tuple(r['pair']),tuple(r['background'])):r for r in rows}
    for (q,context,e,f),keys in sorted(orig_groups.items()):
        masks=set(keys.values());faces,_=faces_for_masks(masks)
        match=[x for x in faces if x['edges']==[e,f]]
        if len(match)!=1 or len(keys)!=20:raise ValueError('Original G1 declared face mismatch')
        row=dict(rowlookup[(q,(e,f),match[0]['background'])]);row['context']=context;declared.append(row)
    return rows,declared,{'acceptance_sha256':G1AUDIT,'plan_sha256':G1PLAN,'observation_sources':sources,
                         'questions':200,'seeds':seeds,'observation_rows':seenrows,'duplicate_same_world_rows':duplicate,
                         'deduplicated_question_mask_seed_cells':len(cells),'original_declared_groups':len(declared),
                         'all_complete_faces':len(rows),'observed_corner_count_histogram':dict(corners)}

def report(rows):
    out=summarize_faces(rows);private=out.pop('private_question_values')
    out['by_edge_pair']={','.join(map(str,p)):{k:v for k,v in summarize_faces([r for r in rows if tuple(r['pair'])==p]).items() if k!='private_question_values'} for p in sorted({tuple(r['pair']) for r in rows})}
    return out,private

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',required=True,type=Path);ap.add_argument('--d2-input',required=True,type=Path);ap.add_argument('--output-dir',required=True,type=Path);args=ap.parse_args()
    if args.output_dir.exists():raise SystemExit('Refuse reuse of run directory')
    source=Path(__file__).resolve().parent;start=time.time();start_cpu=time.process_time()
    freeze=json.loads((source/'PREEXECUTION_FREEZE.json').read_bytes())
    for rel,pin in freeze['sha256'].items():
        if sha(source/rel)!=pin:raise ValueError('Preexecution source/contract pin mismatch: '+rel)
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir/'STARTED.json',{'time_unix':start,'freeze_sha256':sha(source/'PREEXECUTION_FREEZE.json'),'access':'Previously exposed G1/G2 only; first world-level numeric analysis after freeze'})
    rows2,meta2=g2_rows(args.d2_input);rows1,declared1,meta1=g1_rows(args.project_root)
    out={'schema':'mab2_group_intervention_results_v1','evidence':'EXPLORATORY_HISTORICAL_ONLY','new_gpu_hours':0,'new_model_calls':0,'project_stop':False,'metadata':{'G1':meta1,'G2':meta2}}
    populations={'G1_all_observed_faces':rows1,'G1_original_declared_faces':declared1,'G2_all_observed_faces':rows2}
    populations.update({'G2_'+part:[r for r in rows2 if r['partition']==part] for part in ('train','tune','test')})
    for name,rows in populations.items():
        public,private=report(rows);out[name]=public;dump(args.output_dir/'private'/f'{name}_question_values.json',private)
    out['elapsed_seconds']=time.time()-start
    out['cpu_seconds']=time.process_time()-start_cpu
    if out['cpu_seconds']>1800:raise RuntimeError('CPU time budget exceeded')
    dump(args.output_dir/'RESULTS.json',out)
    dump(args.output_dir/'COMPLETE.json',{'results_sha256':sha(args.output_dir/'RESULTS.json'),'freeze_sha256':sha(source/'PREEXECUTION_FREEZE.json'),'elapsed_seconds':out['elapsed_seconds'],'cpu_seconds':out['cpu_seconds'],'new_gpu_hours':0})
    print(json.dumps({'status':'COMPLETE','output':str(args.output_dir),'G1_faces':len(rows1),'G2_faces':len(rows2),'elapsed_seconds':out['elapsed_seconds']}))

if __name__=='__main__':main()
