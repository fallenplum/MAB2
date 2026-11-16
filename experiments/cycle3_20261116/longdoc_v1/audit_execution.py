"""Raw request/journal/native cost/visibility audit, never accesses gold."""
import argparse,hashlib,json,sys,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();r=Path(a.repo);run=Path(a.run);archive=run/'archive';read=lambda f:json.loads(f.read_bytes());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();canon=lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);digest=lambda v:hashlib.sha256(canon(v).encode()).hexdigest()
sys.path.insert(0,str(r/'experiments/cycle2_20261113/durable_backend_v1'));from durable_backend import read_journal
plan=read(run/'PLAN.json');manifest=read(archive/'ARCHIVE_MANIFEST.json');assert sha(run/'PLAN.json')==read(archive/'CLOSURE.json')['plan_sha256']
for name,h in manifest.items():assert sha(archive/name)==h
closure=read(archive/'CLOSURE.json');assert closure['all_workers_exited'] and closure['status']=='COMPLETE_UNSCORED';assert abs(sum(closure['residence_seconds_by_worker'])/3600-closure['actual_GPU_hours'])<1e-12
checks=0;native_count=0;logical_count=0;total_tokens=0;peak=0;all_indices=[];model_identity={};prior_count=0
for worker in archive.glob('worker_*'):
    if not worker.is_dir():continue
    events={}
    for model in ['qwen','mistral']:
        identity=read(worker/('LOAD_'+model+'.json'))['identity'];assert identity==plan['models'][model]['identity'];model_identity[model]=digest(identity)
        journal=read_journal(worker/(model+'_requests.jsonl'));assert not journal['pending_request_ids']
        for v in journal['requests'].values():
            e=v['terminal'];assert e['event']=='RETURNED';g=e['generation'];assert g['call_id'] not in events;events[g['call_id']]=e;assert e['backend_identity_sha256']==model_identity[model]
    seen_calls=set()
    for unit in sorted(worker.glob('unit_*')):
        inp=read(unit/'INPUT.json');mat=read(unit/'MATRIX.json');all_indices.append(inp['index']);payload=inp['payload'];assert digest(payload)==mat['question_fingerprint'];assert mat['status']=='COMPLETE_UNSCORED' and len(mat['conditions'])==24;assert digest(mat['protocol'])==mat['protocol_sha256'];cache={};per_model_calls=dict(qwen=0,mistral=0)
        originals={s['id']:s['text'] for s in payload['segments']};parts={n:{s['id'] for i,s in enumerate(payload['segments']) if i%3==j} for j,n in enumerate('UVR')}
        for row in mat['conditions']:
            c=row['condition'];seed=row['execution_seed'];assert seed in plan['seeds'];costs={m:dict(calls=0,input_tokens=0,output_tokens=0) for m in ['qwen','mistral']}
            for node,record in row['nodes'].items():
                req=record['request'];g=record['generation'];fp=record['request_fingerprint'];assert digest(req)==fp;assert req['question_fingerprint']==digest(payload);assert req['seed']==int(digest(['exogenous_path_rng_v1',payload['id'],seed,node])[:16],16)%(2**31-1);assert req['model_identity_sha256']==model_identity[record['model']];assert req['protocol_sha256']==mat['protocol_sha256'];assert req['max_new_tokens']==(256 if record['is_final'] else 128)
                available=set(record['available_ids']);message=json.loads(req['messages'][1]['content']);assert message['question']==payload['question'];assert message['passages']==[s for s in payload['segments'] if s['id'] in available]
                if c.startswith('G') or c=='FLAT':
                    allowed=parts[node].copy() if node!='M' else set();parents=[u for u,v in row['edges'] if v==node] if c.startswith('G') else list('UVR') if node=='M' else []
                    for parent in parents:allowed.update(row['nodes'][parent]['selected_ids'])
                elif c in ['BM25','DENSE']:allowed=set(payload['retrieval'][c])
                else:allowed=set(originals)
                assert available==allowed
                if not record['is_final']:
                    try:
                        selected=json.loads(g['text'].strip());ok=isinstance(selected,list) and len(selected)<=6 and all(type(v) is str for v in selected) and len(set(selected))==len(selected) and set(selected)<=available
                    except (ValueError,TypeError):ok=False
                    assert record['selection_valid']==ok and record['selected_ids']==(selected if ok else [])
                if record['reused']:assert fp in cache and cache[fp]==g
                else:
                    assert fp not in cache and g['call_id'] not in seen_calls;seen_calls.add(g['call_id']);cache[fp]=g;e=events[g['call_id']];assert e['generation']==g and e['messages']==req['messages'] and e['seed']==req['seed'];assert len(e['native_events'])==1;native=e['native_events'][0];assert native['status']=='SUCCESS' and native['prompt_sha256']==digest(req['messages'])
                    for key in ['call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason']:assert native[key]==g[key]
                    native_count+=1;total_tokens+=g['input_tokens']+g['output_tokens'];peak=max(peak,g['gpu_peak_allocated_bytes'])
                for key in ['input_tokens','output_tokens']:costs[record['model']][key]+=g[key]
                costs[record['model']]['calls']+=1;logical_count+=1;per_model_calls[record['model']]+=1;checks+=19
            for model,v in costs.items():
                for key,num in v.items():assert row['logical_service_cost_by_model'][model][key]==num;checks+=1
        assert sum(per_model_calls.values())==68
        for model,num in per_model_calls.items():assert mat['logical_completed_node_cost_by_model'][model]['calls']==num
    assert seen_calls==set(events)
if plan['mode']=='qualification':
    assert sorted(all_indices)==[0,1,2];assert read(archive/'worker_0/REPLAY.json')['status']=='PASS_EXACT_SEEDED_REPLAY'
    old=run.parent/'longdoc_qualification_v1/archive/worker_0'
    if run.name!='longdoc_qualification_v1':
        for old_unit in old.glob('unit_*'):
            file=old_unit/'MATRIX.json' if (old_unit/'MATRIX.json').exists() else old_unit/'PARTIAL.json';previous=read(file);current=read(archive/'worker_0'/old_unit.name/'MATRIX.json');current_by_fp={v['request_fingerprint']:v for row in current['conditions'] for v in row['nodes'].values()}
            for row in previous['conditions']:
                for rec in row['nodes'].values():
                    now=current_by_fp[rec['request_fingerprint']]
                    for k in ['text','input_tokens','output_tokens','finish_reason']:assert rec['generation'][k]==now['generation'][k]
                    prior_count+=1
else:assert sorted(all_indices)==list(range(16))
result=dict(status='PASS_RAW_EXECUTION_AND_COST_AUDIT',mode=plan['mode'],units=len(all_indices),physical_success_requests=native_count,logical_node_calls=logical_count,physical_tokens=total_tokens,peak_allocated_bytes=peak,scalar_checks=checks,prior_successful_generations_matched=prior_count,actual_GPU_hours=closure['actual_GPU_hours'],archive_sha256=sha(archive/'ARCHIVE_MANIFEST.json'),source_sha256=sha(Path(__file__)),gold_accessed=False)
with (run/'EXECUTION_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
