"""Independent stdlib review of completed synthetic qualification records."""
import argparse,hashlib,json,math
from pathlib import Path

def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]

def review(run,plan_path,protocol_path):
    plan=load(plan_path);protocol=load(protocol_path);resident=load(run/'RESIDENT_STATUS.json')
    status=load(run/'INDEPENDENT12_STATUS.json');identity=load(run/'INDEPENDENT12_IDENTITY.json');qualification=load(run/'QUALIFICATION.json')
    checks=0
    def check(value,message):
        nonlocal checks
        if not value:raise AssertionError(message)
        checks+=1
    check(resident['status']=='CLOSED' and resident['returncode']==0 and not resident['reservation_exceeded'],'resident closure')
    check(resident['plan_sha256']==sha(plan_path),'plan pin')
    check(math.isclose(resident['gpu_hours'],resident['resident_seconds']/3600,abs_tol=1e-12),'occupancy arithmetic')
    check(resident['gpu_hours']<=plan['reserved_gpu_hours']==0.5,'budget')
    check(status['status']==qualification['status']=='PASS_ENGINEERING_REVIEW_REQUIRED','completion status')
    check(identity['qualification_source_commit']==plan['source_commit'],'source commit')
    check(identity['qualification_source_manifest']==plan['source_manifest'],'source manifest')
    check(sha(protocol_path)==plan['source_manifest']['experiments/cycle2_20261113/independent12_qualification_v1/PROTOCOL.json'],'protocol pin')
    check(digest(identity['control_protocol_identity'])==identity['control_protocol_hash'],'protocol identity digest')
    check(identity['control_protocol_identity']['control_code_sha256']==protocol['control_source_sha256'],'control source')
    events={m:lines(run/(m+'_physical_calls.jsonl')) for m in ('qwen','mistral')}
    routes=lines(run/'route_calls.jsonl')
    check({m:len(x) for m,x in events.items()}=={'qwen':24,'mistral':28},'physical model counts')
    physical={};route_map={}
    for model,items in events.items():
        for row in items:
            check(row['status']=='SUCCESS' and row['call_id'] not in physical,'physical event uniqueness/status')
            physical[row['call_id']]=(model,row)
    for row in routes:
        check(row['status']=='SUCCESS' and row['call_id'] not in route_map,'route uniqueness/status')
        route_map[row['call_id']]=row
    check(len(physical)==len(route_map)==52 and physical.keys()==route_map.keys(),'52-call closure')
    billed=[];results=[];formats=0;fresh_fields=0
    integrity_fields=('node','input_fingerprint','seed','messages','generation','wire_message','parsed','message_metadata','received_message_tokens','self_state_tokens')
    expected_nodes=[f'r{s}:w{r}' for r in range(4) for s in range(1,4)]+['aggregate']
    for question in protocol['synthetic_inputs']:
        pair=[]
        for repeat in (1,2):
            result=load(run/(question['id']+'_independent12_fresh'+str(repeat)+'.json'));pair.append(result);results.append(result)
            check(result['status']=='COMPLETE' and len(result['slots'])==13,'execution complete')
            check(result['question_id']==question['id'] and result['seed']==protocol['seed'],'fixture/seed')
            check(result['protocol_hash']==identity['control_protocol_hash'],'execution protocol')
            check([s['node'] for s in result['slots']]==expected_nodes,'all fixed slots')
            check(result['physical_cost']==result['logical_service_cost'] and result['cache_reuse_count']==0,'no cross-control cache')
            cost=result['physical_cost']
            check(cost['calls']==cost['returned_generations']==13 and cost['unattempted_calls']==0,'complete requests')
            check(cost['service_complete'] and cost['cost_fields_complete'] and cost['resident_GPU_hours'] is None,'cost scope')
            for role in range(4):
                messages=[s['messages'] for s in result['slots'][3*role:3*role+3]]
                check(messages[0]==messages[1]==messages[2],'independent worker prompt parity')
            for slot in result['slots']:
                node=slot['node'];record=slot['record'];gen=slot['generation'];call=gen['call_id'];billed.append(call)
                model,event=physical[call];route=route_map[call]
                check(model==slot['model']==route['model'],'model route')
                check(route['node']==node,'node route')
                seed=int(digest([question['id'],protocol['seed'],node])[:16],16)%(2**31-1)
                check(seed==slot['seed']==record['seed']==event['seed']==route['seed'],'seed derivation')
                check(digest(slot['messages'])==event['prompt_sha256']==route['prompt_sha256'],'paid prompt digest')
                check(record['output_digest']==digest({k:record[k] for k in integrity_fields}),'record integrity')
                check(record['generation']==gen and record['messages']==slot['messages'],'record equality')
                check(slot['self_state_tokens']==record['self_state_tokens']==0,'no self-state')
                cap=256 if node=='aggregate' else 128
                check(gen['input_tokens']<=8192 and gen['output_tokens']<=cap,'input/output bounds')
                check(not record['reused'] and record['source_execution_id'] is None,'independent no reuse')
                fingerprint=digest({'question':result['question_fingerprint'],'node':node,'messages':slot['messages'],
                    'seed':seed,'protocol':result['protocol_hash'],'max_new_tokens':cap})
                check(fingerprint==slot['input_fingerprint']==record['input_fingerprint'],'input fingerprint')
                for key in ('call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason'):
                    check(gen[key]==event[key], 'generation/physical '+key)
                formats+=not record['message_metadata']['format_ok']
            for key,field in [('input_tokens','input_tokens'),('output_tokens','output_tokens'),('gpu_seconds','gpu_seconds'),('model_elapsed_seconds','elapsed_seconds')]:
                total=sum(s['generation'][field] for s in result['slots'])
                check(math.isclose(total,cost[key],rel_tol=1e-12,abs_tol=1e-9),'execution cost '+key)
            check(cost['received_message_tokens']==sum(s['record']['received_message_tokens'] for s in result['slots']),'received sum')
        for a,b in zip(pair[0]['slots'],pair[1]['slots']):
            for section,fields in protocol['fresh_equality_fields'].items():
                a_values=a if section=='slot' else a[section];b_values=b if section=='slot' else b[section]
                for key in fields:check(a_values[key]==b_values[key],'fresh semantic equality');fresh_fields+=1
    check(len(billed)==len(set(billed))==52 and set(billed)==set(physical),'all paid calls consumed once')
    check(formats==qualification['format_failures']==0,'format counts')
    check(fresh_fields==qualification['fresh_field_equalities']==572,'semantic comparisons')
    totals={key:sum(row[key] for _,row in physical.values()) for key in ('input_tokens','output_tokens','gpu_seconds','elapsed_seconds')}
    for key in ('input_tokens','output_tokens','gpu_seconds'):
        check(math.isclose(totals[key],qualification['costs'][key],rel_tol=1e-12,abs_tol=1e-9),'qualification total '+key)
    return {'status':'PASS_ROOT_INDEPENDENT_JSON_RECONCILIATION','checks':checks,'physical_calls':52,
        'calls_by_model':{'qwen':24,'mistral':28},'fresh_semantic_equalities':fresh_fields,'format_failures':formats,
        'GPU_occupancy_hours':resident['gpu_hours'],'resident_seconds':resident['resident_seconds'],
        'physical_totals':totals,'source_commit':plan['source_commit'],'plan_sha256':sha(plan_path),
        'protocol_sha256':sha(protocol_path),'resident_receipt_sha256':sha(run/'RESIDENT_STATUS.json'),
        'qualification_sha256':sha(run/'QUALIFICATION.json'),'review_source_sha256':sha(__file__),
        'scope':'Independent JSON/identity/arithmetic reconciliation; no models loaded or tokenizer counts recomputed; successful synthetic path only',
        'not_accuracy_or_novelty_evidence':True,'new_model_calls':0,'new_formal_rounds':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=review(a.run,a.plan,a.protocol)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
