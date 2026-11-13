"""Independent archive audit. Bytes/callback inputs only; never loads labels or models.

No production validator, accounting routine or statistics function is called.
Success yields REVIEW_REQUIRED, not a label-release decision.
"""
from collections import Counter
import ast
import hashlib
import json
import math
import re

N=384
SEEDS=(26111301,26111302)
PROBES=(1,5,7,9)
EDGES=tuple((s,d) for s in range(4) for d in range(4) if s!=d)
NODES=tuple(f'r{r}:w{w}' for r in (1,2,3) for w in range(4))+('aggregate',)
INDEPENDENT=tuple(f'r{s}:w{r}' for r in range(4) for s in (1,2,3))+('aggregate',)
FIRST=('A','B','NO_EDGE','A_MINUS_E','B_MINUS_E','INDEPENDENT12','SINGLE_M')
SECOND=('A','B','A_MINUS_E','B_MINUS_E')
ORDER=tuple(f'S{s}/{x}' for s,names in ((1,FIRST),(2,SECOND)) for x in names)
INTEGRITY=('node','input_fingerprint','seed','messages','generation','wire_message','parsed',
           'message_metadata','received_message_tokens','self_state_tokens')
COST=('calls','input_tokens','output_tokens','received_message_tokens','self_state_tokens',
      'model_elapsed_seconds','gpu_seconds','peak_allocated_bytes')
NATIVE_FIELDS=('call_id','status','seed','started_unix','input_tokens','output_tokens','elapsed_seconds',
               'gpu_seconds','gpu_peak_allocated_bytes','finish_reason','prompt_sha256','error_type')
METADATA=('RAW_OUTPUT_MANIFEST.json','PARENT_RESIDENCE.json','ATTEMPT_RESULT.json')
REVIEW_SCOPES=('source_model_tokenizer_identity','native_durable_generation_and_cost_lineage',
               'cache_input_and_output_lineage','all_attempts_and_physical_residence_ledger','no_labels_during_collection')
REQUIRED_PINS=('protocol_commit','model_files','tokenizer_files','graph_executor_source','independent12_operator_source','single_operator_source',
 'node_seed_derivation','prompts_and_formatter','official_hotpot_scorer','local_answer_adapter','runtime_environment','qualified_operator_receipts',
 'source_train_manifest','payload_normalizer','known_exposure_exclusions','selected_question_manifest','edge_assignment_manifest',
 'label_vault_access_control','budget_and_worker_cap_receipt','R1_closure_receipt')
FINAL_SUFFIX=('\n\nRequired final format: {"answer":"<final answer>","uncertainty":"","evidence":"<brief check>"}. '
    'Include exactly all 3 keys in that order. Never omit uncertainty, even when empty. '
    'Return only the complete single-line JSON object.')

class AuditError(ValueError):pass
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(raw):
    if type(raw) is not bytes:raise AuditError('exact bytes required')
    return hashlib.sha256(raw).hexdigest()
def digest(x):return sha(canonical(x).encode())
def newline_digest(x):return sha((canonical(x)+'\n').encode())
def decode(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise AuditError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(AuditError('nonfinite JSON')))
def require(ok,what):
    if not ok:raise AuditError(what)
def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0
def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8)
def valid_sha(x):return isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None
def model(node):return 'mistral' if node=='aggregate' or int(node[-1])%2 else 'qwen'
def mask(name,edge):
    if name in ('INDEPENDENT12','SINGLE_M'):return None
    bits=4095 if name.startswith('A') else 674 if name.startswith('B') else 0
    if name.endswith('MINUS_E'):bits &= ~(1<<edge)
    return [(bits>>i)&1 for i in range(12)]
def records(execution):return execution['nodes'] if 'nodes' in execution else {s['node']:s['record'] for s in execution['slots'] if s['record'] is not None}
def ordering(name):return ('aggregate',) if name=='SINGLE_M' else INDEPENDENT if name=='INDEPENDENT12' else NODES

def closure(before,after):
    if before is None:return set(NODES)
    affected={f'r{r}:w{EDGES[i][1]}' for i in range(12) if before[i]!=after[i] for r in (2,3)}
    while True:
        old=set(affected)
        for r in (2,3):
            for w in range(4):
                pred={f'r{r-1}:w{w}'}|{f'r{r-1}:w{s}' for i,(s,d) in enumerate(EDGES) if d==w and (before[i] or after[i])}
                if pred&affected:affected.add(f'r{r}:w{w}')
        if any(f'r3:w{w}' in affected for w in range(4)):affected.add('aggregate')
        if old==affected:return affected

def node_cost(rows):
    rows=list(rows)
    out={'calls':len(rows),'input_tokens':0,'output_tokens':0,'received_message_tokens':0,'self_state_tokens':0,
         'model_elapsed_seconds':0.,'gpu_seconds':0.,'peak_allocated_bytes':0}
    for row in rows:
        g=row['generation']
        for name,field in (('input_tokens','input_tokens'),('output_tokens','output_tokens'),
                           ('model_elapsed_seconds','elapsed_seconds'),('gpu_seconds','gpu_seconds')):
            require(finite(g[field]),'unknown generation cost');out[name]+=g[field]
        require(finite(g['gpu_peak_allocated_bytes']),'unknown peak');out['peak_allocated_bytes']=max(out['peak_allocated_bytes'],g['gpu_peak_allocated_bytes'])
        for field in ('received_message_tokens','self_state_tokens'):
            require(type(row[field]) is int and row[field]>=0,'invalid wire token count');out[field]+=row[field]
    return out
def sum_cost(rows):
    rows=list(rows)
    return {k:(max((x[k] for x in rows),default=0) if k=='peak_allocated_bytes' else sum(x[k] for x in rows)) for k in COST}
def check_cost(actual,expected):
    for k,v in expected.items():require(close(actual.get(k),v),'cost mismatch '+k)

def literal_constants(source_bytes,initial=None):
    """Read string literals from frozen source without importing/executing it."""
    values=dict(initial or {})
    def literal(node):
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return literal(node.left)+literal(node.right)
        return ast.literal_eval(node)
    for node in ast.parse(source_bytes).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try:values[node.targets[0].id]=literal(node.value)
            except (ValueError,TypeError,KeyError):pass
    return values

def normalize(raw,tokenizer,cap):
    """Independent reconstruction of the pinned formatter, including failure/trim."""
    text=raw.strip();match=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.I|re.S)
    if match:text=match.group(1).strip()
    parsed=None;normalized=False;error=None
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate_json_key')
            out[k]=v
        return out
    try:
        x=json.loads(text,object_pairs_hook=pairs)
        if not isinstance(x,dict) or set(x)!={'answer','evidence','uncertainty'}:error='missing_or_extra_fields'
        elif not isinstance(x['evidence'],str) or not isinstance(x['uncertainty'],str):error='invalid_field_type'
        elif isinstance(x['answer'],str):parsed=dict(x)
        elif type(x['answer']) in (int,float):
            if isinstance(x['answer'],float) and not math.isfinite(x['answer']):error='non_finite_answer'
            else:parsed={**x,'answer':str(x['answer'])};normalized=True
        else:error='invalid_answer_type'
    except (ValueError,json.JSONDecodeError) as exc:
        error='duplicate_json_key' if str(exc)=='duplicate_json_key' else 'invalid_or_incomplete_json'
    ok=parsed is not None
    if parsed is None:parsed={'answer':'','evidence':'','uncertainty':'FORMAT_ERROR'}
    wire=canonical(parsed);trimmed=False
    for field in ('evidence','uncertainty','answer'):
        while len(tokenizer.encode(wire))>cap and parsed[field]:
            tokens=tokenizer.encode(parsed[field]);overshoot=len(tokenizer.encode(wire))-cap
            parsed[field]=tokenizer.decode(tokens[:max(0,len(tokens)-max(1,overshoot))]);wire=canonical(parsed);trimmed=True
    require(len(tokenizer.encode(wire))<=cap,'empty schema cannot fit')
    return wire,parsed,{'format_ok':ok,'type_normalized':normalized,'format_error':error,'message_trimmed':trimmed,'message_tokens':len(tokenizer.encode(wire))}

def prompt(payload,name,node,rows,graph_mask,constants,tokenizers):
    full=payload['question']+'\n\nSupplied context:\n'+payload['context'];incoming=[];own_count=0;m=model(node)
    if name=='SINGLE_M':system=constants['SINGLE_SYSTEM_PROMPT'];user=full
    elif node=='aggregate':
        system=constants['AGGREGATOR_PROMPT']
        if name=='INDEPENDENT12':
            incoming=[{'role':int(n[-1]),'sample':int(n[1]),'node':n,'message':rows[n]['wire_message']} for n in INDEPENDENT[:-1]]
            user=full+constants['AGGREGATE_INPUT_PREFIX']+canonical(incoming)+constants['AGGREGATE_FORMAT_SUFFIX']
        else:
            incoming=[{'worker':w,'message':rows[f'r3:w{w}']['wire_message']} for w in range(4)]
            user=full+'\n\nFinal worker messages:\n'+canonical(incoming)+FINAL_SUFFIX
    else:
        r,w=int(node[1]),int(node[-1]);system=constants['ROLE_PROMPTS'][w]+' '+constants['MESSAGE_INSTRUCTION'];user=full
        if r==1 or name=='INDEPENDENT12':user+='\n\nRound 1: work independently. No external messages are available.'
        else:
            own=rows[f'r{r-1}:w{w}']['wire_message'];own_count=len(tokenizers[m].encode(own))
            incoming=[{'worker':s,'message':rows[f'r{r-1}:w{s}']['wire_message']} for i,(s,d) in enumerate(EDGES) if graph_mask[i] and d==w]
            user+=f'\n\nRound {r}. Your previous state:\n{own}\n\nExternal messages (ordered by source worker):\n'
            user+=canonical(incoming) if incoming else '[] (no external messages)'
            user+='\nRecheck your candidate using the question, your previous state and these messages.'
    return [{'role':'system','content':system},{'role':'user','content':user}],sum(len(tokenizers[m].encode(x['message'])) for x in incoming),own_count

def collector(row):
    require(set(row)=={'id','question','context','type','level'},'input allowlist')
    x=row['context'];require(set(x)=={'title','sentences'},'context schema')
    require(len(x['title'])==len(x['sentences'])>0 and len(set(x['title']))==len(x['title']),'context coverage')
    require(all(isinstance(t,str) and t.strip() for t in x['title']) and all(isinstance(s,list) and all(isinstance(t,str) for t in s) for s in x['sentences']),'context types')
    return {'id':'hotpotqa_'+row['id'],'question':row['question'],'context':json.dumps([{'title':t,'sentences':s} for t,s in zip(x['title'],x['sentences'])],ensure_ascii=False,separators=(',',':'),allow_nan=False)}

def parse_journal(raw,worker,m,plan,plan_sha,unit_lookup,tokenizer):
    require(raw and raw.endswith(b'\n'),'unclosed durable tail')
    events=[decode(line) for line in raw.splitlines()];header=events[0];identity=plan['runtime']['models'][m]['identity']
    require(header=={'event':'HEADER','schema':'durable_backend_journal_v1','model':m,'backend_identity':identity,
                    'helper_sha256':plan['runtime']['modules']['durable']['sha256']},'durable header')
    require((len(events)-1)%2==0,'unclosed START')
    calls={};rejects=[];request_ids=set();native_index=0
    for i in range(1,len(events),2):
        start,end=events[i:i+2];rid=start['request_id'];context=start['context'];node=context['node']
        require(start['event']=='START' and end['event'] in ('RETURNED','ERROR') and rid not in request_ids,'durable pairing')
        request_ids.add(rid);require(all(end.get(k)==v for k,v in start.items() if k!='event'),'durable metadata changed')
        require(start['schema']=='durable_backend_journal_v1' and start['model']==m and start['backend_identity_sha256']==digest(identity),'request identity')
        require(canonical(start['config'])==canonical(plan['runtime']['graph_config']),'generation configuration')
        unit=unit_lookup[context['unit_index']]
        require(set(context)=={'attempt_id','plan_sha256','worker_id','unit_index','unit_payload_sha256','node'} and
            context['attempt_id']==plan['attempt_id'] and context['plan_sha256']==plan_sha and context['worker_id']==worker==unit['worker'] and
            context['unit_payload_sha256']==unit['payload_sha256'] and node in NODES and model(node)==m,'durable unit context')
        cap=256 if node=='aggregate' else 128;require(start['max_new_tokens']==cap,'request output cap')
        require(start['native_event_start']==native_index,'native event interval start')
        native=end['native_events'];require(end['native_event_end']==native_index+len(native),'native event interval end');native_index+=len(native)
        chat_tokens=tokenizer.chat_count(start['messages'])
        if end['event']=='ERROR':
            require(end['generation'] is None and end['error_type']=='ValueError' and not native,'infrastructure/unknown failure is not admissible')
            require(chat_tokens>8192 or chat_tokens+cap>tokenizer.position_limit,'unproven capacity rejection')
            rejects.append({'start':start,'end':end,'model':m,'worker':worker,'ordinal':i});continue
        g=end['generation'];require(len(native)==1 and native[0]['status']=='SUCCESS','native success multiplicity')
        n=native[0]
        require(g['call_id'] not in calls and isinstance(g['call_id'],str),'duplicate physical call ID')
        require(all(n.get(k)==g[k] for k in ('call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason')),'native/returned cost disagreement')
        require(n['seed']==start['seed'] and n['prompt_sha256']==digest(start['messages']),'native input differs')
        require(g['input_tokens']==chat_tokens<=8192 and chat_tokens+cap<=tokenizer.position_limit,'actual chat token count/cap')
        require(type(g['output_tokens']) is int and 0<=g['output_tokens']<=cap,'actual output cap')
        for key in ('elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes'):require(finite(g[key]),'unknown native cost')
        expected={'max_new_tokens':cap,'min_new_tokens':0,'do_sample':True,'temperature':.7,'top_p':.9,'top_k':0,
                  'num_beams':1,'num_return_sequences':1,'repetition_penalty':1.,'length_penalty':1.,'use_cache':True,
                  'return_dict_in_generate':False,'output_scores':False}
        require(all(n['generation_config'].get(k)==v for k,v in expected.items()),'native generation settings')
        calls[g['call_id']]={'start':start,'end':end,'native':n,'worker':worker,'model':m,'ordinal':i}
    sequence=[]
    for frame in sorted(list(calls.values())+rejects,key=lambda x:x['ordinal']):
        index=frame['start']['context']['unit_index']
        if not sequence or sequence[-1]!=index:sequence.append(index)
    expected=[u['index'] for u in plan['schedule']['units'] if u['worker']==worker]
    require(sequence==expected,'whole-worker unit order differs from frozen schedule')
    return calls,rejects

def check_residence(plan,plan_sha,archive,prior_ledger,prior_receipts):
    r=decode(archive['PARENT_RESIDENCE.json']);attempt=decode(archive['ATTEMPT_RESULT.json']);started=decode(archive['ATTEMPT_STARTED.json'])
    workers=plan['schedule']['worker_count'];keys={str(w) for w in range(workers)};budget=plan['budget']
    require(r['transaction_id']==plan['attempt_id']==attempt['attempt_id']==started['attempt_id'],'attempt identity')
    require(started['plan_sha256']==plan_sha==attempt['plan_sha256'] and started['scores_accessed'] is False,'attempt source')
    require(r['all_workers_closed'] is True and r['unknown_unclosed_residence'] is False and not r['kill_integrity_errors'],'unclosed resident')
    require(set(r['worker_started_monotonic'])==set(r['worker_closed_monotonic'])==keys,'resident workers')
    spent=[]
    for w in range(workers):
        start,end=r['worker_started_monotonic'][str(w)],r['worker_closed_monotonic'][str(w)]
        require(finite(start) and finite(end) and end>=start,'resident interval');spent.append(end-start)
        require(close(spent[w],r['observed_seconds_by_worker'][w]),'resident elapsed disagreement')
        require(spent[w]<=budget['worker_caps_seconds'][w],'worker resident cap exceeded')
    require(close(sum(spent),r['closed_sum_seconds']) and close(sum(spent),r['observed_residence_lower_bound_seconds']),'resident total')
    require(r['cap_used_as_cost'] is False and r['includes_model_loading_and_barrier_idle'] is True,'resident measurement scope')
    require(prior_ledger['status']=='CLOSED_REGISTRY_EXTERNAL_ROOT_REVIEW' and prior_ledger['all_known_attempts_enumerated'] is True,'missing external attempt registry')
    tx=prior_ledger['transactions'];ids=[t['transaction_id'] for t in tx]
    require(len(set(ids))==len(ids) and plan['attempt_id'] not in ids,'duplicate/current transaction in prior registry')
    require(len({t['receipt_sha256'] for t in tx})==len(tx),'same prior receipt charged under aliases')
    require(set(prior_receipts)==set(ids),'prior receipt coverage')
    for row in tx:
        require(finite(row['seconds']) and sha(prior_receipts[row['transaction_id']])==row['receipt_sha256'],'prior cost receipt pin')
        receipt=decode(prior_receipts[row['transaction_id']])
        if 'closed_sum_seconds' in receipt:
            require(receipt['transaction_id']==row['transaction_id'],'prior receipt transaction identity')
            require(receipt['all_workers_closed'] is True and not receipt['unknown_unclosed_residence'],'prior unclosed parent residence')
            measured=receipt['closed_sum_seconds']
        elif 'resident_seconds' in receipt:
            require(receipt['status']=='CLOSED','prior unclosed engineering residence')
            require(row.get('identity_binding')=='ROOT_REGISTRY_BINDING_LEGACY_RECEIPT_WITHOUT_ID' and
                    receipt.get('transaction_id',row['transaction_id'])==row['transaction_id'],'legacy receipt needs explicit root registry identity binding')
            measured=receipt['resident_seconds']
        else:
            require(receipt['status']=='ROOT_REVIEWED_CLOSED_TRANSACTION' and receipt['transaction_id']==row['transaction_id'],'unrecognized prior receipt')
            measured=receipt['closed_seconds']
        require(close(measured,row['seconds']),'prior transaction charge differs')
    candidate=budget['prior_candidate_transactions'];lookup={t['transaction_id']:t for t in tx}
    require(len({t['transaction_id'] for t in candidate})==len(candidate),'duplicate candidate charge')
    require(all(t['transaction_id'] in lookup and t['seconds']==lookup[t['transaction_id']]['seconds'] and t['already_in_global_ledger'] is True for t in candidate),'candidate prior lineage')
    require(set(prior_ledger['candidate_prior_transaction_ids'])=={t['transaction_id'] for t in candidate},'candidate completeness differs')
    require(sum(spent)<=budget['global_available_seconds_for_attempt'] and sum(spent)+sum(t['seconds'] for t in candidate)<=budget['candidate_cap_seconds'],'allocated residence exceeded')
    require(sum(t['seconds'] for t in tx)+sum(spent)<=prior_ledger['authorized_global_cap_seconds'],'global resource cap exceeded')
    return {'current_transaction_id':plan['attempt_id'],'seconds_by_worker':spent,'current_seconds':sum(spent),
            'prior_global_seconds':sum(t['seconds'] for t in tx),'candidate_seconds':sum(spent)+sum(t['seconds'] for t in candidate),
            'global_seconds':sum(spent)+sum(t['seconds'] for t in tx),'external_registry_coverage_is_trusted_not_inferred_from_archive':True}

def check_physical(reported,returned,trace,complete):
    values=node_cost(returned);missing=len(trace)-len(returned)
    require(reported['calls']==len(trace) and reported['returned_generations']==len(returned),'physical request multiplicity')
    require(reported['failed_requests_without_physical_record']==missing,'unknown failure count')
    require(reported['cost_fields_complete']==(missing==0) and reported['service_complete']==complete,'physical completeness summary')
    for key in ('input_tokens','output_tokens','model_elapsed_seconds','gpu_seconds','peak_allocated_bytes'):
        require(close(reported['known_cost_components'][key],values[key]),'known component mismatch')
        require(reported[key] is None if missing else close(reported[key],values[key]),'unknown must remain null')
    require(reported['received_message_tokens']==sum(t['received_message_tokens'] or 0 for t in trace),'physical receiver sum')
    require(reported['self_state_tokens']==sum(t['self_state_tokens'] for t in trace),'physical self sum')
    check_cost(reported['returned_generation_cost'],values)

def check_matrix(matrix,payload,unit,bindings,constants,tokenizers,raw_calls,rejects):
    require(matrix['operator_id']=='single_question_hotpot_applicability_matrix_v1_cpu_staging','matrix operator')
    require(matrix['question_id']==payload['id'] and matrix['question_fingerprint']==digest(payload),'matrix input')
    require(matrix['matrix_protocol_hash']==bindings['matrix_protocol_hash'] and (matrix['seed1'],matrix['seed2'])==SEEDS and matrix['probe_edge']==unit['edge_index'],'matrix protocols')
    conditions=matrix['conditions'];require([c['key'] for c in conditions]==list(ORDER),'384 x 11 named coverage')
    require(matrix['condition_set_terminal'] is True and matrix['scores_computed'] is False and matrix['scientific_admission'] is False,'unclosed/scored matrix')
    require(all(c['status'] in ('COMPLETE','INPUT_REJECTED') for c in conditions),'infrastructure terminal')
    lookup={c['key']:c for c in conditions};consumed=[];logical=[];physical=[];all_trace=[];formats=0;reject_count=0
    expected_stream={m:sorted([r for r in list(raw_calls.values())+rejects if r['start']['context']['unit_index']==unit['index'] and r['model']==m],key=lambda r:r['ordinal']) for m in ('qwen','mistral')}
    observed_stream={m:[] for m in expected_stream}
    for c in conditions:
        prefix,name=c['key'].split('/');seed=SEEDS[int(prefix[1:])-1];gm=mask(name,unit['edge_index']);complete=c['status']=='COMPLETE'
        require(c['name']==name and c['seed_number']==int(prefix[1:]) and c['seed']==seed and c['mask']==gm,'condition identity')
        protocol=bindings['independent12_protocol_hash' if name=='INDEPENDENT12' else 'single_protocol_hash' if name=='SINGLE_M' else 'graph_protocol_hash']
        base_name={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(name)
        base=lookup[prefix+'/'+base_name] if base_name else None
        if base is not None and base['status']!='COMPLETE':base=None
        require(c['cache_source_condition']==(base['key'] if base else None),'base availability lineage')
        order=ordering(name);affected=closure(base['mask'] if base else None,gm) if gm is not None else set(order)
        require(c['planned_physical_calls']==len(affected),'actual planned geometry')
        execution=c['execution'] if complete else c['partial_execution'];require(execution is not None,'missing execution evidence')
        require(execution['seed']==seed and execution['question_fingerprint']==digest(payload) and execution['protocol_hash']==protocol,'execution identity')
        rows=records(execution)
        if complete:
            require(set(rows)==set(order) and execution['question_id']==payload['id'],'logical node coverage')
            require(c['error_type'] is None and c['failure_category'] is None,'complete error metadata')
            require(execution['answer']==rows['aggregate']['parsed']['answer'],'aggregate alias')
            if gm is not None:require(execution['mask']==gm and execution['closure']==sorted(affected),'graph closure')
        else:
            reject_count+=1;require(c['execution'] is None and c['logical_service_cost'] is None,'overflow fabricated completed execution')
            require(c['failure_category'] in ('INPUT_TOKEN_LIMIT','MODEL_POSITION_LIMIT'),'unknown rejection')
            require(c['request_trace'] and c['request_trace'][-1]['generation'] is None,'capacity final request missing')
            failed=c['request_trace'][-1]['node'];require(failed in affected and set(rows)==set(order[:order.index(failed)]),'partial execution is not exact pre-rejection prefix')
            if gm is not None:require(execution['mask']==gm,'partial graph mask')
            else:
                slots=execution['slots'];require([s['node'] for s in slots]==list(order),'partial separate slot order')
                for i,s in enumerate(slots):
                    require(s['index']==i and s['model']==model(s['node']),'partial slot identity')
                    require(s['attempted_backend_request']==(i<=order.index(failed)),'partial slot attempt prefix')
                    require((s['record'] is not None)==(i<order.index(failed)),'partial slot record prefix')
        if gm is None and complete:
            slots=execution['slots'];require([s['node'] for s in slots]==list(order),'independent slot order')
            for i,s in enumerate(slots):
                n=s['node'];require(s['index']==i and s['model']==model(n) and s['record']==rows[n] and s['generation']==rows[n]['generation'],'slot identity')
                require(s['role']==(None if n=='aggregate' else int(n[-1])) and s['sample']==(None if n=='aggregate' else int(n[1])),'role/sample identity')
        fresh=[]
        for node in order:
            if node not in rows:continue
            row=rows[node];m=model(node);cap=256 if node=='aggregate' else 128
            require(row['node']==node and row['output_digest']==digest({k:row[k] for k in INTEGRITY}),'node output digest')
            node_seed=int(digest([payload['id'],seed,node])[:16],16)%(2**31-1)
            messages,received,self_count=prompt(payload,name,node,rows,gm,constants,tokenizers)
            require(row['seed']==node_seed and row['messages']==messages,'complete prompt/seed reconstruction')
            require(row['input_fingerprint']==digest({'question':digest(payload),'node':node,'messages':messages,'seed':node_seed,'protocol':protocol,'max_new_tokens':cap}),'node input fingerprint')
            require(row['received_message_tokens']==received and row['self_state_tokens']==self_count,'receiver/self tokenizer count')
            wire,parsed,metadata=normalize(row['generation']['text'],tokenizers[m],cap)
            require(row['wire_message']==wire and row['parsed']==parsed and row['message_metadata']==metadata,'raw formatter reconstruction')
            formats+=not metadata['format_ok']
            reuse=node not in affected;require(row['reused']==reuse,'conservative closure reuse')
            if reuse:
                original=records(base['execution'])[node]
                require(row['source_execution_id']==base['execution']['execution_id'] and all(row[k]==original[k] for k in INTEGRITY),'exact original base reuse')
            else:require(row['source_execution_id'] is None,'fresh source');fresh.append(row)
            raw=raw_calls[row['generation']['call_id']]
            require(raw['start']['context']['unit_index']==unit['index'] and raw['worker']==unit['worker'] and raw['model']==m and raw['end']['generation']==row['generation'],'raw generation provenance')
            require(raw['start']['messages']==messages and raw['start']['seed']==node_seed and raw['start']['context']['node']==node,'raw input provenance')
        traces=c['request_trace'];routes=c['route_events'];require(len(traces)==len(routes)==len(fresh)+(0 if complete else 1),'trace/route multiplicity')
        require([t['node'] for t in traces[:len(fresh)]]==[r['node'] for r in fresh],'physical node order')
        for ordinal,(t,route) in enumerate(zip(traces,routes)):
            node=t['node'];m=model(node);cap=256 if node=='aggregate' else 128
            require(t['model']==m and t['attempted_backend_request'] is True,'trace model/attempt')
            if t['generation'] is not None:
                row=fresh[ordinal];raw=raw_calls[row['generation']['call_id']]
                require(t['record']==row and t['generation']==row['generation'],'trace node equality')
                native_projection={k:raw['native'][k] for k in NATIVE_FIELDS if k in raw['native']}
                require(t['physical_events']==[native_projection],'native projection differs')
                require(route['status']=='SUCCESS' and route['call_id']==row['generation']['call_id'] and route['input_tokens']==row['generation']['input_tokens'] and route['output_tokens']==row['generation']['output_tokens'],'route result')
            else:
                require(not complete and ordinal==len(traces)-1 and t['error_type']=='ValueError' and not t['physical_events'],'nonterminal rejection')
                candidates=[r for r in expected_stream[m] if r['end']['event']=='ERROR' and r['start']['request_id'] not in consumed]
                require(candidates,'missing durable rejection');raw=candidates[0]
                expected_messages,received,self_count=prompt(payload,name,node,rows,gm,constants,tokenizers)
                require(t['messages']==expected_messages and t['received_message_tokens']==received and t['self_state_tokens']==self_count,'rejected prompt evidence')
                require(route['status']=='FAILED' and route['error_type']=='ValueError','rejected route')
            start=raw['start'];require(start['messages']==t['messages'] and start['seed']==t['seed'] and start['context']['node']==node,'durable/trace input')
            require(route['node']==node and route['model']==m and route['seed']==t['seed'] and route['max_new_tokens']==cap and route['prompt_sha256']==digest(t['messages']),'route input')
            require(route['model_identity_sha256']==start['backend_identity_sha256'],'route backend identity')
            require(start['request_id'] not in consumed,'request consumed twice');consumed.append(start['request_id']);observed_stream[m].append(start['request_id'])
        check_physical(c['physical_cost'],fresh,traces,complete)
        if complete:
            expected=node_cost(rows[n] for n in order);logical.append(expected)
            check_cost(c['logical_service_cost'],expected);check_cost(execution['logical_service_cost'],expected)
            check_cost(execution['physical_cost'],node_cost(fresh))
            require(execution['cache_reuse_count']==len(rows)-len(fresh),'reuse summary')
        physical.extend(fresh);all_trace.extend(traces)
    for m,stream in expected_stream.items():require(observed_stream[m]==[r['start']['request_id'] for r in stream],'durable/condition strict order coverage')
    require(matrix['status']==('COMPLETE_WITH_INPUT_REJECTIONS' if reject_count else 'COMPLETE') and matrix['all_conditions_completed']==(reject_count==0),'matrix summary')
    check_physical(matrix['total_physical_cost'],physical,all_trace,reject_count==0)
    check_cost(matrix['known_completed_logical_service_cost'],sum_cost(logical))
    if reject_count:require(matrix['total_logical_service_cost'] is None,'partial service cost must stay null')
    else:
        check_cost(matrix['total_logical_service_cost'],sum_cost(logical))
        require(len(physical)==87 and sum(x['calls'] for x in logical)==131,'complete 87/131 geometry')
    require(matrix['returned_generations']==len(physical) and matrix['request_count_by_model']==dict(Counter(t['model'] for t in all_trace)),'matrix generation/route counts')
    return {'consumed_request_ids':consumed,'physical':node_cost(physical),'full_logical':None if reject_count else sum_cost(logical),
            'rejections':reject_count,'format_failure_logical_nodes':formats,'attempted_requests':len(all_trace)}

def rebuild_schedule(selected,lengths,workers):
    import numpy as np
    require(len(selected)==len(lengths)==384 and 2<=workers<=4,'schedule size')
    by_id={x['id']:x for x in lengths};require(len(by_id)==384 and set(by_id)=={x['id'] for x in selected},'length coverage')
    order=sorted(range(384),key=lambda i:(max(by_id[selected[i]['id']][k] for k in ('qwen_tokens','mistral_tokens')),selected[i]['id'].encode()))
    strata=[order[j*48:(j+1)*48] for j in range(8)];rng=np.random.Generator(np.random.PCG64(261113104))
    prefix=[s[int(rng.integers(0,48))] for s in strata];remainder=[i for i in range(384) if i not in set(prefix)]
    stratum={i:s for s,group in enumerate(strata) for i in group};units=[]
    for pos,i in enumerate(prefix+remainder):
        q=selected[i];units.append({'index':i,'position':pos,'worker':pos%workers,'phase':'prefix' if pos<8 else 'remainder',
            'length_stratum':stratum[i],'id':q['id'],'edge_index':q['edge_index'],'seed1':SEEDS[0],'seed2':SEEDS[1],
            'input_sha256':q['input_sha256'],'payload_sha256':by_id[q['id']]['payload_sha256']})
    return units,prefix,remainder,strata

def check_scientific_lineage(protocol_raw,selection,bindings,contract,auxiliary):
    """Selection binds the original design; execution may add resource bindings.

    Both contracts are immutable inputs. Execution-only amendments cannot alter
    the scientific keys checked below. No aliasing or replacement of old pins.
    """
    design_pin=bindings['scientific_protocol_sha256']
    require(selection['pins']['protocol_sha256']==design_pin,'selection design lineage')
    protocol=decode(protocol_raw)
    if sha(protocol_raw)==design_pin:
        return  # Same exact frozen contract; retained for single-stage protocols.
    require(contract.get('execution_protocol_sha256')==sha(protocol_raw),'execution protocol pin missing or different')
    require(protocol.get('original_scientific_design_sha256')==design_pin,'execution has no exact original design lineage')
    key=contract.get('scientific_design_key')
    require(isinstance(key,str) and key in auxiliary,'original scientific design bytes required')
    raw=auxiliary[key];require(sha(raw)==design_pin,'original scientific design bytes changed')
    design=decode(raw);require(design.get('status')=='FROZEN' and protocol.get('status')=='FROZEN','both contracts must be frozen')
    for field in ('scientific_scope','population','seeds','execution','failure_policy','statistics','evidence_rules','analysis_pins','selection_pins'):
        require(field in design and field in protocol and canonical(design[field])==canonical(protocol[field]),'scientific design changed: '+field)


def audit_bundle(*,plan_bytes,admission_bytes,contract_bytes,pinned_inputs,auxiliary,sources,archive,tokenizers,prior_receipts):
    """Input maps carry exact bytes. contract is an externally trusted pinned request.

    contract: expected_plan/admission/source_manifest/prior_ledger SHA, bindings,
    label_vault_sha256 (hash only), source_manifest_bytes is supplied in auxiliary
    under contract['source_manifest_key']; prior registry under prior_ledger_key.
    All archive files are required; caller must not omit orphan/failed artifacts.
    Success is REVIEW_REQUIRED and cannot be passed as APPROVED_FOR_SCORING.
    """
    contract=decode(contract_bytes);bindings=contract['bindings'];plan=decode(plan_bytes);admission=decode(admission_bytes)
    plan_sha=sha(plan_bytes);require(plan_sha==contract['plan_sha256']==bindings['plan_sha256'] and sha(admission_bytes)==contract['admission_sha256'],'trusted plan/admission pins')
    require(plan['schema']=='hotpot_collection_plan_v1' and plan['status']=='FROZEN_COLLECTION_PLAN','frozen plan required')
    require(admission['decision']=='APPROVED_FOR_FROZEN_COLLECTION' and admission['plan_sha256']==plan_sha and admission['attempt_id']==plan['attempt_id'],'admission lineage')
    require(admission['new_attempt_only'] is True and plan['attempt_id'] not in admission['previous_attempt_ids'] and admission['R1_closed'] is True and admission['qualification_source_bindings_checked'] is True,'admission scope')
    require(admission['budget_sha256']==newline_digest(plan['budget']),'budget admission')
    require(set(pinned_inputs)==set(plan['paths'])==set(plan['input_sha256']),'input coverage')
    require(all(sha(pinned_inputs[k])==v for k,v in plan['input_sha256'].items()),'frozen input changed')
    require(sha(pinned_inputs['selection'])==bindings['selection_manifest_sha256'],'selection source')
    protocol=decode(pinned_inputs['protocol']);selection=decode(pinned_inputs['selection']);length=decode(pinned_inputs['lengths'])
    check_scientific_lineage(pinned_inputs['protocol'],selection,bindings,contract,auxiliary)
    require(protocol['status']=='FROZEN' and protocol['population']['n_questions']==384 and protocol['population']['edge_indices']==list(PROBES),'scientific frozen population')
    require(set(REQUIRED_PINS)<=set(protocol['required_pins_before_execution']) and all(protocol['required_pins_before_execution'][k] not in (None,'','REQUIRED_BEFORE_ADMISSION') for k in REQUIRED_PINS),'scientific dependency pins')
    require(protocol['execution']['B_mask']==674 and protocol['execution']['A_mask']==4095 and protocol['execution']['no_edge_mask']==0 and protocol['execution']['rounds']==3,'scientific masks')
    require(plan['runtime']==decode(pinned_inputs['bindings']) and plan['budget']==decode(pinned_inputs['budget']),'embedded inputs differ')
    require(plan['budget']['allocation_authorized'] is True and protocol['resource']['cap_is_authorized'] is True and protocol['resource']['actual_cap_seconds']==plan['budget']['candidate_cap_seconds'] and protocol['resource']['per_worker_caps_seconds']==plan['budget']['worker_caps_seconds'],'resource freeze')
    require(selection['classification']=='PRIVATE_NO_LABELS' and selection['pins']['protocol_sha256']==bindings['scientific_protocol_sha256'] and selection['pins']['selector_sha256']==bindings['selector_sha256'],'selection source')
    receipt=decode(pinned_inputs['selection_receipt'])
    require(receipt['status']=='COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION' and receipt['selection_manifest_sha256']==sha(pinned_inputs['selection']) and receipt['questions_jsonl_sha256']==sha(pinned_inputs['questions']),'selection receipt')
    require(length['status']=='TOKENIZER_LENGTHS_COMPLETE_NO_LABELS' and length['labels_read'] is False and length['server_runtime_equivalence'] is True and length['payload_hash_algorithm']=='graph_credit_canonical_utf8_sha256_no_newline','length audit identity')
    require(length['questions_jsonl_sha256']==sha(pinned_inputs['questions']) and length['selection_manifest_sha256']==sha(pinned_inputs['selection']),'length source')
    require(length['collector_source_sha256']==plan['runtime']['collector']['sha256'],'collector/length source')
    selected=selection['questions'];rows=[decode(line) for line in pinned_inputs['questions'].splitlines()]
    require(len(rows)==len(selected)==384 and len({q['id'] for q in selected})==len({q['question_group_sha256'] for q in selected})==384,'384 unique question/groups')
    require(Counter(q['edge_index'] for q in selected)==Counter({e:96 for e in PROBES}),'edge balance')
    workers=plan['schedule']['worker_count'];units,prefix,remainder,strata=rebuild_schedule(selected,length['rows'],workers)
    require(plan['schedule']['units']==units and plan['schedule']['prefix_indices']==prefix and plan['schedule']['remainder_indices']==remainder and plan['schedule']['strata']==strata,'schedule reconstruction')
    by_index={u['index']:u for u in units};by_length={x['id']:x for x in length['rows']};payloads=[]
    for edge in PROBES:
        expected=[]
        for name in FIRST+SECOND:
            arch='independent12' if name=='INDEPENDENT12' else 'single_m' if name=='SINGLE_M' else 'graph'
            base={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(name)
            affected=closure(mask(base,edge) if base else None,mask(name,edge)) if arch=='graph' else set(ordering(name))
            expected.extend('|'.join((arch,model(node),node)) for node in ordering(name) if node in affected)
        require(Counter(plan['request_keys_by_edge'][str(edge)])==Counter(expected),'forecast geometry reconstruction')
    for i,(q,row) in enumerate(zip(selected,rows)):
        require((q['generation_seed1'],q['generation_seed2'])==SEEDS and q['id']==row['id'] and q['input_sha256']==newline_digest(row),'allowlisted source row lineage')
        payload=collector(row);require(digest(payload)==by_index[i]['payload_sha256'],'exact full context projection');payloads.append(payload)
        full=payload['question']+'\n\nSupplied context:\n'+payload['context']
        for m in ('qwen','mistral'):require(by_length[q['id']][m+'_tokens']==len(tokenizers[m].encode(full)),'length independent re-tokenization')
    # The source manifest is a wrapper over exact plan.sources, with no path rewriting.
    source_raw=auxiliary[contract['source_manifest_key']]
    require(sha(source_raw)==bindings['source_manifest_sha256']==contract['source_manifest_sha256'],'source manifest bytes')
    require(decode(source_raw)['source_files']==plan['sources']==plan['runtime']['source_files'],'source closure disagreement')
    require(len({r['path'] for r in plan['sources']})==len(plan['sources']) and set(sources)=={r['path'] for r in plan['sources']},'source inventory')
    require(all(sha(sources[r['path']])==r['sha256'] for r in plan['sources']),'source bytes changed')
    snapshot=decode(auxiliary[contract['runner_snapshot_key']]);require(sha(auxiliary[contract['runner_snapshot_key']])=='4ee76994c11089cb66e609e8ef438110aa4fabced21392b627c2246fd3d14449','public root runner source freeze')
    require(snapshot['original_snapshot_sha256']=='8eca7837d990257f2234cac86b2590b69ce802cb0d7441ebb5b7f03e9da34ba8','original runner snapshot002 lineage')
    for name in ('parent.py','worker.py','runtime_factory.py','plan_tools.py','orchestration_core.py','storage.py','production_guard.py'):
        matches=[raw for path,raw in sources.items() if path.replace('\\','/').rsplit('/',1)[-1]==name]
        require(len(matches)==1 and sha(matches[0])==snapshot['source_files'][name],'actual runner source')
    for spec in plan['runtime']['modules'].values():require(spec['path'] in sources and sha(sources[spec['path']])==spec['sha256'],'module closure')
    spec=plan['runtime']['collector'];require(spec['path'] in sources and sha(sources[spec['path']])==spec['sha256'],'collector source closure')
    require(plan['runtime']['expected_matrix_protocol_hash']==bindings['matrix_protocol_hash'],'matrix identity differs')
    const={}
    for basename in ('executor.py','independent12_executor.py','matrix_executor.py'):
        matches=[raw for path,raw in sources.items() if path.replace('\\','/').rsplit('/',1)[-1]==basename]
        require(len(matches)==1,'ambiguous constant source');const=literal_constants(matches[0],const)
    engineering=plan['runtime']['engineering_admission'];eng_raw=auxiliary[engineering['path']]
    require(sha(eng_raw)==engineering['sha256'],'engineering acceptance pin');eng=decode(eng_raw)
    require(eng['decision']=='PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION' and eng['matrix_protocol_hash']==bindings['matrix_protocol_hash'],'engineering acceptance scope')
    for m in ('qwen','mistral'):
        identity=plan['runtime']['models'][m]['identity'];provenance=tokenizers[m].provenance
        require(eng['model_identity_sha256'][m]==digest(identity)==provenance['backend_identity_sha256'],'tokenizer/qualification model identity')
        require(length['model_identity_sha256'][m]==digest(identity) and length['tokenizer_identity_sha256'][m]==plan['runtime']['models'][m]['tokenizer_identity_sha256'],'length backend identity')
        require(provenance['chat_template_reexecuted'] is True,'chat template evidence missing')
    for name in ('matrix','durable'):require(eng['module_sha256'][name]==plan['runtime']['modules'][name]['sha256'],'engineering module identity')
    prior=decode(auxiliary[contract['prior_ledger_key']]);require(sha(auxiliary[contract['prior_ledger_key']])==contract['prior_ledger_sha256'],'external registry pin')
    resident=check_residence(plan,plan_sha,archive,prior,prior_receipts)
    require(any(t['transaction_id']==eng['candidate_transaction_id'] and t['seconds']==eng['candidate_closed_seconds'] for t in plan['budget']['prior_candidate_transactions']),'qualification charged exactly once')
    raw_manifest=decode(archive['RAW_OUTPUT_MANIFEST.json']);attempt=decode(archive['ATTEMPT_RESULT.json'])
    require(raw_manifest['status']=='CLOSED_SNAPSHOT' and raw_manifest['plan_sha256']==plan_sha and raw_manifest['scores_accessed'] is False,'raw manifest closure')
    excluded={'RAW_OUTPUT_MANIFEST.json','PARENT_RESIDENCE.json','ATTEMPT_RESULT.json'}
    names={name for name in archive if name.rsplit('/',1)[-1]!='PROGRESS.json' and name not in excluded}
    require(set(raw_manifest['files'])==names,'archive orphan/missing inventory')
    require(all(sha(archive[name])==pin for name,pin in raw_manifest['files'].items()),'raw inventory changed')
    require(not any(name.rsplit('/',1)[-1] in ('FAILED.json','PARENT_FAILED.json','PARTIAL_MATRIX.json','DURABLE_ERROR_EVIDENCE.json') for name in archive),'failed attempt artifact present')
    require(not any('.pending-' in name.rsplit('/',1)[-1] or name.endswith('.next') for name in archive),'unresolved atomic-write artifact')
    require(attempt['raw_manifest_sha256']==sha(archive['RAW_OUTPUT_MANIFEST.json']) and attempt['residence_sha256']==sha(archive['PARENT_RESIDENCE.json']),'final metadata ancestry')
    require(decode(archive['ATTEMPT_STARTED.json'])['admission_sha256']==sha(admission_bytes),'started admission binding')
    calls={};rejects=[];runtime_by_worker={}
    for w in range(workers):
        for m in ('qwen','mistral'):
            header=f'worker_{w}/';loaded=decode(archive[header+'MODEL_LOAD_COMPLETE_'+m+'.json']);load_start=decode(archive[header+'MODEL_LOAD_STARTED_'+m+'.json'])
            identity=plan['runtime']['models'][m]['identity'];require(loaded['model']==load_start['model']==m and loaded['worker']==load_start['worker']==w and loaded['identity']==identity,'load identity')
            residence=decode(archive['PARENT_RESIDENCE.json']);begin=residence['worker_started_monotonic'][str(w)];end=residence['worker_closed_monotonic'][str(w)]
            require(begin<=load_start['monotonic']<=load_start['monotonic']+loaded['wall_seconds']<=end+1e-6,'load outside residence')
            c,r=parse_journal(archive[header+m+'_raw_requests.jsonl'],w,m,plan,plan_sha,by_index,tokenizers[m])
            require(not set(c)&set(calls),'cross-worker physical call ID collision');calls.update(c);rejects.extend(r)
            require(all(raw['native']['runtime']==loaded['runtime_private'] for raw in c.values()),'native runtime/load binding')
        rt=decode(archive[f'worker_{w}/RUNTIME_IDENTITY.json']);require(rt['matrix_protocol_hash']==bindings['matrix_protocol_hash'] and rt['durable_helper_sha256']==plan['runtime']['modules']['durable']['sha256'] and rt['model_source_binding_checked'] is True,'worker runtime identity')
        runtime_by_worker[w]=rt
    results=[];review_units=[];consumed=[];unit_receipts={}
    for i in range(384):
        unit=by_index[i];directory=f"worker_{unit['worker']}/unit_{i:03d}";inp=decode(archive[directory+'/INPUT.json']);matrix=decode(archive[directory+'/MATRIX.json']);receipt=decode(archive[directory+'/UNIT_RECEIPT.json'])
        require(inp=={'plan_sha256':plan_sha,'unit':unit,'payload':payloads[i]},'collector actual input differs from source')
        r=check_matrix(matrix,payloads[i],unit,bindings,const,tokenizers,calls,rejects);results.append(r);consumed.extend(r['consumed_request_ids']);unit_receipts[i]=receipt
        require(receipt['schema']=='hotpot_unit_receipt_v1' and receipt['index']==i and receipt['worker']==unit['worker'] and receipt['plan_sha256']==plan_sha and receipt['matrix_protocol_hash']==bindings['matrix_protocol_hash'],'receipt identity')
        require(receipt['input_file_sha256']==sha(archive[directory+'/INPUT.json']) and receipt['matrix_sha256']==sha(archive[directory+'/MATRIX.json']),'receipt exact bytes')
        require(receipt['terminal_keys']==list(ORDER) and receipt['terminal_conditions']==11 and receipt['scores_computed'] is False and receipt['overflow_conditions']==r['rejections'] and receipt['status']==matrix['status'],'receipt coverage')
        require(receipt['total_physical_cost']==matrix['total_physical_cost'] and receipt['cost_fields_complete']==(r['rejections']==0),'receipt cost evidence')
        review_units.append({'index':i,'directory':directory,'sha256':{f:sha(archive[directory+'/'+f]) for f in ('INPUT.json','MATRIX.json','UNIT_RECEIPT.json')}})
        check_profile(matrix,unit,receipt['cost_profile'])
    allraw=[r['start']['request_id'] for r in list(calls.values())+rejects]
    require(len(consumed)==len(set(consumed))==len(allraw) and set(consumed)==set(allraw),'whole archive raw request bijection')
    rejections=sum(r['rejections'] for r in results)
    require(attempt['completed_units']==384 and attempt['terminal_conditions']==4224 and attempt['overflow_conditions']==rejections and attempt['all_worker_residences_closed'] is True,'attempt coverage')
    require(attempt['scores_computed'] is False and attempt['score_release_authorized'] is False and attempt['automatic_retry'] is False,'collector evaluation authority')
    require(attempt['status']==('COMPLETE_WITH_CAPACITY_FAILURES' if rejections else 'COLLECTION_COMPLETE_REQUIRES_LEDGER_AND_SOURCE_AUDIT'),'attempt status')
    # Prefix/release bind exact source receipts; forecast arithmetic audited below.
    for w in range(workers):
        completed=decode(archive[f'worker_{w}/WORKER_COMPLETE.json']);assigned=[u for u in units if u['worker']==w]
        require(completed['status']=='COMPLETE_COLLECTOR_ONLY' and completed['plan_sha256']==plan_sha and completed['worker']==w and completed['units']==len(assigned) and completed['terminal_conditions']==11*len(assigned) and completed['scores_computed'] is False,'worker closure')
        wall=sum(unit_receipts[u['index']]['cost_profile']['unit_wall_seconds'] for u in assigned)
        loads=sum(x['load_wall_seconds'] for x in runtime_by_worker[w]['loads'])
        require(wall+loads<=resident['seconds_by_worker'][w]+1e-6,'sequential work exceeds resident interval')
    check_forecast(plan,plan_sha,archive,unit_receipts,resident)
    template={'schema':'hotpot_closed_archive_score_release_v1','decision':'REVIEW_REQUIRED','bindings':bindings,
        'scientific_protocol_status':'FROZEN','reviewed_scopes':list(REVIEW_SCOPES),'unique_questions':384,'terminal_conditions':4224,
        'runtime_failure_count':0,'all_attempts_closed':True,'physical_ledger_closed':True,'label_vault_sha256':contract.get('label_vault_sha256'),
        'input_overflow_conditions':rejections,'archive_metadata_sha256':{name:sha(archive[name]) for name in METADATA},'units':review_units,
        'resident_ledger_closed':True,'all_native_cost_fields_complete':not rejects,
        'physical_ledger_closed_definition':'All request observations and parent residence closed; proved preflight rejection components may remain null.',
        'unknown_preflight_component_requests':len(rejects),
        'root_must_acknowledge_unknown_preflight_components_before_release':bool(rejects)}
    require(template['label_vault_sha256'] is None or valid_sha(template['label_vault_sha256']),'invalid hash-only label vault pin')
    return {'status':'REVIEW_REQUIRED','external_release_template':template,
        'physical_success_cost':sum_cost(r['physical'] for r in results),'physical_request_count':len(allraw),'returned_generations':len(calls),
        'preflight_rejections_with_no_native_record':len(rejects),'full_logical_cost':None if rejections else sum_cost(r['full_logical'] for r in results),
        'residence':resident,'contract_sha256':sha(contract_bytes),'synthetic_tokenizer_witness':any(t.provenance.get('synthetic_only',False) for t in tokenizers.values()),
        'no_label_vault_read':True,'label_vault_binding_pending':template['label_vault_sha256'] is None,
        'label_release_authorized':False,'GPU_admission_granted':False,
        'limitations':['The external prior-transaction registry completeness is trusted, not provable from this attempt.',
            'Native generated token counts and CUDA time are cross-reconciled; token IDs and GPU execution are not regenerated.',
            'Durable events lack per-request monotonic timestamps; no claim of exact request containment in the parent residence interval.',
            'Input-source bytes and signatures are externally pinned; this is not a pretraining-contamination certificate.',
            'Model weight manifests are bound to qualified identities; this CPU audit does not hash all model weights again.']}

def check_profile(matrix,unit,p):
    maximum={};elapsed=0.;missing=0
    for c in matrix['conditions']:
        arch='independent12' if c['name']=='INDEPENDENT12' else 'single_m' if c['name']=='SINGLE_M' else 'graph'
        for t in c['request_trace']:
            g=t['generation']
            if g is None or not finite(g.get('elapsed_seconds')):missing+=1;continue
            key='|'.join((arch,t['model'],t['node']));maximum[key]=max(maximum.get(key,0),g['elapsed_seconds']);elapsed+=g['elapsed_seconds']
    require((p['index'],p['worker'],p['length_stratum'])==(unit['index'],unit['worker'],unit['length_stratum']),'profile unit identity')
    require(p['request_max_seconds']==maximum and p['missing_request_timings']==missing and close(p['observed_request_seconds_sum'],elapsed),'profile request times')
    require(finite(p['unit_wall_seconds']) and math.isclose(p['raw_wall_residual_seconds'],p['unit_wall_seconds']-elapsed,rel_tol=1e-12,abs_tol=1e-8),'profile wall residual')
    require(close(p['nonnegative_wall_residual_seconds'],max(0.,p['unit_wall_seconds']-elapsed)),'profile nonnegative residual')
    require(p['status']==matrix['status'] and p['terminal_conditions']==11 and p['overflow_conditions']==sum(c['status']=='INPUT_REJECTED' for c in matrix['conditions']) and p['cost_fields_complete']==matrix['total_physical_cost']['cost_fields_complete'],'profile summary')

def check_forecast(plan,plan_sha,archive,receipts,resident):
    """Independent fixed-prefix accounting; no scientific outcomes are inspected."""
    schedule=plan['schedule'];workers=schedule['worker_count'];budget=plan['budget'];profiles=[];pins=[]
    for w in range(workers):
        raw=archive[f'worker_{w}/PREFIX_COMPLETE.json'];p=decode(raw);pins.append(sha(raw))
        require(p['worker']==w and p['plan_sha256']==plan_sha,'prefix identity');profiles.extend(p['profiles'])
    require(len(profiles)==8 and {p['index'] for p in profiles}==set(schedule['prefix_indices']),'fixed prefix8')
    by_stratum={p['length_stratum']:p for p in profiles};require(set(by_stratum)==set(range(8)),'prefix strata')
    own={w:[p for p in profiles if p['worker']==w] for w in range(workers)}
    for p in profiles:
        require(p==receipts[p['index']]['cost_profile'] and p['status']=='COMPLETE' and p['terminal_conditions']==11 and p['overflow_conditions']==0 and p['cost_fields_complete'] is True and p['missing_request_timings']==0,'unsupported prefix forecast')
    frozen=decode(archive['FORECAST_FROZEN.json']);release=decode(archive['RELEASE_REMAINDER.json'])
    require(frozen['plan_sha256']==plan_sha and frozen['attempt_id']==plan['attempt_id'] and frozen['prefix_receipt_sha256']==pins,'forecast pins')
    require(release['plan_sha256']==plan_sha and release['attempt_id']==plan['attempt_id'] and release['forecast_sha256']==sha(archive['FORECAST_FROZEN.json']) and release['release_recheck_fits'] is True and release['scores_accessed'] is False,'blind barrier release')
    margin=plan['forecast_margin'];require(finite(margin) and margin>=.25,'forecast margin')
    remaining=[budget['release_guard_seconds']]*workers
    for u in schedule['units']:
        if u['phase']=='prefix':continue
        sample=by_stratum[u['length_stratum']];keys=plan['request_keys_by_edge'][str(u['edge_index'])]
        require(len(keys)==87,'forecast geometry')
        value=sum(max(sample['request_max_seconds'][k],max(p['request_max_seconds'][k] for p in own[u['worker']] if k in p['request_max_seconds'])) for k in keys)
        value+=max(sample['nonnegative_wall_residual_seconds'],max(p['nonnegative_wall_residual_seconds'] for p in own[u['worker']]))
        remaining[u['worker']]+=value*(1+margin)
    current=frozen['current_resident_seconds_by_worker'];total=[a+b for a,b in zip(current,remaining)]
    require(len(current)==workers and all(finite(x) for x in current),'forecast residence inputs')
    for key,expected in [('predicted_remaining_seconds_by_worker',remaining),('predicted_attempt_seconds_by_worker',total)]:
        require(len(frozen[key])==workers and all(close(a,b) for a,b in zip(frozen[key],expected)),'forecast arithmetic')
    require(frozen['status']=='FIT' and close(frozen['predicted_attempt_sum_seconds'],sum(total)),'forecast total')
    recheck=release['release_recheck_resident_seconds'];require(len(recheck)==workers,'release recheck size')
    require(all(current[w]<=recheck[w]<=resident['seconds_by_worker'][w] and recheck[w]+remaining[w]<=budget['worker_caps_seconds'][w] for w in range(workers)),'release caps/time order')
    require(sum(recheck)+sum(remaining)<=budget['global_available_seconds_for_attempt'] and sum(recheck)+sum(remaining)+sum(t['seconds'] for t in budget['prior_candidate_transactions'])<=budget['candidate_cap_seconds'],'release full budget')
