"""Pure CPU audit of synthetic full-matrix qualification outputs."""
import math

NODE_INTEGRITY_FIELDS=('node','input_fingerprint','seed','messages','generation','wire_message','parsed',
                       'message_metadata','received_message_tokens','self_state_tokens')
GENERATION_SEMANTICS=('text','input_tokens','output_tokens','finish_reason')
COST_SEMANTICS=('calls','planned_calls','unattempted_calls','returned_generations','input_tokens','output_tokens',
                'received_message_tokens','self_state_tokens','cost_fields_complete','service_complete',
                'failed_requests_without_physical_record','call_count_scope')
NODE_SEMANTICS=('node','input_fingerprint','seed','messages','wire_message','parsed','message_metadata',
                'received_message_tokens','self_state_tokens','reused')


def nodes(execution):
    if 'nodes' in execution:return execution['nodes']
    return {slot['node']:slot['record'] for slot in execution['slots']}


def semantic_cost(cost):
    return {key:cost[key] for key in COST_SEMANTICS if key in cost}


def semantic_view(matrix):
    lineage={c['execution']['execution_id']:c['key'] for c in matrix['conditions'] if c['execution'] is not None}
    result={key:matrix[key] for key in ('question_id','question_fingerprint','seed1','seed2','probe_edge','matrix_protocol_hash','status','operator_id')}
    result['conditions']=[]
    for condition in matrix['conditions']:
        item={key:condition[key] for key in ('key','name','seed_number','seed','mask','nominal_physical_calls',
                                           'planned_physical_calls','status','replay_mode','cache_source_condition')}
        execution=condition['execution']
        item['execution_header']={key:execution[key] for key in ('question_id','question_fingerprint','seed','protocol_hash','answer','cache_reuse_count') if key in execution}
        for key in ('mask','closure','control_id','format_failures'):
            if key in execution:item['execution_header'][key]=execution[key]
        item['nodes']={}
        for name,record in nodes(execution).items():
            row={key:record[key] for key in NODE_SEMANTICS}
            source=record['source_execution_id']
            row['source_condition']=lineage[source] if source is not None else None
            row['generation']={key:record['generation'][key] for key in GENERATION_SEMANTICS}
            item['nodes'][name]=row
        if 'slots' in execution:
            item['slots']=[{key:s[key] for key in ('index','node','role','sample','model','seed','status','attempted_backend_request')}
                           for s in execution['slots']]
        item['physical_cost']=semantic_cost(condition['physical_cost'])
        item['logical_service_cost']=semantic_cost(condition['logical_service_cost'])
        result['conditions'].append(item)
    result['total_physical_cost']=semantic_cost(matrix['total_physical_cost'])
    result['total_logical_service_cost']=semantic_cost(matrix['total_logical_service_cost'])
    result['request_count_by_model']=matrix['request_count_by_model']
    return result


def verify_matrix(matrix,question,protocol,digest):
    expected_names=protocol['seed1_order']+protocol['seed2_order']
    expected_counts=protocol['physical_calls_by_condition']
    assert matrix['status']=='COMPLETE' and matrix['condition_set_terminal'] and matrix['all_conditions_completed']
    assert matrix['seed1']==protocol['seed1'] and matrix['seed2']==protocol['seed2'] and matrix['probe_edge']==1
    assert matrix['question_id']==question['id'] and matrix['question_fingerprint']==digest(question)
    assert [c['name'] for c in matrix['conditions']]==expected_names and len(matrix['conditions'])==11
    assert [c['physical_cost']['calls'] for c in matrix['conditions']]==expected_counts
    assert matrix['total_physical_cost']['calls']==matrix['returned_generations']==87
    assert matrix['total_logical_service_cost']['calls']==131 and matrix['request_count_by_model']=={'qwen':40,'mistral':47}
    assert matrix['total_physical_cost']['cost_fields_complete']
    assert matrix['resident_GPU_hours'] is None and not matrix['scores_computed']
    full=question['question']+'\n\nSupplied context:\n'+question['context']
    lineage={c['execution']['execution_id']:c['key'] for c in matrix['conditions']}
    physical={row['generation']['call_id']:row for c in matrix['conditions'] for row in c['request_trace']}
    assert len(physical)==87
    logical_costs=[]
    for index,condition in enumerate(matrix['conditions']):
        assert condition['status']=='COMPLETE' and condition['error_type'] is None
        execution=condition['execution'];records=nodes(execution)
        assert len(records)==(1 if condition['name']=='SINGLE_M' else 13)
        assert condition['physical_cost']['cost_fields_complete'] and condition['physical_cost']['service_complete']
        assert condition['physical_cost']['calls']==len(condition['request_trace'])
        assert condition['logical_service_cost']['calls']==len(records)
        # Reconstruct logical service from every node, including reused nodes,
        # in the frozen execution order rather than relying on fresh equality.
        ordered=[records[node] for node in ([f'r{r}:w{w}' for r in range(1,4) for w in range(4)]+['aggregate']) if node in records]
        if 'slots' in execution:ordered=[s['record'] for s in execution['slots']]
        logical={'calls':len(ordered),
            'input_tokens':sum(r['generation']['input_tokens'] for r in ordered),
            'output_tokens':sum(r['generation']['output_tokens'] for r in ordered),
            'received_message_tokens':sum(r['received_message_tokens'] for r in ordered),
            'self_state_tokens':sum(r['self_state_tokens'] for r in ordered),
            'model_elapsed_seconds':sum(r['generation']['elapsed_seconds'] for r in ordered),
            'gpu_seconds':sum(r['generation']['gpu_seconds'] for r in ordered),
            'peak_allocated_bytes':max(r['generation']['gpu_peak_allocated_bytes'] for r in ordered)}
        for key,value in logical.items():
            assert math.isclose(condition['logical_service_cost'][key],value,rel_tol=1e-12,abs_tol=1e-9),('logical',condition['key'],key)
        logical_costs.append(logical)
        expected_seed=protocol['seed1'] if index<7 else protocol['seed2']
        assert condition['seed']==execution['seed']==expected_seed
        for node,record in records.items():
            assert record['messages'][1]['role']=='user' and record['messages'][1]['content'].startswith(full)
            assert record['output_digest']==digest({key:record[key] for key in NODE_INTEGRITY_FIELDS})
            derived=int(digest([question['id'],expected_seed,node])[:16],16)%(2**31-1)
            cap=256 if node=='aggregate' else 128
            assert record['seed']==derived
            assert record['input_fingerprint']==digest({'question':digest(question),'node':node,'messages':record['messages'],
                'seed':derived,'protocol':execution['protocol_hash'],'max_new_tokens':cap})
            assert record['generation']['input_tokens']<=8192 and record['generation']['output_tokens']<=cap
            assert record['message_metadata']['message_tokens']<=cap
            source=record['source_execution_id']
            if record['reused']:
                assert source is not None and lineage[source]==condition['cache_source_condition']
            else:assert source is None
            raw=physical[record['generation']['call_id']]
            assert raw['generation']==record['generation']
            assert raw['node']==node and raw['seed']==record['seed'] and raw['messages']==record['messages']
        for key in ('input_tokens','output_tokens','gpu_seconds'):
            assert math.isclose(condition['physical_cost'][key],sum(r['generation'][key] for r in condition['request_trace']),rel_tol=1e-12,abs_tol=1e-9)
        assert math.isclose(condition['physical_cost']['model_elapsed_seconds'],sum(r['generation']['elapsed_seconds'] for r in condition['request_trace']),rel_tol=1e-12,abs_tol=1e-9)
    for key in logical_costs[0]:
        value=max(c[key] for c in logical_costs) if key=='peak_allocated_bytes' else sum(c[key] for c in logical_costs)
        assert math.isclose(matrix['total_logical_service_cost'][key],value,rel_tol=1e-12,abs_tol=1e-9),('total_logical',key)
    return len(physical)


def compare_fresh(first,second):
    a,b=semantic_view(first),semantic_view(second)
    assert a==b,'Fresh matrices differ in declared semantic inputs, outputs, tokens, cache or counts'
    return {'conditions_compared':11,'logical_node_records_compared':131,'comparison':'semantic_view_exact',
            'excluded':'execution/call IDs, timing, output_digest and peak allocation; each own digest verified separately'}


def verify_global_calls(matrices,router,backends,protocol,digest):
    routes=router.route_events;physical=[(name,row) for name,b in backends.items() for row in b.physical_calls]
    assert len(routes)==len(physical)==174
    assert all(r['status']=='SUCCESS' for r in routes) and all(r['status']=='SUCCESS' for _,r in physical)
    assert {name:len(b.physical_calls) for name,b in backends.items()}=={'qwen':80,'mistral':94}
    route_by_id={r['call_id']:r for r in routes};physical_by_id={r['call_id']:(name,r) for name,r in physical}
    assert len(route_by_id)==len(physical_by_id)==174 and set(route_by_id)==set(physical_by_id)
    traces=[row for matrix in matrices for c in matrix['conditions'] for row in c['request_trace']]
    assert len(traces)==len({r['generation']['call_id'] for r in traces})==174
    formats=[]
    for trace in traces:
        gen=trace['generation'];call=gen['call_id'];route=route_by_id[call];name,row=physical_by_id[call]
        assert name==route['model']==trace['model'] and route['node']==trace['node']
        assert route['model_identity_sha256']==digest(backends[name].identity)
        assert route['seed']==row['seed']==trace['seed']
        assert route['prompt_sha256']==row['prompt_sha256']==digest(trace['messages'])
        cap=256 if trace['node']=='aggregate' else 128
        assert route['max_new_tokens']==row['generation_config']['max_new_tokens']==cap
        for key in ('call_id','input_tokens','output_tokens','elapsed_seconds','gpu_seconds','gpu_peak_allocated_bytes','finish_reason'):
            assert gen[key]==row[key]
        assert trace['record'] is not None
        formats.append({'model':name,'node':trace['node'],'format_ok':trace['record']['message_metadata']['format_ok']})
    total={key:sum(row[key] for _,row in physical) for key in ('input_tokens','output_tokens','gpu_seconds')}
    total['model_elapsed_seconds']=sum(row['elapsed_seconds'] for _,row in physical)
    for key,value in total.items():
        assert math.isclose(value,sum(m['total_physical_cost'][key] for m in matrices),rel_tol=1e-12,abs_tol=1e-9)
    total.update(calls=174,resident_GPU_hours=None)
    return total,formats


def verify_durable(matrices,backends,journal_paths,digest):
    import dataclasses,hashlib
    from pathlib import Path
    import durable_backend
    from graph_credit.executor import GraphConfig
    from durable_backend import read_journal,canonical
    traces={r['generation']['call_id']:r for m in matrices for c in m['conditions'] for r in c['request_trace']}
    repeats={r['generation']['call_id']:index for index,m in enumerate(matrices,1) for c in m['conditions'] for r in c['request_trace']}
    helper_sha=hashlib.sha256(Path(durable_backend.__file__).read_bytes()).hexdigest()
    assert len(traces)==174
    seen=set();requests=0
    for model,path in journal_paths.items():
        journal=read_journal(path)
        assert not journal['pending_request_ids']
        assert journal['header']['backend_identity']==backends[model].identity
        assert journal['header']['model']==model
        assert journal['header']['helper_sha256']==helper_sha
        native=[]
        for row in journal['requests'].values():
            start=row['start'];terminal=row['terminal'];requests+=1
            assert terminal['event']=='RETURNED'
            generation=terminal['generation'];call=generation['call_id']
            assert call in traces and call not in seen;seen.add(call)
            trace=traces[call]
            assert canonical(generation)==canonical(trace['generation'])
            assert start['messages']==trace['messages'] and start['seed']==trace['seed']
            assert start['model']==trace['model']==model
            assert start['context']['node']==trace['node']
            assert start['context']['repeat']==repeats[call]
            assert start['max_new_tokens']==(256 if trace['node']=='aggregate' else 128)
            assert canonical(start['config'])==canonical(dataclasses.asdict(GraphConfig(rounds=3)))
            assert start['backend_identity_sha256']==digest(backends[model].identity)
            assert terminal['native_event_start']==len(native)
            native.extend(terminal['native_events'])
            assert terminal['native_event_end']==len(native)
        assert canonical(native)==canonical(backends[model].physical_calls)
    assert requests==174 and seen==set(traces)
    return {'requests':174,'canonical_generation_bindings':174,'pending':0,'native_events_exact':True}
