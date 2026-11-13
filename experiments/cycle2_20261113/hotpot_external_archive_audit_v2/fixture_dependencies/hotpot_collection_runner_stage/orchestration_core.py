"""Pure orchestration math and receipts. No model, labels or real-data IO."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import numpy as np

N=384
EDGES=(1,5,7,9)
SEEDS=(26111301,26111302)
FIRST=('A','B','NO_EDGE','A_MINUS_E','B_MINUS_E','INDEPENDENT12','SINGLE_M')
SECOND=('A','B','A_MINUS_E','B_MINUS_E')
TERMINALS=tuple(f'S{s}/{name}' for s,names in ((1,FIRST),(2,SECOND)) for name in names)
FINITE_METRICS=('input_tokens','output_tokens','model_elapsed_seconds','gpu_seconds')
REQUIRED_PINS=('protocol_commit','model_files','tokenizer_files','graph_executor_source',
 'independent12_operator_source','single_operator_source','node_seed_derivation','prompts_and_formatter',
 'official_hotpot_scorer','local_answer_adapter','runtime_environment','qualified_operator_receipts',
 'source_train_manifest','payload_normalizer','known_exposure_exclusions','selected_question_manifest',
 'edge_assignment_manifest','label_vault_access_control','budget_and_worker_cap_receipt','R1_closure_receipt')
EXPECTED_EXECUTION={'worker_models':['Qwen2.5-3B-Instruct','Mistral-7B-Instruct-v0.3','Qwen2.5-3B-Instruct','Mistral-7B-Instruct-v0.3'],
 'aggregator':'Mistral-7B-Instruct-v0.3','rounds':3,'worker_max_new_tokens':128,
 'aggregator_and_single_max_new_tokens':256,'input_limit_tokens':8192,'temperature':.7,'top_p':.9,'do_sample':True,
 'A_mask':4095,'B_mask':674,'no_edge_mask':0,'conditions_total':4224,'graph_conditions':3456,
 'independent12_conditions':384,'single_conditions':384,'expected_all_success_physical_calls':33408,
 'expected_all_success_independent_logical_calls':50304,'graph_service_logical_calls':13,
 'physical_call_estimate_is_hard_bound':False,'peer_edge_deletion_removes_worker_calls':False}


def require(ok,message):
    if not ok: raise ValueError(message)


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def graph_digest(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def is_sha(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def finite(value): return type(value) in (int,float) and math.isfinite(value) and value>=0


def validate_protocol(protocol, *, executable=False):
    if executable:
        require(protocol['status']=='FROZEN','Scientific protocol is not frozen')
        pins=protocol['required_pins_before_execution']
        require(set(REQUIRED_PINS)<=set(pins),'Required scientific pin keys missing')
        require(all(pins[k] not in (None,'','REQUIRED_BEFORE_ADMISSION') for k in REQUIRED_PINS),'Required scientific pin value missing')
    require(protocol['population']['n_questions']==N,'Exactly 384 questions required')
    require(protocol['population']['edge_indices']==list(EDGES),'Edges differ')
    x=protocol['execution'];s=protocol['seeds']
    require(all(k in x and canonical(x[k])==canonical(v) for k,v in EXPECTED_EXECUTION.items()),
            'Scientific execution fields differ from unchanged qualified operator')
    require((s['generation_seed1'],s['generation_seed2'])==SEEDS,'Generation seeds differ')
    require(s['blind_prefix_pcg64']==261113104,'Blind prefix seed differs')
    require(x['conditions_total']==4224 and x['rounds']==3 and x['input_limit_tokens']==8192,'Frozen matrix differs')
    require(x['seed1_sequence']==list(FIRST) and x['seed2_sequence']==list(SECOND),'Condition order differs')
    require(protocol['resource']['blind_benchmark_prefix_units']==8,'Prefix length differs')
    require(protocol['resource']['minimum_forecast_time_margin']>=.25,'Forecast margin too small')


def build_schedule(rows,lengths,worker_count):
    require(type(worker_count) is int and 2<=worker_count<=4,'Require 2-4 workers')
    require(len(rows)==N and len(lengths)==N,'Fixed 384 rows required')
    require(len({r['id'] for r in rows})==N,'Duplicate selected ID')
    require(Counter(r['edge_index'] for r in rows)==Counter({e:96 for e in EDGES}),'Edge balance differs')
    lookup={r['id']:r for r in lengths}
    require(len(lookup)==N and set(lookup)=={r['id'] for r in rows},'Length identity coverage differs')
    for r in rows:
        require((r['generation_seed1'],r['generation_seed2'])==SEEDS,'Manifest seed differs')
        length=lookup[r['id']]
        require(all(type(length[k]) is int and length[k]>0 for k in ('qwen_tokens','mistral_tokens')),'Invalid tokenizer lengths')
        require(is_sha(length['payload_sha256']) and is_sha(r['input_sha256']),'Missing input hashes')
    order=sorted(range(N),key=lambda i:(max(lookup[rows[i]['id']][k] for k in ('qwen_tokens','mistral_tokens')),rows[i]['id'].encode()))
    strata=[order[i*48:(i+1)*48] for i in range(8)]
    rng=np.random.Generator(np.random.PCG64(261113104))
    prefix=[group[int(rng.integers(0,48))] for group in strata]
    remainder=[i for i in range(N) if i not in set(prefix)]
    sequence=prefix+remainder;which={i:s for s,group in enumerate(strata) for i in group}
    units=[]
    for position,i in enumerate(sequence):
        r=rows[i]
        units.append({'index':i,'position':position,'worker':position%worker_count,'phase':'prefix' if position<8 else 'remainder',
                      'length_stratum':which[i],'id':r['id'],'edge_index':r['edge_index'],
                      'seed1':SEEDS[0],'seed2':SEEDS[1],'input_sha256':r['input_sha256'],
                      'payload_sha256':lookup[r['id']]['payload_sha256']})
    return {'units':units,'prefix_indices':prefix,'remainder_indices':remainder,
            'strata':strata,'worker_count':worker_count,'parallel_order_scope':'fixed global dispatch list; each worker preserves its assigned subsequence; completion order may differ'}


def validate_budget(budget,worker_count,*,executable=False):
    keys=('candidate_cap_seconds','global_available_seconds_for_attempt','disk_min_free_bytes','release_guard_seconds')
    require(all(finite(budget.get(k)) for k in keys),'Missing explicit budget; no 48h default')
    require(budget['candidate_cap_seconds']>0 and budget['global_available_seconds_for_attempt']>0,'Positive signed caps required')
    caps=budget['worker_caps_seconds']
    require(len(caps)==worker_count and all(finite(c) and c>0 for c in caps),'Per-worker caps required')
    require(budget['disk_min_free_bytes']>0 and budget['release_guard_seconds']>=1,'Disk/release guard required')
    transactions=budget.get('prior_candidate_transactions',[])
    require(len({t['transaction_id'] for t in transactions})==len(transactions),'Duplicate prior candidate transaction')
    require(all(finite(t['seconds']) and t['already_in_global_ledger'] is True for t in transactions),'Prior costs must have a closed global-ledger lineage')
    prior=sum(t['seconds'] for t in transactions)
    require(prior+sum(caps)<=budget['candidate_cap_seconds'],'Reservations exceed candidate remainder')
    require(sum(caps)<=budget['global_available_seconds_for_attempt'],'Worker caps exceed global available remainder')
    if executable: require(budget.get('allocation_authorized') is True,'No resource allocation authorization')
    return prior


def normalized_terminal(row):
    if row['status']=='COMPLETE': return 'COMPLETE'
    if row['status']=='INPUT_REJECTED': return 'FAIL_INPUT_OVERFLOW'
    return 'INCOMPLETE'


def validate_matrix_result(data,unit):
    require(data['seed1']==unit['seed1'] and data['seed2']==unit['seed2'] and data['probe_edge']==unit['edge_index'],'Matrix seed/edge binding differs')
    require(data['question_id']=='hotpotqa_'+unit['id'],'Collector identity differs')
    require(data['question_fingerprint']==unit['payload_sha256'],'Collector payload fingerprint differs')
    require([c['key'] for c in data['conditions']]==list(TERMINALS),'Unique 11-condition order required')
    require(data.get('scores_computed') is False,'Collector must not score')
    for c,key in zip(data['conditions'],TERMINALS):
        seed_label,name=key.split('/');number=int(seed_label[1:])
        require(c['name']==name and c['seed_number']==number and c['seed']==SEEDS[number-1],'Condition key/name/seed differs')
        if name in ('INDEPENDENT12','SINGLE_M'):
            require(c['mask'] is None,'Separate architecture must not carry graph mask')
        else:
            mask=4095 if name.startswith('A') else 674 if name.startswith('B') else 0
            if name.endswith('MINUS_E'):mask &= ~(1<<unit['edge_index'])
            require(c['mask'] is not None and list(c['mask'])==[(mask>>j)&1 for j in range(12)],'Condition graph mask differs')
    terminal=all(c['status'] in ('COMPLETE','INPUT_REJECTED') for c in data['conditions'])
    require(bool(data['condition_set_terminal'])==terminal,'Terminal summary inconsistent')
    if data['status'] in ('COMPLETE','COMPLETE_WITH_INPUT_REJECTIONS'):
        require(terminal,'Completed unit lacks terminal condition')
        rejected=any(c['status']=='INPUT_REJECTED' for c in data['conditions'])
        require(rejected==(data['status']=='COMPLETE_WITH_INPUT_REJECTIONS'),'Top-level completion and input-rejection statuses conflict')
    return terminal


def cost_profile(data,unit,unit_wall_seconds):
    """Cost-only data; no answer/message/generation text returned."""
    profile={};observed_sum=0.;missing=0
    for c in data['conditions']:
        arch='independent12' if c['name']=='INDEPENDENT12' else 'single_m' if c['name']=='SINGLE_M' else 'graph'
        for t in c['request_trace']:
            g=t.get('generation');seconds=None if g is None else g.get('elapsed_seconds')
            if not finite(seconds): missing+=1;continue
            key='|'.join((arch,t['model'],t['node']))
            profile[key]=max(profile.get(key,0),seconds);observed_sum+=seconds
    require(finite(unit_wall_seconds),'Invalid measured unit wall time')
    return {'index':unit['index'],'worker':unit['worker'],'length_stratum':unit['length_stratum'],
            'status':data['status'],'terminal_conditions':sum(normalized_terminal(c)!='INCOMPLETE' for c in data['conditions']),
            'request_max_seconds':profile,'missing_request_timings':missing,
            'observed_request_seconds_sum':observed_sum,'unit_wall_seconds':unit_wall_seconds,
            'nonnegative_wall_residual_seconds':max(0.,unit_wall_seconds-observed_sum),
            'raw_wall_residual_seconds':unit_wall_seconds-observed_sum,
            'cost_fields_complete':bool(data['total_physical_cost'].get('cost_fields_complete',False)),
            'overflow_conditions':sum(c['status']=='INPUT_REJECTED' for c in data['conditions'])}


def forecast(schedule,prefix_profiles,request_keys_by_edge,current_used,budget,margin=.25):
    workers=schedule['worker_count'];prior=validate_budget(budget,workers)
    require(finite(margin) and margin>=.25,'At least 25% margin required')
    require(set(p['index'] for p in prefix_profiles)==set(schedule['prefix_indices']) and len(prefix_profiles)==8,'All 8 paid prefix units required before forecasting')
    planned={u['index']:u for u in schedule['units'] if u['phase']=='prefix'}
    require(all(p['worker']==planned[p['index']]['worker'] and p['length_stratum']==planned[p['index']]['length_stratum'] for p in prefix_profiles),'Prefix timing identity/worker/stratum differs')
    require(len(current_used)==workers and all(finite(v) for v in current_used),'Parent monotonic costs required')
    by_stratum={p['length_stratum']:p for p in prefix_profiles}
    require(set(by_stratum)==set(range(8)),'Prefix length strata incomplete')
    require(all(p['terminal_conditions']==11 and p['status']=='COMPLETE' and p['overflow_conditions']==0 and p['cost_fields_complete'] and p['missing_request_timings']==0 for p in prefix_profiles),'Prefix cannot support full-cost forecast; stop INCOMPLETE_RESOURCE')
    worker_profiles={w:[p for p in prefix_profiles if p['worker']==w] for w in range(workers)}
    require(all(worker_profiles.values()),'Each worker must have observed paid prefix work')
    remaining=[budget['release_guard_seconds']]*workers
    for u in schedule['units']:
        if u['phase']!='remainder': continue
        observed=by_stratum[u['length_stratum']]
        keys=request_keys_by_edge[str(u['edge_index'])]
        require(len(keys)==87,'Frozen successful-cache geometry must contain 87 requests')
        require(all(k in observed['request_max_seconds'] for k in keys),'Missing model/node/stratum timing; never impute zero')
        own=worker_profiles[u['worker']]
        require(all(any(k in p['request_max_seconds'] for p in own) for k in keys),'Missing worker-specific node/model timing')
        estimate=sum(max(observed['request_max_seconds'][k],max(p['request_max_seconds'].get(k,0) for p in own)) for k in keys)
        estimate+=max(observed['nonnegative_wall_residual_seconds'],max(p['nonnegative_wall_residual_seconds'] for p in own))
        remaining[u['worker']]+=estimate*(1+margin)
    total=[a+b for a,b in zip(current_used,remaining)]
    fits=(all(t<=c for t,c in zip(total,budget['worker_caps_seconds']))
          and sum(total)<=budget['global_available_seconds_for_attempt']
          and prior+sum(total)<=budget['candidate_cap_seconds'])
    return {'status':'FIT' if fits else 'INCOMPLETE_RESOURCE','prefix_units':8,'remaining_units':376,
            'minimum_margin':margin,'current_resident_seconds_by_worker':list(current_used),
            'predicted_remaining_seconds_by_worker':remaining,'predicted_attempt_seconds_by_worker':total,
            'predicted_attempt_sum_seconds':sum(total),'prior_candidate_seconds':prior,
            'predicted_candidate_total_seconds':prior+sum(total),'model_loading_and_barrier_idle_in_current_resident':True,
            'global_remaining_does_not_recharge_prior_transactions':True,
            'basis':'For each architecture/model/node use the maximum of its length-stratum observation and receiving-worker observed prefix maximum; likewise wall residual. Fixed successful-cache geometry, multiplied by at least 1.25.',
            'limits':'Eight length representatives are not a tail guarantee; future overflow can lose cache reuse. Independent hardcaps remain required.',
            'scores_or_answers_accessed':False}


class ResidenceLedger:
    def __init__(self,worker_caps,global_cap):
        self.worker_caps=list(worker_caps);self.global_cap=global_cap;self.started={};self.closed={}
    def start(self,worker,now):
        require(worker not in self.started and finite(now),'Duplicate worker start')
        self.started[worker]=now
    def close(self,worker,now):
        require(worker in self.started and worker not in self.closed and now>=self.started[worker],'Invalid residence close')
        self.closed[worker]=now
    def spent(self,now):
        return [max(0.,self.closed.get(w,now)-self.started[w]) if w in self.started else 0. for w in range(len(self.worker_caps))]
    def exceeded(self,now):
        values=self.spent(now)
        return any(v>=cap for v,cap in zip(values,self.worker_caps)) or sum(values)>=self.global_cap


def monitor_counts(receipts,resident_seconds):
    """Allowlist only; does not forward arbitrary caller fields."""
    return {'completed_units':len(receipts),'terminal_conditions':sum(r['terminal_conditions'] for r in receipts),
            'overflow_conditions':sum(r['overflow_conditions'] for r in receipts),
            'unknown_physical_cost_units':sum(not r['cost_fields_complete'] for r in receipts),
            'resident_seconds_by_worker':list(resident_seconds),'resident_sum_seconds':sum(resident_seconds)}
