"""Single-question applicability matrix, CPU-prepared operator only.

No data loading, sampling, grading, statistics, GPU admission or model loading.
The caller exclusively lends two already constructed model backends for a run.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
import threading
import time
import uuid
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Mapping

import graph_credit.executor as frozen_graph
import routed_executor as frozen_route
import independent12_executor as frozen_independent
from graph_credit.executor import Executor,GraphConfig,MESSAGE_INSTRUCTION,canonical,derive_seed,digest
from routed_executor import RoutedBackend,RoutedExecutor
from independent12_executor import Independent12Executor,Independent12Result,SampleSlot,PHYSICAL_EVENT_FIELDS
from matrix_geometry import A,B,NO_EDGE,B_EDGES,ROUTE,NODES,geometry,without

MATRIX_ID='single_question_hotpot_applicability_matrix_v1_cpu_staging'
SINGLE_ID='single_mistral_full_context_direct_v1_cpu_staging'
DEPENDENCY_PINS={
    'executor':'85acab26b3b713900e8099a5d77197af698525f57c2d9d07d1193c68b2c383b1',
    'router':'0273aebcfba546fd7112c82937b8f27447d0eee87243fbd9c00137d107938625',
    'independent12':'d3ec41ca2df218592ef8b4dc06c7e26441ee2f0f20daad5b70c6df1ae10b9dbc',
}
SINGLE_SYSTEM_PROMPT=(
    'You are a careful question-answering assistant. Answer the question directly using the supplied context. '
    'Independently verify the relevant evidence, resolve ambiguous references, and check that the answer '
    'follows from that context. Do detailed reasoning internally. '+MESSAGE_INSTRUCTION
)
SEED1_ORDER=('A','B','NO_EDGE','A_MINUS_E','B_MINUS_E','INDEPENDENT12','SINGLE_M')
SEED2_ORDER=('A','B','A_MINUS_E','B_MINUS_E')
COST_KEYS=('calls','input_tokens','output_tokens','received_message_tokens','self_state_tokens',
           'model_elapsed_seconds','gpu_seconds','peak_allocated_bytes')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_dependencies():
    actual={'executor':_sha(frozen_graph.__file__),'router':_sha(frozen_route.__file__),
            'independent12':_sha(frozen_independent.__file__)}
    if actual!=DEPENDENCY_PINS:
        raise RuntimeError('Frozen graph/router/independent12-v2 dependency mismatch')
    return actual


def _full_question_text(payload):
    text=payload['question']
    if payload.get('context'):
        text+='\n\nSupplied context:\n'+payload['context']
    return text


def _sum_costs(costs):
    result={}
    for key in COST_KEYS:
        values=[cost.get(key) for cost in costs]
        if any(v is None for v in values):
            result[key]=None
        else:
            result[key]=max(values,default=0) if key=='peak_allocated_bytes' else sum(values)
    result['resident_GPU_hours']=None
    return result


def _node_cost(records):
    records=list(records)
    return {'calls':len(records),'input_tokens':sum(r.generation.input_tokens for r in records),
            'output_tokens':sum(r.generation.output_tokens for r in records),
            'received_message_tokens':sum(r.received_message_tokens for r in records),
            'self_state_tokens':sum(r.self_state_tokens for r in records),
            'model_elapsed_seconds':sum(r.generation.elapsed_seconds for r in records),
            'gpu_seconds':sum(r.generation.gpu_seconds for r in records),
            'peak_allocated_bytes':max((r.generation.gpu_peak_allocated_bytes for r in records),default=0)}


def classify_failure(error):
    """Only the two exact frozen backend preflight messages permit continuation."""
    seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current,ValueError):
            text=str(current)
            if re.fullmatch(r'Prompt has \d+ tokens; frozen cap is \d+',text):
                return 'INPUT_TOKEN_LIMIT'
            if text=='Prompt plus output reservation exceeds model position limit':
                return 'MODEL_POSITION_LIMIT'
        current=current.__cause__
    return 'INFRASTRUCTURE_OR_IMPLEMENTATION_FAILURE'


class ObservingModelBackend:
    """Transparent observer: forwards identity, tokenizer and generation unchanged."""
    def __init__(self,model,underlying,node_getter,traces):
        self.model=model;self.underlying=underlying;self.node_getter=node_getter;self.traces=traces

    @property
    def identity(self):return self.underlying.identity

    @property
    def physical_calls(self):return getattr(self.underlying,'physical_calls',[])

    def encode(self,text):return self.underlying.encode(text)

    def decode(self,tokens):return self.underlying.decode(tokens)

    def generate(self,messages,seed,max_new_tokens,config):
        node=self.node_getter()
        if node is None:raise RuntimeError('Observation requires the original active route node')
        slot=SampleSlot(len(self.traces),node,None if node=='aggregate' else int(node[-1]),
                        None if node=='aggregate' else int(node[1]),self.model,seed,
                        attempted_backend_request=True,messages=copy.deepcopy(messages))
        self.traces.append(slot)
        physical=self.physical_calls;start=len(physical)
        try:
            generation=self.underlying.generate(messages,seed,max_new_tokens,config)
            slot.generation=generation;slot.status='GENERATION_RETURNED'
            return generation
        except Exception as error:
            slot.status='BACKEND_ERROR';slot.error_type=type(error).__name__
            raise
        finally:
            slot.physical_events=[copy.deepcopy({k:v for k,v in row.items() if k in PHYSICAL_EVENT_FIELDS})
                                  for row in physical[start:]]


class ObservedGraphExecutor(RoutedExecutor):
    """Observe existing local outputs; do not alter any graph input or reuse decision."""
    def execute(self,question,mask,seed,base=None):
        self.partial_nodes={};self.node_accounting={};self.active_node=None
        result=super().execute(question,mask,seed,base)
        self.partial_nodes=dict(result.nodes)
        return result

    def _messages(self,question,node,graph,outputs):
        self.partial_nodes=dict(outputs);self.active_node=node
        messages,received,own=super()._messages(question,node,graph,outputs)
        self.node_accounting[node]=(received,own)
        return messages,received,own


@dataclass
class SingleMResult(Independent12Result):
    def cost(self):
        # Reuse the audited v2 same-call-id reconciliation. Only plan-size metadata differs.
        costs=super().cost();costs['planned_calls']=1;costs['unattempted_calls']=1-costs['calls']
        return costs

    def to_dict(self):
        costs=self.cost()
        return {**dataclasses.asdict(self),'control_id':SINGLE_ID,'physical_cost':costs,
                'logical_service_cost':copy.deepcopy(costs),'cache_reuse_count':0,
                'format_failures':sum(s.record is not None and not s.record.message_metadata['format_ok'] for s in self.slots),
                'scientific_admission':False}


class SingleMExecutionError(RuntimeError):
    def __init__(self,partial_result):
        super().__init__('Single-M direct execution failed; inspect partial_result')
        self.partial_result=partial_result


class SingleMExecutor(Independent12Executor):
    """New direct-answer operator, sharing only pinned generation/normalization plumbing."""
    def __init__(self,backend,config=None):
        config=config or GraphConfig(rounds=3)
        if config!=GraphConfig(rounds=3):raise ValueError('Single-M uses the exact qualified config with 256 output tokens')
        super().__init__(backend,config)
        self.protocol_identity={'control_id':SINGLE_ID,'config':dataclasses.asdict(config),
            'backend':copy.deepcopy(backend.identity),'dependency_pins':_check_dependencies(),
            'operator_code_sha256':_sha(__file__),'system_prompt':SINGLE_SYSTEM_PROMPT,
            'user_prompt':'complete question plus explicit Supplied context only',
            'model':'mistral','output_cap':256,'node_alias':'aggregate',
            'seed_policy':'unchanged derive_seed(question_id,execution_seed,aggregate)',
            'candidates_visible':False,'cache_policy':'disabled',
            'failure_cost_policy':'pinned independent12-v2 reconciliation with planned_calls=1'}
        self.protocol_hash=digest(self.protocol_identity)

    def _aggregate_messages(self,payload,slots):
        return [{'role':'system','content':SINGLE_SYSTEM_PROMPT},
                {'role':'user','content':_full_question_text(payload)}],0

    def execute(self,question,seed):
        payload=self.question_payload(question)
        if digest(self.protocol_identity)!=self.protocol_hash or self.protocol_identity['backend']!=self.backend.identity:
            raise RuntimeError('Frozen single-M identity changed')
        if tuple(self.backend.worker_models)!=ROUTE or self.backend.aggregator_model!='mistral':
            raise RuntimeError('Frozen route changed')
        if dataclasses.asdict(self.config)!=self.protocol_identity['config']:
            raise RuntimeError('Frozen configuration changed')
        slot=SampleSlot(0,'aggregate',None,None,'mistral',derive_seed(payload['id'],seed,'aggregate'))
        result=SingleMResult(uuid.uuid4().hex,payload['id'],digest(payload),seed,self.protocol_hash,[slot])
        if not self.backend._execution_lock.acquire(blocking=False):raise RuntimeError('Concurrent routed execution forbidden')
        started=time.perf_counter()
        try:
            try:self._execute_slot(payload,slot,result)
            except Exception as error:
                slot.status=('POSTPROCESS_ERROR' if slot.generation is not None else
                             'BACKEND_ERROR' if slot.attempted_backend_request else 'PREPARATION_ERROR')
                slot.error_type=type(error).__name__;result.status='FAILED'
                raise SingleMExecutionError(result) from error
            result.status='COMPLETE';result.answer=slot.record.parsed['answer']
            return result
        finally:
            result.elapsed_seconds=time.perf_counter()-started
            self.backend.clear_node();self.backend._execution_lock.release()


def _request_cost(traces,planned_calls,complete):
    holder=Independent12Result('request-ledger','', '',0,'',traces,status='COMPLETE' if complete else 'FAILED')
    cost=holder.cost()
    cost['planned_calls']=planned_calls;cost['unattempted_calls']=max(0,planned_calls-cost['calls'])
    cost['self_state_tokens']=sum(s.self_state_tokens for s in traces)
    cost['returned_generation_cost']['self_state_tokens']=sum(s.self_state_tokens for s in traces if s.generation is not None)
    return cost


@dataclass
class ConditionResult:
    key:str
    name:str
    seed_number:int
    seed:int
    mask:tuple[int,...]|None
    nominal_physical_calls:int
    planned_physical_calls:int
    status:str='NOT_RUN'
    replay_mode:str|None=None
    cache_source_condition:str|None=None
    execution:dict[str,Any]|None=None
    partial_execution:dict[str,Any]|None=None
    physical_cost:dict[str,Any]|None=None
    logical_service_cost:dict[str,Any]|None=None
    known_partial_logical_cost:dict[str,Any]|None=None
    request_trace:list[dict[str,Any]]=field(default_factory=list)
    route_events:list[dict[str,Any]]=field(default_factory=list)
    error_type:str|None=None
    failure_category:str|None=None


@dataclass
class MatrixResult:
    execution_id:str
    question_id:str
    question_fingerprint:str
    seed1:int
    seed2:int
    probe_edge:int
    matrix_protocol_hash:str
    conditions:list[ConditionResult]
    status:str='RUNNING'
    elapsed_seconds:float=0.0
    aggregate_physical_ledger:dict[str,Any]|None=None

    def to_dict(self):
        attempted=[c for c in self.conditions if c.physical_cost is not None]
        costs=[c.physical_cost for c in attempted]
        terminal=all(c.status in ('COMPLETE','INPUT_REJECTED') for c in self.conditions)
        all_complete=all(c.status=='COMPLETE' for c in self.conditions)
        known=[]
        for cost in costs:
            row={**cost,**cost['known_cost_components']}
            known.append(row)
        data=dataclasses.asdict(self);data.pop('aggregate_physical_ledger')
        aggregate=self.aggregate_physical_ledger
        known_total=_sum_costs(known)
        if aggregate is not None:
            known_total={key:aggregate['known_cost_components'].get(key,aggregate.get(key)) for key in COST_KEYS}
            known_total['resident_GPU_hours']=None
        return {**data,'operator_id':MATRIX_ID,
            'nominal_success_physical_calls':87,'nominal_success_logical_calls':131,
            'physical_call_count_scope':'attempted_backend_requests_including_rejections; 87 equals physical generations only on complete success',
            'total_physical_cost':copy.deepcopy(aggregate) if aggregate is not None else _sum_costs(costs),
            'known_physical_cost_components':known_total,
            'total_logical_service_cost':_sum_costs([c.logical_service_cost for c in self.conditions]) if all_complete else None,
            'known_completed_logical_service_cost':_sum_costs([c.logical_service_cost for c in self.conditions if c.status=='COMPLETE']),
            'request_count_by_model':{m:sum(row['model']==m for c in attempted for row in c.request_trace) for m in ('qwen','mistral')},
            'returned_generations':sum(c.physical_cost['returned_generations'] for c in attempted),
            'condition_set_terminal':terminal,'all_conditions_completed':all_complete,
            'scientific_admission':False,'scores_computed':False,'resident_GPU_hours':None}


class MatrixExecutionError(RuntimeError):
    def __init__(self,key,partial_result):
        super().__init__(f'Matrix incomplete at {key}; inspect partial_result')
        self.partial_result=partial_result


class HotpotApplicabilityMatrixExecutor:
    """Owns one frozen route over two caller-supplied, exclusively used backends."""
    def __init__(self,backends:Mapping[str,Any],config:GraphConfig|None=None,route_log=None):
        if set(backends)!={'qwen','mistral'}:raise ValueError('Exactly qwen and mistral backends are required')
        self.config=config or GraphConfig(rounds=3)
        if self.config!=GraphConfig(rounds=3):raise ValueError('This matrix requires the unchanged qualified three-round configuration')
        self._traces=[];self._matrix_lock=threading.Lock()
        proxies={name:ObservingModelBackend(name,b,lambda:self.router._selected.get(),self._traces) for name,b in backends.items()}
        self.router=RoutedBackend(proxies,ROUTE,'mistral',route_log)
        self.graph=ObservedGraphExecutor(self.router,self.config)
        self.independent=Independent12Executor(self.router,self.config)
        self.single=SingleMExecutor(self.router,self.config)
        self.protocol_identity={'operator_id':MATRIX_ID,'operator_code_sha256':_sha(__file__),
            'geometry_code_sha256':_sha(Path(__file__).with_name('matrix_geometry.py')),
            'dependency_pins':_check_dependencies(),'config':dataclasses.asdict(self.config),
            'backend':copy.deepcopy(self.router.identity),'masks':{'A':list(A),'B':list(B),'NO_EDGE':list(NO_EDGE)},
            'edge_indexing':'zero_based_source_major','allowed_probe_edges':sorted(B_EDGES),
            'seed1_order':list(SEED1_ORDER),'seed2_order':list(SEED2_ORDER),
            'graph_protocol_hash':self.graph.protocol_hash,'independent12_protocol_hash':self.independent.protocol_hash,
            'single_protocol_hash':self.single.protocol_hash,'original_base_policy':{'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'},
            'missing_base_policy':'fresh_same_control_same_seed_no_partial_cache',
            'overflow_policy':'retain_condition_terminal_and_continue_without_truncation_or_scores',
            'other_failure_policy':'stop_INCOMPLETE_no_retry','observer_policy':'identity_prompt_RNG_return_values_unchanged'}
        self.protocol_hash=digest(self.protocol_identity)

    def _plan(self,seed1,seed2,edge):
        geom=geometry(edge);masks={'A':A,'B':B,'NO_EDGE':NO_EDGE,'A_MINUS_E':without(A,edge),'B_MINUS_E':without(B,edge)}
        return [ConditionResult(f'S{ordinal}/{name}',name,ordinal,seed,masks.get(name),
                    geom['physical_calls_by_condition'][name],geom['physical_calls_by_condition'][name])
                for ordinal,seed,order in ((1,seed1,SEED1_ORDER),(2,seed2,SEED2_ORDER)) for name in order]

    def _finish_condition(self,condition,traces,route_events,records,accounting,complete):
        for slot in traces:
            if slot.node in accounting:
                slot.received_message_tokens,slot.self_state_tokens=accounting[slot.node]
            if slot.node in records:
                slot.record=records[slot.node];slot.input_fingerprint=slot.record.input_fingerprint
                slot.status='SUCCESS' if slot.record.message_metadata['format_ok'] else 'FORMAT_ERROR'
            slot.route_events=[copy.deepcopy(row) for row in route_events if row['node']==slot.node]
        condition.route_events=copy.deepcopy(route_events)
        condition.request_trace=[dataclasses.asdict(s) for s in traces]
        condition.physical_cost=_request_cost(traces,condition.planned_physical_calls,complete)
        if not complete:condition.known_partial_logical_cost=_node_cost(records.values())

    def execute(self,question:Mapping[str,Any],seed1:int,seed2:int,probe_edge:int)->MatrixResult:
        payload=Executor.question_payload(question)
        if type(seed1) is not int or type(seed2) is not int or seed1==seed2:
            raise ValueError('Two distinct explicit integer seeds are required')
        geometry(probe_edge)
        if digest(self.protocol_identity)!=self.protocol_hash or self.protocol_identity['backend']!=self.router.identity:
            raise RuntimeError('Frozen matrix identity was modified')
        if dataclasses.asdict(self.config)!=self.protocol_identity['config'] or any(
                operator.config!=self.config for operator in (self.graph,self.independent,self.single)):
            raise RuntimeError('Frozen matrix configuration was modified')
        if self.graph.protocol_hash!=self.protocol_identity['graph_protocol_hash'] or self.independent.protocol_hash!=self.protocol_identity['independent12_protocol_hash'] or self.single.protocol_hash!=self.protocol_identity['single_protocol_hash']:
            raise RuntimeError('A control protocol identity was modified')
        if tuple(self.router.worker_models)!=ROUTE or self.router.aggregator_model!='mistral':
            raise RuntimeError('Frozen model route was modified')
        if not self._matrix_lock.acquire(blocking=False):raise RuntimeError('Concurrent matrix use is forbidden')
        started=time.perf_counter()
        self._traces.clear()
        result=MatrixResult(uuid.uuid4().hex,payload['id'],digest(payload),seed1,seed2,probe_edge,self.protocol_hash,
                            self._plan(seed1,seed2,probe_edge))
        bases={}
        try:
            for condition in result.conditions:
                trace_start=len(self._traces);route_start=len(self.router.route_events)
                records={};accounting={};returned=None;failure=None
                try:
                    if condition.mask is not None:
                        base_name={'B':'A','NO_EDGE':'A','A_MINUS_E':'A','B_MINUS_E':'B'}.get(condition.name)
                        base_key=f'S{condition.seed_number}/{base_name}' if base_name else None
                        base=bases.get(base_key) if base_key else None
                        condition.cache_source_condition=base_key if base is not None else None
                        condition.replay_mode='ORIGINAL_BASE' if base is not None else ('FRESH_BASE_UNAVAILABLE' if base_name else 'FRESH')
                        if base is None:condition.planned_physical_calls=13
                        returned=self.graph.execute(payload,condition.mask,condition.seed,base=base)
                        records=returned.nodes;accounting=dict(self.graph.node_accounting)
                    elif condition.name=='INDEPENDENT12':
                        condition.replay_mode='FRESH_SEPARATE_PROTOCOL'
                        returned=self.independent.execute(payload,condition.seed)
                        records=returned.nodes;accounting={s.node:(s.received_message_tokens or 0,s.self_state_tokens) for s in returned.slots}
                    else:
                        condition.replay_mode='FRESH_DIRECT_NO_CANDIDATES'
                        returned=self.single.execute(payload,condition.seed)
                        records=returned.nodes;accounting={s.node:(s.received_message_tokens or 0,s.self_state_tokens) for s in returned.slots}
                    condition.execution=returned.to_dict();condition.logical_service_cost=condition.execution['logical_service_cost']
                    condition.status='COMPLETE'
                except Exception as error:
                    failure=error;condition.error_type=type(error).__name__;condition.failure_category=classify_failure(error)
                    condition.status='INPUT_REJECTED' if condition.failure_category in ('INPUT_TOKEN_LIMIT','MODEL_POSITION_LIMIT') else 'INCOMPLETE'
                    partial=getattr(error,'partial_result',None)
                    if partial is not None:
                        condition.partial_execution=partial.to_dict();records=partial.nodes
                        accounting={s.node:(s.received_message_tokens or 0,s.self_state_tokens) for s in partial.slots}
                    elif condition.mask is not None:
                        records=dict(self.graph.partial_nodes);accounting=dict(self.graph.node_accounting)
                        condition.partial_execution={'complete':False,'protocol_hash':self.graph.protocol_hash,
                            'question_fingerprint':digest(payload),'seed':condition.seed,'mask':condition.mask,
                            'active_node':self.graph.active_node,'nodes':{n:dataclasses.asdict(r) for n,r in records.items()},'answer':None}
                traces=self._traces[trace_start:];events=self.router.route_events[route_start:]
                self._finish_condition(condition,traces,events,records,accounting,failure is None)
                result.aggregate_physical_ledger=_request_cost(self._traces,sum(c.planned_physical_calls for c in result.conditions),False)
                if failure is None:
                    try:
                        if len(traces)!=condition.planned_physical_calls:
                            raise RuntimeError('Actual physical requests differ from the declared original-base geometry')
                        if not condition.physical_cost['cost_fields_complete']:
                            raise RuntimeError('Completed condition has unresolved physical cost evidence')
                        if any('call_id_reused_across_backend_requests' in group['identity_issues']
                               for request in result.aggregate_physical_ledger['request_cost_evidence'] for group in request['groups']):
                            raise RuntimeError('Call identity was reused across matrix requests; physical cost remains unresolved')
                        original_cost=condition.execution['physical_cost']
                        for key in COST_KEYS:
                            if condition.physical_cost[key]!=original_cost[key]:
                                raise RuntimeError('Observed physical ledger disagrees with the unmodified control cost: '+key)
                        if condition.name in ('A','B'):
                            bases[condition.key]=returned
                    except Exception as error:
                        condition.status='INCOMPLETE';condition.error_type=type(error).__name__
                        condition.failure_category='INTEGRITY_FAILURE';failure=error
                if failure is not None and condition.status!='INPUT_REJECTED':
                    result.status='INCOMPLETE'
                    raise MatrixExecutionError(condition.key,result) from failure
            result.status='COMPLETE_WITH_INPUT_REJECTIONS' if any(c.status=='INPUT_REJECTED' for c in result.conditions) else 'COMPLETE'
            result.aggregate_physical_ledger=_request_cost(self._traces,sum(c.planned_physical_calls for c in result.conditions),result.status=='COMPLETE')
            if result.status=='COMPLETE' and sum(c.physical_cost['calls'] for c in result.conditions)!=87:
                result.status='INCOMPLETE';raise MatrixExecutionError('final-geometry',result)
            return result
        finally:
            result.elapsed_seconds=time.perf_counter()-started
            self._matrix_lock.release()
