"""Full-context DAG with exact terminal-ancestor execution; no labels or policy."""
import copy,hashlib,time,uuid
from pathlib import Path
import exogenous_path_operator as base
from factorial_operator import FactorialPathOperator

SCHEDULE=base.GRAPH_CONDITIONS+('INDEPENDENT3_Q','SINGLE_Q','VOTE_Q3')

def live_nodes(edges):
    live={'M'}
    while True:
        new=live|{u for u,v in edges if v in live}
        if new==live:return tuple(n for n in base.NODES if n in live)
        live=new

class ClosureOperator(FactorialPathOperator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.protocol.update(operator_id='full_context_terminal_ancestor_v1',schedule=SCHEDULE,
            extension_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            treatment='All executed nodes receive all ten raw documents; only nominal incoming graph edges vary',
            graph_route={'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},
            graph_intermediate_instruction=base.SOURCE_INSTRUCTION,
            execution='Execute precisely terminal ancestors in fixed topological order; no unconnected node calls',
            graph_compiler_is_shared_by_all_policies=True,
            seeds='Two artificial engineering seeds or five formal generation seeds; independently seeded per node')
        self.protocol_hash=base.digest(self.protocol)

    def _condition(self,payload,seed,condition,shock,shards):
        if condition not in base.GRAPH_CONDITIONS:
            return super()._condition(payload,seed,condition,shock,shards)
        edges=base.condition_edges(condition,shock);live=live_nodes(edges)
        outputs={};row={'key':f'{seed}/{condition}','condition':condition,'execution_seed':seed,
            'edges':[list(e) for e in edges],'executed_nodes':list(live),
            'skipped_dead_nodes':[n for n in base.NODES if n not in live],
            'status':'STARTED','nodes':outputs}
        self._conditions.append(row)
        for node in live:
            final=node=='M';model='qwen' if final else base.MODEL_ROUTE[node]
            inbound=[{'source':n,'text':outputs[n]['wire_message']} for n in base.NODES if (n,node) in edges]
            messages=base._messages(payload,base._documents(payload,range(10)),inbound,
                base.FINAL_INSTRUCTION if final else base.SOURCE_INSTRUCTION)
            received=sum(len(self.backends[model].encode(v['text'])) for v in inbound)
            outputs[node]=self._request(payload,seed,condition,node,model,messages,received,final)
        row.update(status='COMPLETE',answer=outputs['M']['generation']['text'].strip(),
            logical_service_cost_by_model=base._logical_cost(outputs.values()),
            cache_reuse_count=sum(v['reused'] for v in outputs.values()))

    def _snapshot(self,*args,**kwargs):
        r=base.PathOperator._snapshot(self,*args,**kwargs)
        r.update(planned_terminal_conditions=9*len(r['execution_seeds']),planned_logical_calls=31*len(r['execution_seeds']))
        return r

    def run_matrix(self,payload,seeds,removed_upstream):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent backend use forbidden')
        started=time.perf_counter()
        try:
            if self._poisoned:raise RuntimeError('Failed operator cannot silently retry')
            p=base.validate_payload(payload)
            if len(seeds) not in (2,5) or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('Two engineering or five formal seeds required')
            if removed_upstream not in ['U','V']:raise ValueError('Frozen exogenous edge removal required')
            if len({base.node_seed(p['id'],s,n) for s in seeds for n in base.NODES})!=4*len(seeds):raise ValueError('RNG collision')
            shards=base.partition_document_indices(p)
            self._matrix_id=uuid.uuid4().hex;self._cache={};self._call_ids=set();self._traces=[];self._conditions=[]
            self._native_starts={n:len(b.physical_calls) for n,b in self.backends.items()}
            try:
                for seed in seeds:
                    for condition in SCHEDULE:self._condition(p,seed,condition,removed_upstream,shards)
                r=self._snapshot(p,seeds,removed_upstream,shards,started,'COMPLETE_UNSCORED')
                assert r['complete_terminal_conditions']==9*len(seeds)
                assert sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values())==31*len(seeds)
                return r
            except Exception as error:
                self._poisoned=True
                if self._conditions and self._conditions[-1]['status']!='COMPLETE':self._conditions[-1]['status']='INPUT_REJECTED' if isinstance(error,base.InputRejected) else 'FAILED'
                raise base.PathExecutionError(self._snapshot(p,seeds,removed_upstream,shards,started,'INCOMPLETE_OR_INVALID')) from error
        finally:self._lock.release()
