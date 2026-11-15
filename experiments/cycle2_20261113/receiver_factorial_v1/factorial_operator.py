"""Receiver-model x raw-context factorial extension; upstream messages stay fixed."""
import copy,hashlib,time,uuid,re,string
from pathlib import Path
import exogenous_path_operator as base

VARIANTS=('Q_PARTIAL','M_FULL','Q_FULL')
SCHEDULE=base.SCHEDULE+tuple(g+'__'+v for v in VARIANTS for g in base.GRAPH_CONDITIONS)+('INDEPENDENT3_Q','VOTE_Q3')
assert len(SCHEDULE)==29 and len(set(SCHEDULE))==29

def normalized_vote(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())

def vote_choice(texts):
    keys=[normalized_vote(t) for t in texts]
    # All answers, including empty ones, remain in the fixed denominator.
    return max(range(len(keys)),key=lambda i:(keys.count(keys[i]),-i))

class FactorialPathOperator(base.PathOperator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.protocol=copy.deepcopy(self.protocol)
        self.protocol.update(operator_id='receiver_model_access_factorial_v1',schedule=SCHEDULE,
            extension_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            treatment='Only final receiver model and final raw-document visibility vary; upstream model, prompt and seed unchanged',
            Q_final_route={'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},
            baseline_independent_Q_has_full_context=True)
        self.protocol['vote_baseline']='Three full-context Qwen final-answer calls with M/U/V node streams; normalized plurality, tie by M then U then V; return selected complete raw text'
        self.protocol_hash=base.digest(self.protocol)

    def _condition(self,payload,seed,condition,shock,shards):
        if condition in base.SCHEDULE:return super()._condition(payload,seed,condition,shock,shards)
        if condition=='VOTE_Q3':
            outputs={};row={'key':f'{seed}/{condition}','condition':condition,'execution_seed':seed,'edges':[],
               'status':'STARTED','nodes':outputs,'answer_rule':'normalized_plurality_first_order'}
            self._conditions.append(row)
            messages=base.build_full_context_messages(payload,'SINGLE_Q')
            order=['M','U','V']
            for node in order:outputs[node]=self._request(payload,seed,condition,node,'qwen',messages,0,True)
            selected=order[vote_choice([outputs[n]['generation']['text'] for n in order])]
            row.update(status='COMPLETE',selected_node=selected,answer=outputs[selected]['generation']['text'].strip(),
               logical_service_cost_by_model=base._logical_cost(outputs.values()),cache_reuse_count=sum(v['reused'] for v in outputs.values()))
            return
        independent=condition=='INDEPENDENT3_Q'
        graph,variant=(None,'Q_FULL') if independent else condition.split('__')
        if not independent and (graph not in base.GRAPH_CONDITIONS or variant not in VARIANTS):raise ValueError('Unfrozen factorial condition')
        edges=() if independent else base.condition_edges(graph,shock)
        outputs={};row={'key':f'{seed}/{condition}','condition':condition,'execution_seed':seed,'edges':[list(e) for e in edges],
           'status':'STARTED','nodes':outputs,'receiver_model':'qwen' if variant.startswith('Q') else 'mistral',
           'receiver_raw_documents':variant.endswith('FULL')}
        self._conditions.append(row)
        for node in base.NODES:
            final=node=='M';model=('qwen' if variant.startswith('Q') else 'mistral') if final else base.MODEL_ROUTE[node]
            if independent:
                notes={n:outputs[n]['wire_message'] for n in ['U','V','R']} if final else None
                messages=base.build_full_context_messages(payload,'INDEPENDENT3_'+node,notes)
                inbound=[] if notes is None else [{'source':n,'text':notes[n]} for n in ['U','V','R']]
            else:
                parents=[n for n in base.NODES if (n,node) in edges]
                inbound=[{'source':n,'text':outputs[n]['wire_message']} for n in parents]
                docs=base._documents(payload,shards[node]) if node in ['U','V'] else base._documents(payload,range(10)) if final and variant.endswith('FULL') else []
                instruction=base.FINAL_INSTRUCTION if final else base.SOURCE_INSTRUCTION if node in ['U','V'] else base.RELAY_INSTRUCTION
                messages=base._messages(payload,docs,inbound,instruction)
            received=sum(len(self.backends[model].encode(v['text'])) for v in inbound)
            outputs[node]=self._request(payload,seed,condition,node,model,messages,received,final)
        row.update(status='COMPLETE',answer=outputs['M']['generation']['text'].strip(),
            logical_service_cost_by_model=base._logical_cost(outputs.values()),cache_reuse_count=sum(v['reused'] for v in outputs.values()))

    def _snapshot(self,*args,**kwargs):
        result=super()._snapshot(*args,**kwargs)
        result.update(planned_terminal_conditions=58,planned_logical_calls=218)
        return result

    def run_matrix(self,payload,seeds,removed_upstream):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent backend use forbidden')
        started=time.perf_counter()
        try:
            if self._poisoned:raise RuntimeError('Failed operator cannot silently retry')
            p=base.validate_payload(payload)
            if len(seeds)!=2 or len(set(seeds))!=2 or any(type(s) is not int or s<0 for s in seeds):raise ValueError('Two fixed distinct seeds required')
            if removed_upstream not in ['U','V']:raise ValueError('Frozen upstream shock required')
            if len({base.node_seed(p['id'],s,n) for s in seeds for n in base.NODES})!=8:raise ValueError('RNG stream collision')
            shards=base.partition_document_indices(p)
            self._matrix_id=uuid.uuid4().hex;self._cache={};self._call_ids=set();self._traces=[];self._conditions=[]
            self._native_starts={n:len(b.physical_calls) for n,b in self.backends.items()}
            try:
                for seed in seeds:
                    for condition in SCHEDULE:self._condition(p,seed,condition,removed_upstream,shards)
                r=self._snapshot(p,seeds,removed_upstream,shards,started,'COMPLETE_UNSCORED')
                if r['complete_terminal_conditions']!=58:raise ValueError('Incomplete factorial coverage')
                if sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values())!=218:raise ValueError('Logical factorial cost mismatch')
                return r
            except Exception as error:
                self._poisoned=True
                if self._conditions and self._conditions[-1]['status']!='COMPLETE':self._conditions[-1]['status']='INPUT_REJECTED' if isinstance(error,base.InputRejected) else 'FAILED'
                r=self._snapshot(p,seeds,removed_upstream,shards,started,'INPUT_REJECTED' if isinstance(error,base.InputRejected) else 'INCOMPLETE_OR_INVALID')
                raise base.PathExecutionError(r) from error
        finally:self._lock.release()
