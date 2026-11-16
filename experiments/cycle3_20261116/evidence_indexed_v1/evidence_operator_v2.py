"""Input-only evidence-indexed DAG. Original document bytes are never summarized."""
import copy, hashlib, json, time, uuid
from pathlib import Path
import exogenous_path_operator as base
from closure_operator import ClosureOperator, live_nodes

SCHEDULE=base.GRAPH_CONDITIONS+('FLAT_EVIDENCE','SINGLE_Q','SINGLE_M','INDEPENDENT3_Q','VOTE_Q3')
SELECT='Rank the supplied documents by usefulness for answering the question. Output only a JSON array of their exact document_id strings in decreasing relevance order. The communication budget forwards only the first two IDs. Include only useful documents; output [] if none is useful. Do not answer the question or invent IDs.'
RULE='Raw title UTF8 SHA256 sort, original index tie; first3 U, next3 V, last4 R; canonical original index order'

def partition(p):
    hs=[hashlib.sha256(t.encode()).hexdigest() for t in p['context']['title']]
    order=sorted(range(10),key=lambda i:(hs[i],i))
    return dict(U=sorted(order[:3]),V=sorted(order[3:6]),R=sorted(order[6:]),raw_title_sha256=hs,rule=RULE)

def documents(p,indices):
    return [dict(document_id=f'D{i:02d}',**base._documents(p,[i])[0]) for i in sorted(set(indices))]

def parse_selection(text,allowed):
    try:
        ids=json.loads(text)
        valid=(type(ids) is list and all(type(x) is str for x in ids)
               and len(set(ids))==len(ids) and set(ids)<=set(allowed))
    except (ValueError,TypeError):valid=False
    return (sorted(ids[:2]) if valid else []),valid

class EvidenceOperator(ClosureOperator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.protocol.update(operator_id='evidence_rank_budget_terminal_ancestor_v2',schedule=SCHEDULE,
            extension_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            treatment='Disjoint raw evidence 3/3/4 at U/V/R; sources select up to two IDs; M receives exact union of selected original documents',
            partition_rule=RULE,wire_rule='Strict JSON unique allowed ID ranking; deterministic first2 budget projection; structurally invalid becomes empty; no retry',
            select_instruction=SELECT,source_instruction=SELECT,relay_instruction=SELECT,
            graph_intermediate_instruction=SELECT,
            graph_route={'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},
            route={'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},
            seeds='Two applicability seeds; five independent confirmation seeds if separately preregistered',
            certification='Equal canonical final document sets plus identical model, node RNG and generation configuration imply equal final requests; no claim about unequal sets')
        self.protocol_hash=base.digest(self.protocol)

    def _condition(self,payload,seed,condition,shock,shards):
        if condition in ('INDEPENDENT3_Q','VOTE_Q3'):
            return super()._condition(payload,seed,condition,shock,shards)
        if condition in base.GRAPH_CONDITIONS:edges=base.condition_edges(condition,shock)
        elif condition=='FLAT_EVIDENCE':edges=(('U','M'),('V','M'),('R','M'))
        else:edges=()
        live=live_nodes(edges);outputs={}
        row=dict(key=f'{seed}/{condition}',condition=condition,execution_seed=seed,
            edges=[list(e) for e in edges],executed_nodes=list(live),
            skipped_dead_nodes=[n for n in base.NODES if n not in live],status='STARTED',nodes=outputs)
        self._conditions.append(row)
        for node in live:
            final=node=='M';model='mistral' if node=='R' or condition=='SINGLE_M' else 'qwen'
            inbound=[n for n in base.NODES if (n,node) in edges]
            ids=set(shards[node]) if not final else set()
            if condition in ('SINGLE_Q','SINGLE_M'):ids=set(range(10))
            for n in inbound:ids.update(int(x[1:]) for x in outputs[n]['evidence_ids'])
            docs=documents(payload,ids)
            messages=base._messages(payload,docs,[],base.FINAL_INSTRUCTION if final else SELECT)
            # Transmitted evidence bytes, separately from full prompt input tokens.
            received=sum(len(self.backends[model].encode(base.canonical(documents(payload,[int(x[1:]) for x in outputs[n]['evidence_ids']])))) for n in inbound)
            rec=self._request(payload,seed,condition,node,model,messages,received,final)
            rec['available_evidence_ids']=[d['document_id'] for d in docs]
            if not final:
                rec['evidence_ids'],rec['selection_valid']=parse_selection(rec['generation']['text'],rec['available_evidence_ids'])
                raw_ids=json.loads(rec['generation']['text']) if rec['selection_valid'] else []
                rec['ranking_length']=len(raw_ids);rec['budget_projection_applied']=len(raw_ids)>2
            outputs[node]=rec
        row.update(status='COMPLETE',answer=outputs['M']['generation']['text'].strip(),
            logical_service_cost_by_model=base._logical_cost(outputs.values()),
            cache_reuse_count=sum(v['reused'] for v in outputs.values()),
            terminal_evidence_ids=outputs['M']['available_evidence_ids'])

    def _snapshot(self,*args,**kwargs):
        r=base.PathOperator._snapshot(self,*args,**kwargs)
        r.update(planned_terminal_conditions=11*len(r['execution_seeds']),planned_logical_calls=36*len(r['execution_seeds']))
        return r

    def run_matrix(self,payload,seeds,removed_upstream):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent backend use forbidden')
        started=time.perf_counter()
        try:
            if self._poisoned:raise RuntimeError('Failed operator cannot silently retry')
            p=base.validate_payload(payload)
            if len(seeds) not in (2,5) or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('Two or five distinct nonnegative seeds required')
            if removed_upstream not in ('U','V'):raise ValueError('Frozen exogenous edge removal required')
            if len({base.node_seed(p['id'],s,n) for s in seeds for n in base.NODES})!=4*len(seeds):raise ValueError('RNG collision')
            shards=partition(p)
            self._matrix_id=uuid.uuid4().hex;self._cache={};self._call_ids=set();self._traces=[];self._conditions=[]
            self._native_starts={n:len(b.physical_calls) for n,b in self.backends.items()}
            try:
                for seed in seeds:
                    for condition in SCHEDULE:self._condition(p,seed,condition,removed_upstream,shards)
                r=self._snapshot(p,seeds,removed_upstream,shards,started,'COMPLETE_UNSCORED')
                assert r['complete_terminal_conditions']==11*len(seeds)
                assert sum(v['calls'] for v in r['logical_completed_node_cost_by_model'].values())==36*len(seeds)
                return r
            except Exception as error:
                self._poisoned=True
                if self._conditions and self._conditions[-1]['status']!='COMPLETE':self._conditions[-1]['status']='INPUT_REJECTED' if isinstance(error,base.InputRejected) else 'FAILED'
                raise base.PathExecutionError(self._snapshot(p,seeds,removed_upstream,shards,started,'INCOMPLETE_OR_INVALID')) from error
        finally:self._lock.release()
