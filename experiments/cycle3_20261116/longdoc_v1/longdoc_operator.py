"""Long-document exact-span DAG. Input-only; no gold or quality selection."""
import copy, hashlib, json, re, time, uuid
from collections import Counter
from pathlib import Path
import exogenous_path_operator as base

SCHEDULE=base.GRAPH_CONDITIONS+('FLAT','SINGLE_Q','SINGLE_M','BM25','DENSE','VOTE_Q3')
SELECT='Select up to six passage IDs most useful for answering the question. Rank them by usefulness. Return only a JSON array of distinct supplied IDs, for example ["P0001","P0005"]. Do not answer the question or invent IDs.'
FINAL='Answer the question using the supplied paper text. Output only a concise final answer. For a yes/no question output Yes or No. If the paper does not answer the question, output Unanswerable. Do not add an explanation, label, JSON, Markdown, or quotation marks.'

def validate(payload):
    assert set(payload)=={'id','question','segments','retrieval'}
    assert all(isinstance(payload[k],str) and payload[k].strip() for k in ('id','question'))
    seg=payload['segments'];assert isinstance(seg,list) and len(seg)>=3
    assert all(set(x)=={'id','text'} and x['id']==f'P{i:04d}' and isinstance(x['text'],str) and x['text'] for i,x in enumerate(seg))
    ids={x['id'] for x in seg};assert set(payload['retrieval'])=={'BM25','DENSE'}
    for v in payload['retrieval'].values():assert isinstance(v,list) and len(v)==min(18,len(seg)) and len(set(v))==len(v) and set(v)<=ids
    return copy.deepcopy(payload)

def select(text,allowed):
    try:
        v=json.loads(text.strip())
        assert type(v) is list and len(v)<=6 and all(type(x) is str for x in v) and len(v)==len(set(v)) and set(v)<=set(allowed)
        return v,True
    except (ValueError,TypeError,AssertionError):return [],False

def messages(payload,ids,final):
    chosen=set(ids)
    return [{'role':'system','content':FINAL if final else SELECT},{'role':'user','content':base.canonical({'question':payload['question'],'passages':[x for x in payload['segments'] if x['id'] in chosen]})}]

def live_nodes(edges):
    live={'M'}
    while True:
        nxt=live|{u for u,v in edges if v in live}
        if nxt==live:return tuple(n for n in base.NODES if n in live)
        live=nxt

def vote(texts):
    # Gold-independent normalized exact plurality; stable first-node tie.
    keys=[' '.join(re.sub(r'\b(a|an|the)\b',' ',re.sub(r'[^\w\s]','',t.lower())).split()) for t in texts]
    counts=Counter(keys);return texts[max(range(len(texts)),key=lambda i:(counts[keys[i]],-i))].strip()

class LongdocOperator(base.PathOperator):
    def __init__(self,*a,**kw):
        kw['config']=base.PathConfig(message_tokens=128,final_tokens=256,max_input_tokens=32000)
        super().__init__(*a,**kw)
        self.protocol={'operator_id':'longdoc_exact_span_v1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'base_source_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
          'config':vars(self.config),'model_identities':self.identities,'schedule':SCHEDULE,
          'G0_edges':base.G0_EDGES,'route':{'U':'qwen','V':'qwen','R':'mistral','M':'qwen'},
          'selector':SELECT,'final':FINAL,'selection':'strict JSON distinct <=6 available IDs, invalid empty, no retry',
          'partition':'original contiguous exact substrings <=256 Qwen tokens; round-robin segment index mod3',
          'terminal':'original text union in original order; full controls all segments; BM25/DENSE top18',
          'seed_rule':'qualified base.node_seed; common node RNG across conditions',
          'cache':'exact model request+seed+config+protocol, per matrix; nominal service charges all nodes',
          'vote':'three full-context Qwen calls seeds U/V/R, normalized exact plurality, first-node ties',
          'labels_used':False}
        self.protocol_hash=base.digest(self.protocol)

    def _condition(self,p,seed,c,shock,shards):
        graph=c in base.GRAPH_CONDITIONS
        edges=base.condition_edges(c,shock) if graph else ()
        order=live_nodes(edges) if graph else ('U','V','R','M') if c=='FLAT' else ('U','V','R') if c=='VOTE_Q3' else ('M',)
        row={'key':f'{seed}/{c}','condition':c,'execution_seed':seed,'edges':[list(v) for v in edges],
          'executed_nodes':list(order),'status':'STARTED','nodes':{}}
        self._conditions.append(row);outputs=row['nodes'];allids=[x['id'] for x in p['segments']]
        for node in order:
            final=node=='M' or c=='VOTE_Q3'
            model='mistral' if c=='SINGLE_M' or (node=='R' and not final) else 'qwen'
            if c in ('SINGLE_Q','SINGLE_M','VOTE_Q3'):ids=allids
            elif c in ('BM25','DENSE'):ids=p['retrieval'][c]
            else:
                ids=set() if final else set(shards[node])
                parents=[n for n in base.NODES if (n,node) in edges] if graph else list('UVR') if final else []
                for n in parents:ids.update(outputs[n]['selected_ids'])
                ids=[x for x in allids if x in ids]
            record=self._request(p,seed,c,node,model,messages(p,ids,final),0,final)
            record.update(available_ids=list(ids))
            if not final:
                selected,valid=select(record['generation']['text'],ids)
                record.update(selected_ids=selected,selection_valid=valid)
            outputs[node]=record
        row.update(status='COMPLETE',answer=vote([outputs[n]['generation']['text'] for n in order]) if c=='VOTE_Q3' else outputs['M']['generation']['text'].strip(),
          logical_service_cost_by_model=base._logical_cost(outputs.values()),cache_reuse_count=sum(v['reused'] for v in outputs.values()))

    def run_matrix(self,payload,seeds,removed_upstream):
        if not self._lock.acquire(blocking=False):raise RuntimeError('Concurrent operator use')
        started=time.perf_counter()
        try:
            if self._poisoned:raise RuntimeError('Poisoned operator')
            p=validate(payload);assert len(seeds)==2 and len(set(seeds))==2 and removed_upstream in ('U','V')
            assert len({base.node_seed(p['id'],s,n) for s in seeds for n in base.NODES})==8
            shards={n:[x['id'] for i,x in enumerate(p['segments']) if i%3==j] for j,n in enumerate('UVR')}
            self._matrix_id=uuid.uuid4().hex;self._cache={};self._call_ids=set();self._traces=[];self._conditions=[]
            self._native_starts={n:len(b.physical_calls) for n,b in self.backends.items()}
            try:
                for seed in seeds:
                    for c in SCHEDULE:self._condition(p,seed,c,removed_upstream,shards)
                result=self._snapshot(p,seeds,removed_upstream,shards,started,'COMPLETE_UNSCORED')
                result.update(planned_terminal_conditions=24,planned_logical_calls=68)
                assert result['complete_terminal_conditions']==24
                assert sum(v['calls'] for v in result['logical_completed_node_cost_by_model'].values())==68
                return result
            except Exception:
                self._poisoned=True
                raise base.PathExecutionError(self._snapshot(p,seeds,removed_upstream,shards,started,'INCOMPLETE_OR_INVALID'))
        finally:self._lock.release()
