"""CPU tokenizer capacity only; project row projection is explicitly no-label."""
from pathlib import Path
import os,sys,time,json,hashlib,importlib.util,threading
START=time.perf_counter(); CPU_START=time.process_time()
for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1'}.items(): os.environ[k]=v
from collections import Counter
from collections.abc import Mapping
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse
parser=argparse.ArgumentParser(description='Local tokenizer version, sampled input capacity precheck; no inference or server-equivalence claim.')
parser.add_argument('--collector-module',required=True,type=Path)
parser.add_argument('--executor-module',required=True,type=Path)
parser.add_argument('--engineering-protocol',required=True,type=Path)
parser.add_argument('--qwen-tokenizer',required=True,type=Path)
parser.add_argument('--mistral-tokenizer',required=True,type=Path)
parser.add_argument('--data-root',type=Path)
parser.add_argument('--audit-summary',type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--synthetic-only',action='store_true',help='Validate the interface on three built-in synthetic inputs; no benchmark data read')
args=parser.parse_args()
OUT=args.output.expanduser().resolve()
if OUT.exists():parser.error('Output already exists; refusing to overwrite')
if not args.synthetic_only and (args.data_root is None or args.audit_summary is None):
    parser.error('--data-root and --audit-summary are required for the historical 2000-row replay')
OUT.mkdir(parents=True,exist_ok=False)
PRIVATE=OUT/'private'; PRIVATE.mkdir()
if args.synthetic_only:
    from synthetic_probe import run_synthetic
    print(json.dumps(run_synthetic(args,HERE),indent=2))
    sys.exit(0)
DATA=args.data_root.expanduser().resolve(strict=True)
AUDIT=args.audit_summary.expanduser().resolve(strict=True)
TOKENIZER_PATHS={'qwen':args.qwen_tokenizer.expanduser().resolve(strict=True),'mistral':args.mistral_tokenizer.expanduser().resolve(strict=True)}
PROTOCOL=json.loads((HERE/'PROTOCOL.json').read_text('utf-8'))
ALLOWED=PROTOCOL['allowed_columns']
assert set(ALLOWED)=={'id','question','context','type','level'}
assert not (OUT/'RESULTS.json').exists()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
def distribution(values):
    a=np.asarray(values,dtype=np.int64)
    return {'n':int(len(a)),'min':int(a.min()),'max':int(a.max()),'mean':float(a.mean()),'quantiles':{str(q):float(np.quantile(a,q)) for q in [0,.5,.9,.95,.99,1]}}
def budget_report(values):
    a=np.asarray(values,dtype=np.int64)
    return {'tokens':distribution(a),'headroom_to_8192':distribution(8192-a),'hypothetical_extra_token_reserves':[{'extra_tokens':r,'over_8192':int((a+r>8192).sum()),'within_8192':int((a+r<=8192).sum())} for r in PROTOCOL['hypothetical_reserves']]}

MEMORY={'peak_rss_bytes':psutil.Process().memory_info().rss}
stop=threading.Event()
def monitor():
    while not stop.wait(.05):
        MEMORY['peak_rss_bytes']=max(MEMORY['peak_rss_bytes'],psutil.Process().memory_info().rss)
thread=threading.Thread(target=monitor,daemon=True);thread.start()

collector_path=args.collector_module.expanduser().resolve(strict=True)
executor_path=args.executor_module.expanduser().resolve(strict=True)
engineering_path=args.engineering_protocol.expanduser().resolve(strict=True)
collector=load_module('capacity_collector',collector_path)
engine=load_module('capacity_executor',executor_path)
engineering=json.loads(engineering_path.read_text('utf-8'))
config=engine.GraphConfig(**engineering['graph'])
assert config.rounds==3 and config.max_input_tokens==8192
audit=json.loads(AUDIT.read_text('utf-8'))

read_start=time.perf_counter();read_cpu=time.process_time()
metadata=[]; total=0
for rec in audit['source_files']:
    source=DATA/rec['path'];assert source.stat().st_size==rec['bytes']
    pf=pq.ParquetFile(source)
    assert set(ALLOWED).issubset(pf.schema_arrow.names)
    metadata.append({'relative_path':rec['path'],'rows':pf.metadata.num_rows,'row_groups':pf.metadata.num_row_groups,'bytes':source.stat().st_size,'prior_audit_sha256':rec['sha256'],'global_start':total})
    total+=pf.metadata.num_rows
assert total==PROTOCOL['population_rows']==90447
selected=np.sort(np.random.Generator(np.random.PCG64(PROTOCOL['sample']['seed'])).choice(total,size=2000,replace=False))
selection={'schema':'private_capacity_sample_not_formal_split_v1','seed':PROTOCOL['sample']['seed'],'global_row_indices':selected.tolist(),'sample_size':2000,'population_rows':total,'no_outcome_access':True,'source_order':[r['relative_path'] for r in metadata]}
with (PRIVATE/'SELECTION.json').open('x',encoding='utf-8') as f:json.dump(selection,f,indent=2);f.write('\n')

payloads=[];private_rows=[];types=Counter();levels=Counter();norm_groups=Counter()
with (PRIVATE/'SAMPLE_MANIFEST.jsonl').open('x',encoding='utf-8') as index:
    for meta in metadata:
        pf=pq.ParquetFile(DATA/meta['relative_path'])
        local_selected=selected[(selected>=meta['global_start'])&(selected<meta['global_start']+meta['rows'])]-meta['global_start']
        offset=0
        for batch in pf.iter_batches(batch_size=1024,columns=ALLOWED):
            assert set(batch.schema.names)==set(ALLOWED),'Nonallowlisted projected values'
            take=local_selected[(local_selected>=offset)&(local_selected<offset+batch.num_rows)]-offset
            if len(take):
                # Convert only the predetermined selected rows to Python values.
                selected_batch=batch.take(pa.array(take,type=pa.int64()))
                for pos,row in zip(take,selected_batch.to_pylist()):
                    assert set(row)==set(ALLOWED)
                    payload=engine.Executor.question_payload(collector.collector_payload(row))
                    qtext=payload['question']+'\n\nSupplied context:\n'+payload['context']
                    idx=int(meta['global_start']+offset+pos)
                    nh=collector.normalized_question_group(row['question']);norm_groups[nh]+=1
                    item={'global_row':idx,'source_file':meta['relative_path'],'source_row':int(offset+pos),'id_sha256':hashlib.sha256(row['id'].encode()).hexdigest(),'normalized_question_sha256':nh,'payload_sha256':hashlib.sha256(qtext.encode()).hexdigest(),'type':row['type'],'level':row['level'],'context_documents':len(row['context']['title']),'payload_characters':len(qtext)}
                    index.write(json.dumps(item,ensure_ascii=False)+'\n')
                    payloads.append((payload,qtext));private_rows.append({'global_row':idx,'payload_sha256':item['payload_sha256']})
                    types[row['type']]+=1;levels[row['level']]+=1
            offset+=batch.num_rows
        assert offset==meta['rows']
assert len(payloads)==2000 and [r['global_row'] for r in private_rows]==selected.tolist()
read_timing={'wall_seconds':time.perf_counter()-read_start,'cpu_seconds':time.process_time()-read_cpu}
print(json.dumps({'stage':'NO_LABEL_SAMPLE_READ_COMPLETE','rows':len(payloads),'read_timing':read_timing}),flush=True)

import_start=time.perf_counter();import_cpu=time.process_time()
from transformers import AutoTokenizer,LlamaTokenizerFast
import transformers,tokenizers
import_timing={'wall_seconds':time.perf_counter()-import_start,'cpu_seconds':time.process_time()-import_cpu}

class TokenizerOnlyBackend:
    identity={'backend':'CPU_TOKENIZER_ONLY_NO_GENERATION'}
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def encode(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
    def generate(self,*args,**kwargs):raise RuntimeError('Model inference forbidden in capacity probe')

def chat_ids(tok,messages):
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    if isinstance(ids,Mapping):ids=ids['input_ids']
    if ids and isinstance(ids[0],list):
        assert len(ids)==1;ids=ids[0]
    assert all(isinstance(i,int) for i in ids)
    return ids

results={'status':'COMPLETE_LOCAL_CPU_TOKENIZER_CAPACITY_PRECHECK','source_dataset_revision':PROTOCOL['dataset_revision'],'population_rows':total,'sample_rows':2000,'sample_formal_split':False,'allowed_value_columns':ALLOWED,'answer_values_read':False,'supporting_facts_values_read':False,'validation_or_test_read':False,'R1_results_read':False,'weights_loaded':False,'new_model_calls':0,'new_gpu_hours':0,'new_formal_rounds':0,'context_truncated_or_filtered':False,'source_file_verification':'Current metadata row counts and file sizes checked against prior pinned audit. Parquet byte hashes are prior audit pins; no full-file rehash was performed. Only allowlisted columns were decoded.','source_metadata':metadata,'source_pins':{'collector_sha256':sha(collector_path),'executor_sha256':sha(executor_path),'engineering_protocol_sha256':sha(engineering_path),'capacity_protocol_sha256':sha(HERE/'PROTOCOL.json'),'template_addendum_sha256':sha(HERE/'TEMPLATE_ADDENDUM.json'),'prior_audit_summary_sha256':sha(AUDIT),'script_sha256':sha(Path(__file__))},'private_sample_manifest_sha256':sha(PRIVATE/'SAMPLE_MANIFEST.jsonl'),'private_selection_sha256':sha(PRIVATE/'SELECTION.json'),'sample_types':dict(types),'sample_levels':dict(levels),'sample_normalized_question_duplicate_excess':sum(n-1 for n in norm_groups.values()),'engineering_graph_config':engineering['graph'],'formal_hotpot_scientific_template':False,'timing':{'sample_read':read_timing,'transformers_import':import_timing},'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'pyarrow':pa.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'sentencepiece_installed':False},'models':{},'limitations':['The maximum covers only this fixed 2000-row capacity sample, not all train.','No answer, supporting-fact, graph-accuracy, inference-memory or GPU-time feasibility claim.','Server tokenizer/runtime equivalence remains unestablished; counts are exact for the reported local tokenizer.','Aggregate base uses four empty wire placeholders; later message reserves are hypothetical recipient-token additions, not measured generations or certified cross-tokenizer bounds.']}

from types import SimpleNamespace
for name in ('qwen','mistral'):
    began=time.perf_counter();cpu=time.process_time()
    path=TOKENIZER_PATHS[name]
    tok=(AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=True) if name=='qwen' else LlamaTokenizerFast.from_pretrained(path,local_files_only=True,trust_remote_code=False,legacy=False))
    backend=TokenizerOnlyBackend(tok);exe=engine.Executor(backend,config)
    backend_serialized=tok.backend_tokenizer.to_str()
    comparison=None
    if name=='mistral':
        auto=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=True)
        comparison={'auto_class':type(auto).__name__,'explicit_class':type(tok).__name__,'same_backend_serialization':auto.backend_tokenizer.to_str()==backend_serialized,'same_chat_template':auto.chat_template==tok.chat_template,'payload_token_id_matches':0}
    lengths=[]
    for j,(payload,qtext) in enumerate(payloads):
        pid=backend.encode(qtext)
        r1=[]
        for w in range(4):
            messages,received,selfcount=exe._messages(payload,f'r1:w{w}',tuple([1]*12),{})
            assert received==selfcount==0
            ids=chat_ids(tok,messages)
            if j<5:
                assert ids==tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False)
            r1.append(len(ids))
        empty={f'r{config.rounds}:w{w}':SimpleNamespace(wire_message='') for w in range(4)}
        messages,received,selfcount=exe._messages(payload,'aggregate',tuple([1]*12),empty)
        assert received==selfcount==0
        aggregate=len(chat_ids(tok,messages))
        values=[len(pid),*r1,aggregate]
        lengths.append(values)
        private_rows[j][name]={'payload_tokens':values[0],'round1_worker_tokens':r1,'aggregate_empty_wire_tokens':aggregate}
        if comparison is not None:
            assert pid==auto.encode(qtext,add_special_tokens=False)
            comparison['payload_token_id_matches']+=1
        if (j+1)%500==0:
            print(json.dumps({'stage':'TOKEN_COUNT_PROGRESS','model':name,'completed_rows':j+1,'wall_seconds':time.perf_counter()-began}),flush=True)
    arr=np.asarray(lengths,dtype=np.int64)
    summary={'tokenizer_class':type(tok).__name__,'tokenizer_module':type(tok).__module__,'is_fast':tok.is_fast,'legacy_requested':False if name=='mistral' else None,'legacy_attribute':getattr(tok,'legacy',None),'server_runtime_equivalence_established':False,'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),'backend_tokenizer_sha256':hashlib.sha256(backend_serialized.encode()).hexdigest(),'local_auto_vs_explicit_equivalence':comparison,'payload_tokens':distribution(arr[:,0]),'round1_workers':{f'w{w}':budget_report(arr[:,1+w]) for w in range(4)},'round1_max_across_workers':budget_report(arr[:,1:5].max(axis=1)),'aggregate_empty_wire_placeholder':budget_report(arr[:,5]),'max_base_across_round1_and_placeholder':budget_report(arr[:,1:].max(axis=1)),'timing':{'wall_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu,'rows_per_wall_second':2000/(time.perf_counter()-began)},'tokenizer_model_max_length_metadata':int(tok.model_max_length),'input_cap_used':8192}
    results['models'][name]=summary
    del tok,backend,exe
    if name=='mistral':del auto

with (PRIVATE/'TOKEN_COUNTS.jsonl').open('x',encoding='utf-8') as f:
    for row in private_rows:f.write(json.dumps(row)+'\n')
results['private_token_counts_sha256']=sha(PRIVATE/'TOKEN_COUNTS.jsonl')
results['timing']['total']={'wall_seconds':time.perf_counter()-START,'cpu_seconds':time.process_time()-CPU_START}
stop.set();thread.join()
MEMORY['final_rss_bytes']=psutil.Process().memory_info().rss
results['process_memory']=MEMORY
results['cuda_initialized']=bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
assert not results['cuda_initialized']
with (OUT/'RESULTS.json').open('x',encoding='utf-8') as f:json.dump(results,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'status':results['status'],'models':{k:{'payload':v['payload_tokens'],'round1_max':v['round1_max_across_workers']['tokens'],'aggregate_placeholder':v['aggregate_empty_wire_placeholder']['tokens'],'timing':v['timing']} for k,v in results['models'].items()},'total_timing':results['timing']['total'],'process_memory':MEMORY,'cuda_initialized':False},indent=2),flush=True)
