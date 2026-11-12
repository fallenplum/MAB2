"""Synthetic-only interface check. No benchmark source or labels are opened."""
from pathlib import Path
from collections.abc import Mapping
from types import SimpleNamespace
import hashlib,json,importlib.util,sys,time

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
def ids(tok,messages):
    x=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    if isinstance(x,Mapping):x=x['input_ids']
    if x and isinstance(x[0],list):assert len(x)==1;x=x[0]
    rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert x==tok.encode(rendered,add_special_tokens=False)
    return x
def run_synthetic(args,here):
    started=time.perf_counter();cpu=time.process_time();sys.dont_write_bytecode=True
    historical=json.loads((here/'HISTORICAL_RESULTS.json').read_text('utf-8'))
    expected=historical['source_pins']
    for path,key in [(args.collector_module,'collector_sha256'),(args.executor_module,'executor_sha256'),(args.engineering_protocol,'engineering_protocol_sha256')]:
        assert sha(path)==expected[key],'Source template differs from historical hash'
    collector=load('synthetic_capacity_collector',args.collector_module)
    engine=load('synthetic_capacity_executor',args.executor_module)
    protocol=json.loads(args.engineering_protocol.read_text('utf-8'))
    config=engine.GraphConfig(**protocol['graph']);assert config.rounds==3
    synthetic=[
        {'id':'synthetic_capacity_0','question':'Which person has the blue card?','context':{'title':['Card record','Second record'],'sentences':[['Mira has a blue card.','Leon has a red card.'],['Each person has one card.']]},'type':'synthetic','level':'synthetic'},
        {'id':'synthetic_capacity_1','question':'What label appears on the second box?','context':{'title':['Quoted labels'],'sentences':[['The first label is "A".','The second label is "B".\nBoth labels are printed clearly.']]},'type':'synthetic','level':'synthetic'},
        {'id':'synthetic_capacity_2','question':'Which entry contains the word café?','context':{'title':['Unicode entries','More entries'],'sentences':[['The first entry says café.','The second entry says 茶.'],['No entries are omitted.']]},'type':'synthetic','level':'synthetic'},
    ]
    payloads=[]
    for row in synthetic:
        assert set(row)=={'id','question','context','type','level'}
        payload=engine.Executor.question_payload(collector.collector_payload(row))
        assert json.loads(payload['context'])==[{'title':t,'sentences':s} for t,s in zip(row['context']['title'],row['context']['sentences'])]
        payloads.append(payload)
    from transformers import AutoTokenizer,LlamaTokenizerFast
    import transformers,tokenizers
    declared=json.loads((here/'TOKENIZER_HASHES.json').read_text('utf-8'))
    paths={'qwen':args.qwen_tokenizer,'mistral':args.mistral_tokenizer}
    for item in declared['files']:
        assert sha(paths[item['model']]/item['filename'])==item['sha256'],'Local tokenizer file differs'
    report={}
    class Backend:
        identity={'backend':'CPU_TOKENIZER_ONLY_NO_GENERATION'}
        def __init__(self,tok):self.tok=tok
        def encode(self,text):return self.tok.encode(text,add_special_tokens=False)
        def generate(self,*a,**kw):raise RuntimeError('Inference is forbidden')
    for name,path in paths.items():
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_fast=True) if name=='qwen' else LlamaTokenizerFast.from_pretrained(path,local_files_only=True,trust_remote_code=False,legacy=False)
        exe=engine.Executor(Backend(tok),config)
        lengths=[]
        for payload in payloads:
            row=[]
            for w in range(4):
                messages,received,own=exe._messages(payload,f'r1:w{w}',tuple([1]*12),{})
                assert received==own==0;row.append(len(ids(tok,messages)))
            empty={f'r3:w{w}':SimpleNamespace(wire_message='') for w in range(4)}
            messages,received,own=exe._messages(payload,'aggregate',tuple([1]*12),empty)
            assert received==own==0;row.append(len(ids(tok,messages)))
            lengths.append(row)
        flat=[n for row in lengths for n in row]
        report[name]={'synthetic_rows':3,'template_render_encode_checks':15,'min_prompt_tokens':min(flat),'max_prompt_tokens':max(flat),'tokenizer_class':type(tok).__name__,'legacy_attribute':getattr(tok,'legacy',None),'server_runtime_equivalence':False}
    cuda=bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized());assert not cuda
    result={'schema':'synthetic_public_capacity_interface_validation_v1','status':'PASS_SYNTHETIC_INTERFACE_ONLY','synthetic_rows':3,'benchmark_rows_read':0,'source_hashes_match_historical':True,'tokenizer_file_hashes_match_historical':True,'context_preservation_checked':True,'full_2000_historical_count_repeated':False,'server_runtime_equivalence':False,'models':report,'new_model_calls':0,'new_gpu_hours':0,'new_formal_rounds':0,'CUDA_initialized':False,'environment':{'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,'public_source_sha256':sha(here/'run_capacity.py'),'synthetic_probe_sha256':sha(Path(__file__))}
    with (args.output/'SYNTHETIC_VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    return result
