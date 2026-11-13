"""CPU input lengths only. No backend import, model constructor or label source."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
sys.dont_write_bytecode=True

N=384
NAMES=('qwen','mistral')
TOKENIZER_FILES=('config.json','tokenizer_config.json','special_tokens_map.json',
 'tokenizer.json','tokenizer.model','vocab.json','merges.txt','added_tokens.json','vocab.txt')
TEXT_RULE='question + ("\\n\\nSupplied context:\\n" + context if context else "")'


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value,newline=False):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+('\n' if newline else '')).encode()


def digest(value,newline=False):return hashlib.sha256(canonical(value,newline)).hexdigest()
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def load_json(path):
    def unique(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique)


def pinned(path,expected):
    require(sha(path)==expected,'Input pin mismatch')
    value=load_json(path);require(sha(path)==expected,'Input changed during read')
    return value


def write_new(path,value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value,True));stream.flush();os.fsync(stream.fileno())
    return sha(path)


def full_text(payload):
    require(set(payload)=={'id','question','context'},'Unexpected collector payload fields')
    require(all(isinstance(payload[k],str) for k in payload),'Collector payload must contain strings')
    return payload['question']+('\n\nSupplied context:\n'+payload['context'] if payload['context'] else '')


def package_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def copy_tokenizer_only(model,destination):
    source=Path(model['path']).resolve(strict=True);manifest=model['identity']['model_files']
    destination.mkdir(exist_ok=False);copied={}
    # External chat templates were not recorded by the qualified backend's
    # manifest suffix allowlist. Refuse them rather than silently omit content.
    require(not (source/'chat_template.jinja').exists() and not (source/'chat_templates').exists(),
            'External chat templates require a separately qualified file manifest')
    for name in TOKENIZER_FILES:
        original=source/name
        if not original.exists():continue
        require(name in manifest,'Tokenizer/config file is absent from qualified manifest')
        resolved=original.resolve(strict=True)
        require(resolved.is_relative_to(source),'Tokenizer file symlink escapes model directory')
        require(original.stat().st_size<=64*1024*1024,'Tokenizer-only small-file size limit exceeded')
        expected=manifest[name]
        require(original.stat().st_size==expected['bytes'] and sha(original)==expected['sha256'],'Qualified tokenizer file changed')
        raw=original.read_bytes()
        with (destination/name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        require(sha(destination/name)==expected['sha256'] and sha(original)==expected['sha256'],'Tokenizer file changed during copy')
        copied[name]={'bytes':len(raw),'sha256':expected['sha256']}
    require('tokenizer_config.json' in copied and any(k in copied for k in ('tokenizer.json','tokenizer.model','vocab.json','vocab.txt')),
            'Incomplete tokenizer-only snapshot')
    return copied


def load_tokenizer(directory,qualified_identity):
    # Explicitly disable framework/model loading; only the tokenizer is needed.
    require('torch' not in sys.modules and 'graph_credit.backend' not in sys.modules,
            'Run tokenizer probe in a fresh process without model/backend imports')
    os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import transformers
    from transformers import AutoTokenizer
    require(transformers.__version__==qualified_identity['transformers'],'Transformers differs from qualified server')
    require(platform.python_version()==qualified_identity['python'],'Python differs from qualified server')
    tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    require(tokenizer.pad_token_id is not None,'Tokenizer has neither pad nor EOS token')
    require(type(tokenizer).__name__==qualified_identity['tokenizer_class'],'Tokenizer class differs from qualification')
    template=tokenizer.chat_template or ''
    require(hashlib.sha256(template.encode()).hexdigest()==qualified_identity['tokenizer_chat_template_sha256'],
            'Tokenizer chat template differs from qualification')
    require('torch' not in sys.modules and 'graph_credit.backend' not in sys.modules,
            'Tokenizer-only execution unexpectedly imported model/backend runtime')
    return tokenizer,{'python':platform.python_version(),'transformers':transformers.__version__,
        'tokenizers':package_version('tokenizers'),'sentencepiece':package_version('sentencepiece'),
        'protobuf':package_version('protobuf')}


def tokenizer_identity(tokenizer,files,versions):
    return {'schema':'qualified_server_tokenizer_input_encoding_v1','files':files,'versions':versions,
        'constructor':'transformers.AutoTokenizer.from_pretrained',
        'constructor_arguments':{'local_files_only':True,'trust_remote_code':False},
        'tokenizer_class':type(tokenizer).__name__,'tokenizer_module':type(tokenizer).__module__,
        'is_fast':bool(tokenizer.is_fast),'legacy_attribute':getattr(tokenizer,'legacy',None),
        'chat_template_sha256':hashlib.sha256((tokenizer.chat_template or '').encode()).hexdigest(),
        'backend_tokenizer_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id,
        'bos_token_id':tokenizer.bos_token_id,'encode_arguments':{'add_special_tokens':False},
        'pad_fallback':'eos_when_pad_is_none_same_as_HFBackend'}


def run_probe(*,selection,selection_sha256,selection_receipt,selection_receipt_sha256,
              questions,questions_sha256,bindings,bindings_sha256,output,_loader=None):
    started=time.perf_counter();cpu_started=time.process_time()
    manifest=pinned(selection,selection_sha256);selected=manifest['questions']
    receipt=pinned(selection_receipt,selection_receipt_sha256)
    binding=pinned(bindings,bindings_sha256)
    require(receipt['status']=='COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION','Selection did not complete')
    require(receipt['selection_manifest_sha256']==selection_sha256 and receipt['questions_jsonl_sha256']==questions_sha256,'Selection/input lineage differs')
    require(len(selected)==N and len({r['id'] for r in selected})==N,'Exactly 384 unique selected IDs required')
    require(sha(questions)==questions_sha256,'Questions bytes differ from selection')
    require(binding['schema']=='hotpot_server_tokenizer_probe_bindings_v1','Unknown probe binding schema')
    sources=binding['sources']
    require(set(sources)=={'collector','backend','executor'},'Pinned input/tokenizer implementation sources required')
    for source in sources.values():require(sha(source['path'])==source['sha256'],'Pinned source changed')
    acceptance_spec=binding['engineering_acceptance']
    acceptance=pinned(acceptance_spec['path'],acceptance_spec['sha256'])
    require(acceptance['decision']=='PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION','Qualified root acceptance required')
    require(acceptance['matrix_protocol_hash']==binding['matrix_protocol_hash'],'Qualified matrix identity differs')
    identity_evidence=acceptance['evidence']['identity'];relative=Path(identity_evidence['path'])
    bundle=Path(acceptance_spec['path']).resolve().parent
    require(not relative.is_absolute() and '..' not in relative.parts,'Qualification identity must be bundle-relative')
    qualified_path=(bundle/relative).resolve()
    require(qualified_path.is_relative_to(bundle),'Qualification identity escapes acceptance bundle')
    qualified=pinned(qualified_path,identity_evidence['sha256'])
    require(qualified['matrix_protocol_hash']==binding['matrix_protocol_hash'],'Qualification identity matrix differs')
    for name in ('backend','executor'):
        require(qualified['source_manifest']['src/graph_credit/'+name+'.py']==sources[name]['sha256'],
                'Input/tokenizer source is not the qualified source')
    require(set(binding['models'])==set(NAMES),'Both tokenizer model references required')
    for name in NAMES:
        identity=binding['models'][name]['identity']
        require(digest(identity)==acceptance['model_identity_sha256'][name],'Model identity is not the accepted qualified identity')
        require(digest(identity)==digest(qualified['backend'][name]),'Model identity differs from original qualification artifact')
        require(identity['backend_code_sha256']==sources['backend']['sha256'],'Tokenizer construction source differs')
    spec=importlib.util.spec_from_file_location('pinned_input_only_collector',sources['collector']['path'])
    collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
    rows=[]
    for line in Path(questions).read_bytes().splitlines():
        value=json.loads(line)
        require(set(value)=={'id','question','context','type','level'},'Only selected five input fields are permitted')
        rows.append(value)
    require(len(rows)==N,'Selected question coverage differs')
    payloads=[]
    for i,row in enumerate(rows):
        require(row['id']==selected[i]['id'] and digest(row,True)==selected[i]['input_sha256'],'Manifest order or row bytes differ')
        payload=collector.collector_payload(row)
        require(payload['id']=='hotpotqa_'+row['id'],'Collector question identity differs')
        payloads.append((payload,full_text(payload)))
    require(sha(questions)==questions_sha256,'Question file changed during decode')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    private=out/'private_tokenizers';private.mkdir()
    length_rows=[{'id':row['id'],'payload_sha256':digest(payload)} for row,(payload,_) in zip(rows,payloads)]
    identities={};model_timing={};loader=_loader or load_tokenizer
    for name in NAMES:
        begin=time.perf_counter();begin_cpu=time.process_time();model=binding['models'][name]
        copied=copy_tokenizer_only(model,private/name)
        tokenizer,versions=loader(private/name,model['identity'])
        identity=tokenizer_identity(tokenizer,copied,versions)
        if model.get('tokenizer_identity_sha256') is not None:
            require(digest(identity)==model['tokenizer_identity_sha256'],'Previously pinned tokenizer identity differs')
        for record,(_,text) in zip(length_rows,payloads):
            count=len(tokenizer.encode(text,add_special_tokens=False))
            require(type(count) is int and count>0,'Invalid complete-payload token length')
            record[name+'_tokens']=count
        identities[name]=identity
        model_timing[name]={'wall_seconds':time.perf_counter()-begin,'cpu_seconds':time.process_time()-begin_cpu}
    # All pinned data/source bytes must still match before successful receipt.
    for path,pin in ((selection,selection_sha256),(selection_receipt,selection_receipt_sha256),
                     (questions,questions_sha256),(bindings,bindings_sha256),(acceptance_spec['path'],acceptance_spec['sha256']),
                     (qualified_path,identity_evidence['sha256'])):
        require(sha(path)==pin,'Input changed during tokenizer counting')
    for source in sources.values():require(sha(source['path'])==source['sha256'],'Source changed during counting')
    for name in NAMES:
        for file,meta in identities[name]['files'].items():
            require(sha(private/name/file)==meta['sha256'] and sha(Path(binding['models'][name]['path'])/file)==meta['sha256'],'Tokenizer files changed during counting')
    runtime={'status':'TOKENIZER_ONLY_NO_MODEL_INFERENCE','classification':'PRIVATE_NO_LABELS','tokenizer_identities':identities,
        'qualified_model_identity_reference':{name:binding['models'][name]['identity'] for name in NAMES},
        'qualification_acceptance_sha256':acceptance_spec['sha256'],'probe_binding_sha256':bindings_sha256,
        'source_pins':{name:row['sha256'] for name,row in sources.items()},
        'probe_source_sha256':sha(__file__),'timing':model_timing,
        'weights_opened':False,'backend_imported':False,'GPU_calls':0,'labels_read':False,
        'scope':'Exact current server tokenizer input encoding; qualified GPU/model execution is referenced, not rerun.'}
    runtime_pin=write_new(out/'RUNTIME.json',runtime)
    result={'status':'TOKENIZER_LENGTHS_COMPLETE_NO_LABELS','classification':'PRIVATE_NO_LABELS','labels_read':False,'server_runtime_equivalence':True,
        'server_runtime_equivalence_scope':'server_tokenizer_input_encoding_only_not_new_GPU_runtime_qualification',
        'questions_jsonl_sha256':questions_sha256,'selection_manifest_sha256':selection_sha256,
        'selection_receipt_sha256':selection_receipt_sha256,'collector_source_sha256':sources['collector']['sha256'],
        'payload_hash_algorithm':'graph_credit_canonical_utf8_sha256_no_newline',
        'tokenizer_identity_sha256':{name:digest(identity) for name,identity in identities.items()},
        'model_identity_sha256':{name:digest(binding['models'][name]['identity']) for name in NAMES},
        'rows':length_rows,'text_rule':TEXT_RULE,'add_special_tokens':False,
        'chat_or_system_or_future_messages_included':False,'context_truncated_or_filtered':False,
        'runtime_receipt_sha256':runtime_pin,'model_calls':0,'GPU_hours':0}
    result_pin=write_new(out/'LENGTHS.json',result)
    complete={'status':'COMPLETE_TOKENIZER_ONLY_NOT_GPU_ADMISSION','rows':N,'lengths_sha256':result_pin,
        'runtime_sha256':runtime_pin,'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu_started,
        'new_GPU_hours':0,'labels_read':False,'weights_opened':False,'model_calls':0}
    write_new(out/'COMPLETE.json',complete)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('selection','selection-sha256','selection-receipt','selection-receipt-sha256',
                 'questions','questions-sha256','bindings','bindings-sha256','output'):
        parser.add_argument('--'+name,required=True)
    try:
        result=run_probe(**vars(parser.parse_args()))
        print(json.dumps({'status':result['status'],'rows':len(result['rows']),'model_calls':0,'GPU_hours':0}))
    except Exception as error:
        # Do not print exception text that might contain private input text.
        print(json.dumps({'status':'INCOMPLETE_TOKENIZER_PROBE','error_type':type(error).__name__,'automatic_retry':False}))
        raise SystemExit(2)


if __name__=='__main__':main()
