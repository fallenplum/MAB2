import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import tokenizer_lengths as probe

BASE=Path(__file__).parent.resolve()
(BASE/'cpu_tmp').mkdir(exist_ok=True)
tempfile.tempdir=str(BASE/'cpu_tmp')


class FakeTokenizer:
    pad_token_id=None
    eos_token_id=2
    bos_token_id=1
    is_fast=True
    chat_template='synthetic template only'
    backend_tokenizer=types.SimpleNamespace(to_str=lambda:'synthetic tokenizer backend')
    def encode(self,text,add_special_tokens):
        if add_special_tokens is not False:raise AssertionError('Special tokens changed')
        return list(text.encode())


def fixture(root):
    collector=root/'collector.py'
    collector.write_text('import json\ndef collector_payload(row):\n return {"id":"hotpotqa_"+row["id"],"question":row["question"],"context":json.dumps([{"title":t,"sentences":s} for t,s in zip(row["context"]["title"],row["context"]["sentences"])],ensure_ascii=False,separators=(",",":"))}\n',encoding='utf-8')
    sources={'collector':{'path':str(collector),'sha256':probe.sha(collector)}}
    for name in ('backend','executor'):
        path=root/(name+'.py');path.write_text('raise AssertionError("Never import this source")\n',encoding='utf-8')
        sources[name]={'path':str(path),'sha256':probe.sha(path)}
    models={}
    for name in probe.NAMES:
        path=root/name;path.mkdir()
        files={}
        for filename,content in [('tokenizer_config.json',b'{}'),('tokenizer.json',b'{"fake":true}'),
                                 ('config.json',b'{}'),('model.safetensors',b'NEVER OPEN THESE SYNTHETIC WEIGHT BYTES')]:
            (path/filename).write_bytes(content)
            files[filename]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
        models[name]={'path':str(path),'identity':{'model_files':files,
            'backend_code_sha256':sources['backend']['sha256'],'tokenizer_class':'FakeTokenizer',
            'tokenizer_chat_template_sha256':hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest(),
            'transformers':'SYNTHETIC','python':platform.python_version()}}
    qualified=root/'IDENTITY.json'
    probe.write_new(qualified,{'matrix_protocol_hash':'a'*64,'backend':{n:m['identity'] for n,m in models.items()},
        'source_manifest':{'src/graph_credit/'+n+'.py':sources[n]['sha256'] for n in ('backend','executor')}})
    acceptance=root/'ACCEPTANCE.json'
    probe.write_new(acceptance,{'decision':'PASS_SYNTHETIC_MATRIX_ENGINEERING_FOR_COLLECTION','matrix_protocol_hash':'a'*64,
        'model_identity_sha256':{n:probe.digest(m['identity']) for n,m in models.items()},
        'evidence':{'identity':{'path':qualified.name,'sha256':probe.sha(qualified)}}})
    bindings=root/'BINDINGS.json'
    probe.write_new(bindings,{'schema':'hotpot_server_tokenizer_probe_bindings_v1','sources':sources,
        'models':models,'matrix_protocol_hash':'a'*64,'engineering_acceptance':{'path':str(acceptance),'sha256':probe.sha(acceptance)}})
    rows=[{'id':f'artificial_{i:03d}','question':f'Full synthetic question {i}',
           'context':{'title':['Document'],'sentences':[['All artificial context retained.']]},'type':'bridge','level':'easy'} for i in range(384)]
    selection=root/'SELECTION.json'
    probe.write_new(selection,{'questions':[{'id':r['id'],'input_sha256':probe.digest(r,True),
        'edge_index':(1,5,7,9)[i%4],'generation_seed1':26111301,'generation_seed2':26111302} for i,r in enumerate(rows)]})
    questions=root/'questions.jsonl';questions.write_bytes(b''.join(probe.canonical(r,True) for r in rows))
    receipt=root/'SELECTION_RECEIPT.json'
    probe.write_new(receipt,{'status':'COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION',
        'selection_manifest_sha256':probe.sha(selection),'questions_jsonl_sha256':probe.sha(questions)})
    args={name:str(path) for name,path in [('selection',selection),('selection_receipt',receipt),('questions',questions),('bindings',bindings)]}
    args.update({name+'_sha256':probe.sha(path) for name,path in list(args.items())})
    args['output']=str(root/'output')
    return rows,args


def fake_loader(path,identity):
    tokenizer=FakeTokenizer();tokenizer.pad_token_id=tokenizer.eos_token_id
    return tokenizer,{'python':platform.python_version(),'transformers':'SYNTHETIC',
        'tokenizers':'SYNTHETIC','sentencepiece':None,'protobuf':None}


class LengthTests(unittest.TestCase):
    def test_full_payload_receipt_passes_unchanged_runner_schema_without_weights(self):
        runner=Path(os.environ.get('MAB2_RUNNER_DIR',str(BASE.parent/'hotpot_collection_runner_stage')))
        sys.path.insert(0,str(runner))
        import orchestration_core
        import plan_tools
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rows,args=fixture(root);native_open=Path.open;opened=[]
            def guarded(path,*a,**k):
                if path.suffix in ('.safetensors','.bin'):
                    opened.append(str(path));raise AssertionError('Weight opened')
                return native_open(path,*a,**k)
            with patch.object(Path,'open',guarded):
                result=probe.run_probe(**args,_loader=fake_loader)
            self.assertEqual(opened,[]);self.assertEqual(len(result['rows']),384)
            runtime=probe.load_json(args['bindings'])
            runtime['collector']=runtime['sources']['collector']
            for name in probe.NAMES:
                runtime['models'][name]['tokenizer_identity_sha256']=result['tokenizer_identity_sha256'][name]
            plan_tools.validate_lengths(result,runtime)
            selected=probe.load_json(args['selection'])['questions']
            plan_tools.verify_payloads(rows,selected,result,runtime)
            schedule=orchestration_core.build_schedule(selected,result['rows'],3)
            self.assertEqual(len(schedule['prefix_indices']),8)
            self.assertEqual(set(p.suffix for p in (root/'output/private_tokenizers').rglob('*') if p.is_file()),{'.json'})
            self.assertNotIn('Full synthetic question',(root/'output/LENGTHS.json').read_text())
            self.assertFalse(result['chat_or_system_or_future_messages_included'])

    def test_changed_tokenizer_file_rejected_before_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,args=fixture(root);(root/'qwen/tokenizer.json').write_bytes(b'changed')
            loaded=[]
            with self.assertRaises(ValueError):probe.run_probe(**args,_loader=lambda *a:loaded.append(True))
            self.assertEqual(loaded,[]);self.assertFalse((root/'output/LENGTHS.json').exists())

    def test_changed_question_or_order_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,args=fixture(root)
            raw=Path(args['questions']).read_bytes().splitlines();raw.reverse()
            Path(args['questions']).write_bytes(b'\n'.join(raw)+b'\n')
            with self.assertRaises(ValueError):probe.run_probe(**args,_loader=fake_loader)
            self.assertFalse((root/'output').exists())

    def test_wrong_qualified_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,args=fixture(root);binding=probe.load_json(args['bindings'])
            binding['models']['qwen']['identity']['python']='wrong'
            Path(args['bindings']).write_bytes(probe.canonical(binding,True));args['bindings_sha256']=probe.sha(args['bindings'])
            with self.assertRaises(ValueError):probe.run_probe(**args,_loader=fake_loader)

    def test_new_output_required(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,args=fixture(root);Path(args['output']).mkdir()
            with self.assertRaises(FileExistsError):probe.run_probe(**args,_loader=fake_loader)

    def test_loader_uses_exact_auto_constructor_and_pad_fallback(self):
        seen=[]
        def construct(path,**kwargs):seen.append(kwargs);return FakeTokenizer()
        module=types.SimpleNamespace(__version__='SYNTHETIC',AutoTokenizer=types.SimpleNamespace(from_pretrained=construct))
        qualified={'transformers':'SYNTHETIC','python':platform.python_version(),
            'tokenizer_class':'FakeTokenizer','tokenizer_chat_template_sha256':hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest()}
        with patch.dict(sys.modules,{'transformers':module}),patch.dict(os.environ,{},clear=False):
            tokenizer,_=probe.load_tokenizer(Path('UNUSED_SYNTHETIC'),qualified)
            self.assertEqual(tokenizer.pad_token_id,2)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(seen,[{'local_files_only':True,'trust_remote_code':False}])

    def test_loader_version_drift_fails_closed(self):
        module=types.SimpleNamespace(__version__='WRONG',AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:self.fail('Must not load')))
        with patch.dict(sys.modules,{'transformers':module}),patch.dict(os.environ,{},clear=False):
            with self.assertRaises(ValueError):probe.load_tokenizer(Path('UNUSED'),{'transformers':'SYNTHETIC'})

    def test_text_rule_preserves_whole_context(self):
        payload={'id':'synthetic','question':' Q\n','context':' [all context]\n'}
        self.assertEqual(probe.full_text(payload),' Q\n\n\nSupplied context:\n [all context]\n')
        self.assertEqual(probe.full_text({**payload,'context':''}),' Q\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    started=__import__('time').perf_counter();stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LengthTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue(),end='')
    out=Path(args.output);out.mkdir(exist_ok=False)
    (out/'STDOUT.txt').write_text(stream.getvalue(),encoding='utf-8')
    probe.write_new(out/'RECEIPT.json',{'status':'PASS_CPU_SYNTHETIC_ONLY_NOT_REAL_TOKENIZER_VALIDATION' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'source_sha256':probe.sha(BASE/'tokenizer_lengths.py'),'test_sha256':probe.sha(__file__),
        'elapsed_seconds':__import__('time').perf_counter()-started,'real_questions_read':False,'models_loaded':False,'GPU_hours':0})
    if not result.wasSuccessful():raise SystemExit(1)
