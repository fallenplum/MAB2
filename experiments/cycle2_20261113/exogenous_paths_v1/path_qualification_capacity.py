import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1')
import argparse,json,hashlib,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()==a.plan_sha
plan=json.loads(Path(a.plan).read_bytes())
for f,h in plan['source_pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
for path in reversed(plan['import_paths']):sys.path.insert(0,path)
from path_runtime_v2 import synthetic
from exogenous_path_operator import input_only_messages
from transformers import AutoTokenizer
import transformers
tokens={}
for model in ['qwen','mistral']:
 spec=plan['models'][model];assert transformers.__version__==spec['identity']['transformers']
 tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True,trust_remote_code=False)
 assert hashlib.sha256((tok.chat_template or '').encode()).hexdigest()==spec['identity']['tokenizer_chat_template_sha256']
 tokens[model]=tok
rows=[]
for long in [False,True]:
 counts={k:len(tokens[v['model']].apply_chat_template(v['messages'],tokenize=True,add_generation_prompt=True)) for k,v in input_only_messages(synthetic(long)).items()}
 rows.append({'synthetic_long':long,'static_prompt_tokens':counts});assert max(counts.values())<=7168
result={'status':'QUALIFICATION_STATIC_CAPACITY_PASS','rows':rows,'GPU_hours':0,'labels_read':False,'weights_loaded':False,
 'dynamic_overflow_still_guarded':True,'static_check_not_dynamic_capacity_proof':True,'plan_sha256':a.plan_sha}
out=Path(a.output);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
