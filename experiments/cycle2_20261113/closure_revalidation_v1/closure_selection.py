"""Qualified-server tokenizers and input-only capacity selection, no GPU or gold."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1')
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--plan-sha',required=True);a=ap.parse_args()
 assert sha(a.plan)==a.plan_sha;plan=read(a.plan);assert plan['source_sha256']==sha(__file__) and plan['action']=='INPUT_ONLY_CLOSURE_CAPACITY'
 for v in plan['pins'].values():assert sha(v['path'])==v['sha256']
 for d in reversed(plan['import_paths']):sys.path.insert(0,d)
 import tokenizer_lengths as tk
 import select_hotpot as h
 import exogenous_path_operator as op
 contract=read(plan['pins']['contract']['path']);assert contract['status']=='FROZEN_INPUT_ONLY' and contract['selected_questions']==640
 pool=[json.loads(x) for x in Path(plan['pins']['pool']['path']).read_bytes().splitlines()];assert len(pool)==1024
 receipt=read(plan['pins']['pool_receipt']['path']);assert receipt['pool_sha256']==sha(plan['pins']['pool']['path']) and not receipt['labels_projected']
 exp=read(plan['pins']['exposures']['path']);ids=set(exp['ids']);groups=set(exp['question_group_hashes']);content=set(exp['content_hashes'])
 out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);toks={};identities={}
 for n in ['qwen','mistral']:
  spec=plan['models'][n];dest=out/('private_tokenizer_'+n);files=tk.copy_tokenizer_only(spec,dest);tok,versions=tk.load_tokenizer(dest,spec['identity']);toks[n]=tok;identities[n]=tk.tokenizer_identity(tok,files,versions)
 measures=[];eligible=[]
 for i,row in enumerate(pool):
  m={'index':i,'input_sha256':hashlib.sha256(h.canonical_bytes(row)).hexdigest(),'ten_documents':len(row['context']['title'])==10,'lengths':{}}
  if m['ten_documents']:
   p={k:row[k] for k in ['id','question','context']}
   for n,mode in [('qwen','INDEPENDENT3_U'),('mistral','INDEPENDENT3_R'),('qwen','SINGLE_Q')]:m['lengths'][mode]=len(toks[n].apply_chat_template(op.build_full_context_messages(p,mode),tokenize=True,add_generation_prompt=True))
  measures.append(m)
  assert row['id'] not in ids and h.question_group_sha256(row['question']) not in groups and h.content_sha256(row) not in content
  if m['ten_documents'] and max(m['lengths'].values())<=7168:eligible.append(i)
 assert len(eligible)>=640,'No input pool resampling after capacity result'
 chosen=eligible[:640];assert len({h.question_group_sha256(pool[i]['question']) for i in chosen})==640
 shocks=np.random.Generator(np.random.PCG64(261115702)).permutation(['U']*4+['V']*4).tolist()
 put(out/'CAPACITY.json',{'measurements':measures,'tokenizers':identities,'dynamic_capacity_guarantee':False})
 with (out/'QUESTIONS.jsonl').open('xb') as f:
  for i in chosen:f.write(h.canonical_bytes(pool[i]))
 selection={'schema':'mab2_closure_ics_selection_v1','contract_sha256':plan['pins']['contract']['sha256'],'questions_sha256':sha(out/'QUESTIONS.jsonl'),'capacity_sha256':sha(out/'CAPACITY.json'),
  'generation_seeds':contract['generation_seeds'],'episode_shocks':shocks,'questions':[{'index':j,'pool_index':i,'episode':j//80,'role':'I' if j%80<8 else 'C' if j%80<16 else 'S','id':pool[i]['id'],
   'input_sha256':hashlib.sha256(h.canonical_bytes(pool[i])).hexdigest(),'question_group_sha256':h.question_group_sha256(pool[i]['question']),'content_sha256':h.content_sha256(pool[i]),'removed_upstream':shocks[j//80]} for j,i in enumerate(chosen)]}
 put(out/'SELECTION_MANIFEST.json',selection)
 result={'status':'COMPLETE_INPUT_ONLY_ICS_CAPACITY_SELECTION','selected_questions':640,'episodes':8,'roles_per_episode':{'I':8,'C':8,'S':64},'eligible_pool_questions':len(eligible),'known_exposure_union_groups':len(groups),
  'questions_sha256':sha(out/'QUESTIONS.jsonl'),'manifest_sha256':sha(out/'SELECTION_MANIFEST.json'),'capacity_sha256':sha(out/'CAPACITY.json'),'source_sha256':sha(__file__),'plan_sha256':a.plan_sha,
  'labels_read':False,'GPU_hours':0,'official_validation_test_opened':False,'scientific_admission':False,'static_max':7168,'runtime_max':8192,'unknown_cross_project_exposure':'UNKNOWN'}
 put(out/'RECEIPT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
