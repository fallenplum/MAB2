"""CPU-only exact request/tokenizer/cache replay of closed private archives."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,copy,dataclasses,hashlib,json,sys,types
from pathlib import Path
sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
@dataclasses.dataclass
class Generation:
 text:str;input_tokens:int;output_tokens:int;elapsed_seconds:float;gpu_seconds:float;gpu_peak_allocated_bytes:int;finish_reason:str;call_id:str
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert sha(a.plan)==a.plan_sha
 plan=read(a.plan);root=Path(plan['output']);out=Path(a.output);assert not out.exists()
 for path,pin in plan['source_pins'].items():assert sha(path)==pin
 for path in reversed(plan['import_paths']):sys.path.insert(0,path)
 from closure_operator import ClosureOperator as PathOperator
 from exogenous_path_operator import digest
 from durable_backend import read_journal
 from transformers import AutoTokenizer
 import transformers
 manifest=read(root/'ARCHIVE_MANIFEST.json')
 for rel,pin in manifest.items():
  path=(root/rel).resolve();assert path.is_relative_to(root.resolve()) and sha(path)==pin
 closure=read(root/'CLOSURE.json');assert closure['all_workers_exited']
 class ReplayBackend:
  def __init__(self,name,journal):
   self.identity=journal['header']['backend_identity'];self.physical_calls=[];self.cursor=0
   self.events=list(journal['requests'].values());assert not journal['pending_request_ids']
   model=plan['models'][name];assert digest(self.identity)==digest(model['identity'])
   assert transformers.__version__==self.identity['transformers']
   self.tokenizer=AutoTokenizer.from_pretrained(model['path'],local_files_only=True,trust_remote_code=False)
   if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
   cfg=Path(model['path'])/'config.json'
   assert sha(cfg)==model['identity']['model_files']['config.json']['sha256']
   self.model=types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=read(cfg)['max_position_embeddings']))
  def encode(self,text):return self.tokenizer.encode(text,add_special_tokens=False)
  def generate(self,messages,seed,max_new_tokens,config):
   row=self.events[self.cursor];self.cursor+=1;s=row['start'];t=row['terminal'];assert t and t['event']=='RETURNED'
   assert s['messages']==messages and s['seed']==seed and s['max_new_tokens']==max_new_tokens and s['config']==dataclasses.asdict(config)
   self.physical_calls.extend(copy.deepcopy(t['native_events']))
   return Generation(**t['generation'])
 checks=0;complete=0;native=0;logical=0;nonempty=0
 for wd in sorted(root.glob('worker_*')):
  if not wd.is_dir():continue
  bs={name:ReplayBackend(name,read_journal(wd/(name+'_requests.jsonl'))) for name in ['qwen','mistral']}
  op=PathOperator(bs['qwen'],bs['mistral'])
  for u in sorted(wd.glob('unit_*')):
   if not (u/'COMPLETE.json').exists():continue
   inp=read(u/'INPUT.json');original=read(u/'MATRIX.json')
   replay=op.run_matrix(inp['payload'],plan['seeds'],inp['shock'])
   assert replay['protocol_sha256']==original['protocol_sha256'];checks+=1
   assert replay['logical_completed_node_cost_by_model']==original['logical_completed_node_cost_by_model'];checks+=1
   assert replay['physical_cost_by_model']==original['physical_cost_by_model'];checks+=1
   for c1,c2 in zip(original['conditions'],replay['conditions']):
    for field in ['key','edges','status','answer','logical_service_cost_by_model','cache_reuse_count']:
     assert c1[field]==c2[field];checks+=1
    if 'executed_nodes' in c1:
     assert c1['executed_nodes']==c2['executed_nodes'] and c1['skipped_dead_nodes']==c2['skipped_dead_nodes'];checks+=2
    if c1['condition']=='VOTE_Q3':
     assert c1['selected_node']==c2['selected_node'] and c1['answer_rule']==c2['answer_rule'];checks+=2
    for node in c1['nodes']:
     for field in ['request','request_fingerprint','generation','wire_message','received_message_tokens','reused','is_final']:
      assert c1['nodes'][node][field]==c2['nodes'][node][field];checks+=1
    nonempty+=bool(c1['answer'])
   complete+=1;logical+=sum(x['calls'] for x in original['logical_completed_node_cost_by_model'].values())
  for b in bs.values():
   assert b.cursor==len(b.events);native+=b.cursor;checks+=1
 expected=2 if plan['mode']=='qualification' else 640
 assert complete==expected and closure['status']=='COMPLETE_UNSCORED'
 result={'status':'PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY','completed_units':complete,'logical_calls_replayed':logical,
  'original_physical_calls_replayed':native,'field_equalities':checks,'nonempty_final_outputs':nonempty,
  'new_GPU_hours':0,'new_model_calls':0,'labels_read':False,'scores_computed':False,'bootstrap_runs':0,
  'plan_sha256':a.plan_sha,'archive_manifest_sha256':sha(root/'ARCHIVE_MANIFEST.json'),'source_sha256':sha(__file__),
  'limitation':'Replay uses recorded generations; it does not regenerate stochastic tokens or establish efficacy'}
 out.mkdir(parents=True);(out/'RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
