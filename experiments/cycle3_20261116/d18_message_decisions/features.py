"""CPU-only, input-only realized-message features. Never accesses gold/scored labels."""
import argparse,hashlib,importlib.util,json,re,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--repo',required=True);ap.add_argument('--encoder',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.perf_counter();p=Path(a.private_root);r=Path(a.repo);enc=Path(a.encoder);out=Path(a.output);assert not out.exists();torch.set_num_threads(2);torch.set_num_interop_threads(2)
 read=lambda f:json.loads(f.read_bytes());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 spec=importlib.util.spec_from_file_location('d13',r/'experiments/cycle3_20261116/d13_abstention_replay/diagnose.py');d13=importlib.util.module_from_spec(spec);spec.loader.exec_module(d13)
 pooling=read(enc/'1_Pooling/config.json');assert pooling['pooling_mode_mean_tokens'] and not pooling['pooling_mode_cls_token'];pins={str(f.relative_to(enc)):sha(f) for f in enc.rglob('*') if f.is_file() and f.name!='README.md'}
 tokenizer=AutoTokenizer.from_pretrained(str(enc),local_files_only=True);model=AutoModel.from_pretrained(str(enc),local_files_only=True).cpu().eval();records={};texts=[];text_ids={}
 def add(t):
  if t not in text_ids:text_ids[t]=len(texts);texts.append(t)
  return text_ids[t]
 for task,folder in [('hotpot','annotation_formal7_v2'),('2wiki','wiki_annotation_formal8_v2')]:
  for unit in (p/folder/'monitor/archive').glob('worker_*/unit_*'):
   inp=read(unit/'INPUT.json');mat=read(unit/'MATRIX.json');i=inp['index'];q=inp['payload']['question']
   for row in mat['conditions']:
    if row['condition']!='INDEPENDENT3_Q':continue
    j=[26111671,26111672].index(row['execution_seed']);notes=[row['nodes'][n]['generation']['text'] for n in ['U','V','R']];ids=[add(q)]+[add(t) for t in notes];norm=lambda t:re.sub(r'\W+',' ',t.casefold()).strip();numeric=[float(d13.abstains(t)) for t in notes]+[float(np.log1p(len(t))) for t in notes]+[float(norm(notes[u])==norm(notes[v])) for u,v in [(0,1),(0,2),(1,2)]]
    records[task,i,j]=dict(ids=ids,numeric=numeric,question_id_sha256=hashlib.sha256(inp['payload']['id'].encode()).hexdigest())
 assert len(records)==384
 embeddings=[];truncated=0
 with torch.inference_mode():
  for offset in range(0,len(texts),32):
   batch=texts[offset:offset+32];untruncated=tokenizer(batch,add_special_tokens=True,truncation=False)['input_ids'];truncated+=sum(len(v)>256 for v in untruncated);tokens=tokenizer(batch,padding=True,truncation=True,max_length=256,return_tensors='pt');hidden=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1);v=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1);v=torch.nn.functional.normalize(v,p=2,dim=1);embeddings.append(v.numpy())
  
 embedding=np.concatenate(embeddings);assert embedding.shape==(len(texts),384) and np.isfinite(embedding).all();arrays={};private_groups={}
 old=read(p/'d15_star_input_v1/ENTRIES.json');crossed=read(p/'d16_crossed_input_v1/ENTRIES.json');new={v['job_id']:v['generation']['input_tokens'] for f in (p/'d16_crossed_collection_v1/archive').glob('worker_*/RESULTS.jsonl') for v in map(json.loads,f.read_bytes().splitlines())}
 for task in ['hotpot','2wiki']:
  qemb=np.zeros((96,2,384));msg=np.zeros((96,2,1536));numeric=np.zeros((96,2,15));prefix=np.full((96,2,8),np.nan);terminal=np.full((96,2,8),np.nan);groups=[]
  for i in range(96):
   groups.append(records[task,i,0]['question_id_sha256']);assert records[task,i,1]['question_id_sha256']==groups[-1]
   for j in range(2):
    row=records[task,i,j];v=embedding[row['ids']];qemb[i,j]=v[0];msg[i,j]=v.reshape(-1);numeric[i,j]=row['numeric']+[float(v[u+1]@v[vv+1]) for u,vv in [(0,1),(0,2),(1,2)]]+[float(v[0]@v[k]) for k in [1,2,3]]
  for e in old:
   if e['task']==task and e['mode']=='FILTERED':prefix[e['index'],[26111671,26111672].index(e['seed']),e['mask']]=e['prefix_tokens']
  for e in crossed:
   if e['task']!=task:continue
   i=e['index'];j=[26111671,26111672].index(e['source_seed']);m=e['mask'];value=e['cached']['generation']['input_tokens'] if e['cache_hit'] else new[e['job_id']]
   assert np.isnan(terminal[i,j,m]) or terminal[i,j,m]==value;terminal[i,j,m]=value
  assert np.isfinite(prefix).all() and np.isfinite(terminal).all() and len(set(groups))==96
  for mask in range(8):assert np.array_equal(prefix[:,:,mask],sum((prefix[:,:,1<<e] for e in range(3) if mask&(1<<e)),np.zeros((96,2))))
  for name,value in [('question',qemb),('message',msg),('numeric',numeric),('prefix_tokens',prefix),('terminal_input_tokens',terminal)]:arrays[task+'_'+name]=value
  private_groups[task]=groups
 assert not set(private_groups['hotpot'])&set(private_groups['2wiki']);out.mkdir();np.savez_compressed(out/'FEATURES.npz',**arrays);(out/'PRIVATE_GROUPS.json').write_text(json.dumps(private_groups),encoding='utf-8')
 receipt=dict(status='COMPLETE_INPUT_ONLY_CPU_FEATURES',model_pins=pins,unique_texts=len(texts),truncated_texts=truncated,question_groups=192,source_states=384,embedding_dimensions=384,device='cpu',torch_threads=2,cpu_wall_seconds=time.perf_counter()-start,new_generative_calls=0,new_GPU_hours=0,features_sha256=sha(out/'FEATURES.npz'),groups_sha256=sha(out/'PRIVATE_GROUPS.json'),source_sha256=sha(Path(__file__)),gold_accessed=False,terminal_text_used=False)
 (out/'FEATURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
