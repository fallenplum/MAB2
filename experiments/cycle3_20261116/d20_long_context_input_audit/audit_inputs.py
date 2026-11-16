"""Schema/length/group audit. Gold is vaulted and never used for eligibility."""
import argparse,collections,hashlib,json,re,time
from pathlib import Path
from transformers import AutoTokenizer
import psutil
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--tokenizer',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'longbench_input_audit_v1';receipt=json.loads((run/'DOWNLOAD_RECEIPT.json').read_bytes());source=Path(receipt['local_path']);assert psutil.virtual_memory().available>4*1024**3;start=time.perf_counter();raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['sha256'];rows=json.loads(raw);del raw;assert isinstance(rows,list) and len(rows)==503;tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);meta=[];vault={};parent=list(range(len(rows)));owners={};required={'_id','domain','sub_domain','difficulty','length','question','choice_A','choice_B','choice_C','choice_D','answer','context'}
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 def digest(t):return hashlib.sha256(t.encode('utf-8')).hexdigest()
 for i,row in enumerate(rows):
  assert required<=set(row);qid=row['_id'];assert qid not in vault and row['answer'] in ['A','B','C','D'];vault[qid]=row['answer'];context=' '.join(row['context'].split());question=' '.join(row['question'].casefold().split());contextsha=digest(context);qsha=digest(question)
  for kind,h in [('context',contextsha),('question',qsha)]:
   if (kind,h) in owners:union(i,owners[kind,h])
   else:owners[kind,h]=i
  domain=row['domain'];norm=domain.lower().replace('-',' ').replace('_',' ');category='single' if 'single' in norm and 'qa' in norm else ('multi' if 'multi' in norm and 'qa' in norm else None);words=len(context.split());count=None
  if category and words<=30000:count=len(tokenizer.encode(row['context'],add_special_tokens=False,truncation=False))
  eligible=category is not None and words<=30000 and count is not None and 8192<=count<=24576;meta.append(dict(index=i,id=qid,domain=domain,category=category,context_sha256=contextsha,question_sha256=qsha,context_words=words,context_tokens=count,eligible=eligible))
 groups=collections.defaultdict(list)
 for i,row in enumerate(meta):groups[find(i)].append(row['id'])
 group_hash={key:digest(json.dumps(sorted(ids),separators=(',',':'))) for key,ids in groups.items()}
 for i,row in enumerate(meta):row['group_sha256']=group_hash[find(i)]
 eligible={c:{row['group_sha256'] for row in meta if row['eligible'] and row['category']==c} for c in ['single','multi']};mixed=eligible['single']&eligible['multi'];counts={c:len(v-mixed) for c,v in eligible.items()};pilot=32 if min(counts.values())>=48 else (16 if min(counts.values())>=24 else 0)
 # Mixed-domain groups are excluded from domain quota eligibility, never split.
 for row in meta:row['quota_eligible']=row['eligible'] and row['group_sha256'] not in mixed
 def write_new(name,obj):
  path=run/name
  with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
  return hashlib.sha256(path.read_bytes()).hexdigest()
 vaultsha=write_new('SEALED_GOLD_VAULT.json',vault);metasha=write_new('PRIVATE_INPUT_METADATA.json',meta);domain_counts=dict(collections.Counter(row['domain'] for row in meta));coverage={domain:dict(total=sum(row['domain']==domain for row in meta),eligible=sum(row['domain']==domain and row['eligible'] for row in meta)) for domain in domain_counts};result=dict(status='COMPLETE_INPUT_ONLY_LONG_CONTEXT_AUDIT',records=len(rows),groups=len(groups),domain_coverage=coverage,eligible_distinct_groups=counts,mixed_domain_groups_excluded=len(mixed),eligible_pilot_question_count=pilot,feasible=pilot>0,official_raw_sha256=receipt['sha256'],metadata_sha256=metasha,vault_sha256=vaultsha,answers_displayed=False,answer_values_used_for_selection=False,new_model_calls=0,new_GPU_hours=0,cpu_wall_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());write_new('INPUT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
