import argparse,hashlib,json,collections,time
from pathlib import Path
from transformers import AutoTokenizer
def canonical_document(paper):
 pieces=['TITLE',paper['title'],'ABSTRACT',paper['abstract']]
 for section in paper['full_text']:
  pieces.append(section['section_name'] or '')
  for paragraph in section['paragraphs']:assert isinstance(paragraph,str);pieces.append(paragraph)
 for figure in paper['figures_and_tables']:pieces.append(figure['caption'] or '')
 assert all(isinstance(v,str) for v in pieces)
 return '\n\n'.join(pieces)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--private-root',required=True);ap.add_argument('--tokenizer',required=True);a=ap.parse_args();p=Path(a.private_root);run=p/'qasper_input_audit_v1';read=lambda f:json.loads(f.read_bytes());receipt=read(run/'DOWNLOAD_RECEIPT.json');raw=Path(receipt['train_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==receipt['train_sha256'];papers=json.loads(raw);del raw;start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);meta=[];vault={};titles={};seen_questions=set();norm=lambda x:' '.join(x.casefold().split());sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
 for paper_id,paper in papers.items():
  context=canonical_document(paper);words=len(context.split());tokens=len(tokenizer.encode(context,add_special_tokens=False,truncation=False)) if words<=30000 else None;eligible=tokens is not None and 8192<=tokens<=24576;group=sha(' '.join(context.split()));qmeta=[]
  for q in paper['qas']:
   qid=q['question_id'];assert qid not in seen_questions;seen_questions.add(qid);assert isinstance(q['answers'],list) and q['answers'];vault[qid]=[v['answer'] for v in q['answers']];qmeta.append(dict(id=qid,question_sha256=sha(norm(q['question']))))
  eligible=eligible and len(qmeta)>0;meta.append(dict(paper_id=paper_id,group_sha256=group,context_words=words,context_tokens=tokens,eligible=eligible,questions=qmeta))
  if eligible and len(norm(paper['title']))>=12:titles[paper_id]=norm(paper['title'])
 longrun=p/'longbench_input_audit_v1';longmeta=read(longrun/'PRIVATE_INPUT_METADATA.json');longreceipt=read(longrun/'DOWNLOAD_RECEIPT.json');longraw=Path(longreceipt['local_path']).read_bytes();assert hashlib.sha256(longraw).hexdigest()==longreceipt['sha256'];longrows=json.loads(longraw);del longraw;overlaps=[];retained=set();candidates=set()
 for row in longmeta:
  if not row['quota_eligible'] or row['category']!='single':continue
  candidates.add(row['group_sha256']);text=norm(longrows[row['index']]['context']);matched=[key for key,title in titles.items() if title in text]
  if matched:overlaps.append(dict(longbench_id=row['id'],longbench_group=row['group_sha256'],qasper_paper_ids=matched))
 excluded={v['longbench_group'] for v in overlaps};retained=candidates-excluded;groups={v['group_sha256'] for v in meta if v['eligible']};counts=collections.Counter(v['group_sha256'] for v in meta);feasible=len(groups)>=96 and len(retained)>=32
 def save(name,obj):
  file=run/name
  with file.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
  return hashlib.sha256(file.read_bytes()).hexdigest()
 vaultsha=save('SEALED_GOLD_VAULT.json',vault);metasha=save('PRIVATE_INPUT_METADATA.json',meta);oversha=save('PRIVATE_OVERLAP_SCREEN.json',overlaps);retsha=save('PRIVATE_RESERVED_TRANSFER_GROUPS.json',sorted(retained));result=dict(status='COMPLETE_INPUT_ONLY_QASPER_FEASIBILITY',training_papers=len(papers),training_questions=len(seen_questions),eligible_papers=sum(v['eligible'] for v in meta),eligible_paper_groups=len(groups),eligible_questions=sum(len(v['questions']) for v in meta if v['eligible']),duplicate_canonical_paper_groups=sum(n>1 for n in counts.values()),longbench_transfer_groups_before=len(candidates),title_overlap_groups_excluded=len(excluded),longbench_transfer_groups_remaining=len(retained),feasible=feasible,metadata_sha256=metasha,vault_sha256=vaultsha,overlap_sha256=oversha,reserved_transfer_groups_sha256=retsha,raw_train_sha256=receipt['train_sha256'],answers_displayed=False,answer_values_used_for_selection=False,new_model_calls=0,new_GPU_hours=0,cpu_wall_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());save('INPUT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
