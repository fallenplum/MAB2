"""Frozen paper partition and input-only retrieval. No answer/evidence labels."""
import argparse,hashlib,json,math,re,time,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def h(text):return hashlib.sha256(text.encode()).hexdigest()
def chat_count(tokenizer,prompt):
    ids=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_dict=False)
    assert isinstance(ids,list) and all(type(v) is int for v in ids) and len(ids)>0
    return len(ids)
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def segment(text,tokenizer):
    offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)['offset_mapping']
    cuts=[0]+sorted({offsets[i][0] for i in range(256,len(offsets),256) if 0<offsets[i][0]<len(text)})+[len(text)]
    values=[{'id':f'P{i:04d}','text':text[a:b]} for i,(a,b) in enumerate(zip(cuts,cuts[1:]))]
    assert ''.join(v['text'] for v in values)==text
    return values
def bm25(question,segments):
    tok=lambda s:re.findall(r'\w+',s.lower());q=tok(question);docs=[Counter(tok(x['text'])) for x in segments];length=[sum(x.values()) for x in docs];avg=sum(length)/len(length);df=Counter(t for v in docs for t in v)
    score=[sum(math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*v[t]*2.5/(v[t]+1.5*(.25+.75*length[i]/avg)) for t in q if v[t]) for i,v in enumerate(docs)]
    return [segments[i]['id'] for i in sorted(range(len(docs)),key=lambda i:(-score[i],i))[:18]]

def main():
    a=argparse.ArgumentParser();a.add_argument('--repo',required=True);a.add_argument('--private-root',required=True);a.add_argument('--qwen',required=True);a.add_argument('--mistral',required=True);a.add_argument('--encoder',required=True);args=a.parse_args();r=Path(args.repo);p=Path(args.private_root);run=p/'longdoc_formal9_v1';t=time.perf_counter();read=lambda f:json.loads(f.read_bytes());torch.set_num_threads(2);torch.set_num_interop_threads(2)
    sys.path.insert(0,str(r/'experiments/cycle3_20261116/d21_qasper_input_audit'));from audit_inputs import canonical_document
    sys.path.insert(0,str(r/'experiments/cycle2_20261113/exogenous_paths_v1'));sys.path.insert(0,str(r/'experiments/cycle3_20261116/longdoc_v1'));from longdoc_operator import messages,validate
    old=p/'qasper_input_audit_v1';audit=read(old/'INPUT_AUDIT.json');assert sha(old/'PRIVATE_INPUT_METADATA.json')==audit['metadata_sha256'];meta=read(old/'PRIVATE_INPUT_METADATA.json');receipt=read(old/'DOWNLOAD_RECEIPT.json');assert sha(receipt['train_path'])==audit['raw_train_sha256'];papers=read(Path(receipt['train_path']))
    eligible=[v for v in meta if v['eligible']];eligible.sort(key=lambda v:(h('MAB2-F9-paper-partition-v1|'+v['group_sha256']),v['group_sha256']));assert len(eligible)==109 and len({v['group_sha256'] for v in eligible})==109
    partition=[]
    for i,m in enumerate(eligible):
        qid=min(m['questions'],key=lambda v:(h('MAB2-F9-question-v1|'+v['id']),v['id']))['id'];partition.append(dict(paper_id=m['paper_id'],group_sha256=m['group_sha256'],question_id=qid,partition='pilot' if i<16 else 'development' if i<45 else 'confirmation'))
    write(run/'PRIVATE_PARTITION.json',partition)
    qt=AutoTokenizer.from_pretrained(args.qwen,local_files_only=True);mt=AutoTokenizer.from_pretrained(args.mistral,local_files_only=True);et=AutoTokenizer.from_pretrained(args.encoder,local_files_only=True);em=AutoModel.from_pretrained(args.encoder,local_files_only=True).cpu().eval();truncated=0;maxsegment=0;rows=[];selection=[];caps=[]
    for i,part in enumerate(partition[:16]):
        paper=papers[part['paper_id']];question=next(x['question'] for x in paper['qas'] if x['question_id']==part['question_id']);context=canonical_document(paper);segments=segment(context,qt);maxsegment=max(maxsegment,max(len(qt.encode(x['text'],add_special_tokens=False)) for x in segments));texts=[question]+[x['text'] for x in segments];vectors=[]
        with torch.inference_mode():
            for at in range(0,len(texts),16):
                batch=texts[at:at+16];truncated+=sum(len(v)>512 for v in et(batch,truncation=False)['input_ids']);enc=et(batch,padding=True,truncation=True,max_length=512,return_tensors='pt');hidden=em(**enc).last_hidden_state;mask=enc['attention_mask'].unsqueeze(-1);v=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1);vectors.append(torch.nn.functional.normalize(v,p=2,dim=1).numpy())
        v=np.concatenate(vectors);scores=v[1:]@v[0];dense=[segments[j]['id'] for j in sorted(range(len(segments)),key=lambda j:(-float(scores[j]),j))[:18]];payload=validate(dict(id=part['question_id'],question=question,segments=segments,retrieval={'BM25':bm25(question,segments),'DENSE':dense}));raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n';rows.append(raw);selection.append(dict(index=i,input_sha256=h(raw),removed_upstream='U' if i%2==0 else 'V',paper_group_sha256=part['group_sha256']))
        full=messages(payload,[x['id'] for x in segments],True)
        for name,tokenizer in [('qwen',qt),('mistral',mt)]:
            count=chat_count(tokenizer,full);caps.append(dict(index=i,model=name,input_tokens=count,complete_text_preserved=True,accepted=count<=32000 and count+256<=32768))
    with (run/'PILOT_INPUTS.jsonl').open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(rows))
    write(run/'SELECTION.json',{'questions':selection});write(run/'PRIVATE_CAPACITY.json',caps)
    result=dict(status='PASS_INPUT_PREPARATION' if all(v['accepted'] for v in caps) else 'BLOCKED_INPUT_CAPACITY',pilot_papers=16,development_papers=29,confirmation_papers=64,formal_inputs=16,input_tokens_max_by_model={n:max(v['input_tokens'] for v in caps if v['model']==n) for n in ['qwen','mistral']},max_retokenized_segment_tokens=maxsegment,dense_texts_truncated_at512=truncated,all_text_preserved=True,answers_used=False,new_GPU_hours=0,cpu_main_wall_seconds=time.perf_counter()-t,partition_sha256=sha(run/'PRIVATE_PARTITION.json'),inputs_sha256=sha(run/'PILOT_INPUTS.jsonl'),selection_sha256=sha(run/'SELECTION.json'),capacity_sha256=sha(run/'PRIVATE_CAPACITY.json'),source_sha256=sha(__file__),encoder_pins={str(f.relative_to(Path(args.encoder))):sha(f) for f in Path(args.encoder).rglob('*') if f.is_file() and f.name!='README.md'})
    write(run/'INPUT_RECEIPT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='encoder_pins'}))
if __name__=='__main__':main()
