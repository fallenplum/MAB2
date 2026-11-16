"""Explain input-version differences using aggregate component hashes only."""
from pathlib import Path
from collections import Counter
import ast,json,hashlib
import pyarrow.parquet as pq
from audit_updated_train import records,input_projection,digest,file_sha

def signatures(v):
    ctx=v['context']
    clean=lambda x:' '.join(x.split())
    return dict(question_raw=digest(v['question']),context_raw=digest(ctx),
        ordered_titles=digest([x[0] for x in ctx]),
        context_joined_sentences=digest([[t,' '.join(s)] for t,s in ctx]),
        context_whitespace_normalized=digest([[clean(t),clean(' '.join(s))] for t,s in ctx]),
        context_whitespace_normalized_unordered=digest(sorted([[clean(t),clean(' '.join(s))] for t,s in ctx])),
        sentence_counts=digest([len(s) for t,s in ctx]))

root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');p=root/'artifacts/private/cycle3_20261116';old={}
for batch in pq.ParquetFile(root/'data/2wiki_author_train_v1/train.parquet').iter_batches(batch_size=128,columns=['_id','question','context']):
    for row in batch.to_pylist():
        try:row['context']=json.loads(row['context'])
        except json.JSONDecodeError:row['context']=ast.literal_eval(row['context'])
        v=input_projection(row);assert v['_id'] not in old;old[v['_id']]=signatures(v)
equal=Counter();n=0
with (root/'data/2wiki_author_ids_april7_v1/train.json').open(encoding='utf-8') as stream:
    for row in records(stream):
        v=input_projection(row);row=None;new=signatures(v);prior=old[v['_id']];n+=1
        for k in new:equal[k]+=int(new[k]==prior[k])
assert n==len(old)==167454
receipt=dict(status='INPUT_VERSION_COMPONENT_COMPARISON_COMPLETE',records=n,equal_records_by_component=dict(equal),
    changed_records_by_component={k:n-v for k,v in equal.items()},supervision_used=False,
    JSON_record_parser_transiently_decodes_discarded_label_values=True,content_emitted=False,new_GPU_calls=0,
    source_sha256=file_sha(Path(__file__)))
out=p/'2wiki_updated_input_audit_v1/VERSION_COMPONENT_COMPARISON.json';assert not out.exists()
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
