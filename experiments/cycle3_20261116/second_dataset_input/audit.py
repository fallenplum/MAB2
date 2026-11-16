"""Input-only eligibility audit; never projects answer/evidence/support columns."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,sys,time
import pyarrow.parquet as pq
import numpy as np
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');r=root/'worktrees/cycle3-credit-transport-20261116';p=root/'artifacts/private/cycle3_20261116'
sys.path.insert(0,str(r/'experiments/cycle2_20261113/hotpot_selection_v1'));import select_hotpot as h
source=root/'data/2wiki_author_train_v1';desc=json.loads((source/'SOURCE_DESCRIPTOR.json').read_bytes());file=source/'train.parquet'
assert hashlib.sha256(file.read_bytes()).hexdigest()==desc['sha256']
old=json.loads((p/'EXPOSURES_UNION_AFTER_FORMAL6_SELECTION.json').read_bytes());oldgroups=set(old['question_group_hashes'])
out=p/'2wiki_input_audit_v1';out.mkdir(parents=True,exist_ok=False)
seenids=set();groups=set();counts=Counter();formats=Counter();length=[];overlap=0;bad=0;rows=0
for batch in pq.ParquetFile(file).iter_batches(batch_size=128,columns=['_id','question','context']):
    for row in batch.to_pylist():
        rows+=1;assert row['_id'] not in seenids;seenids.add(row['_id'])
        try:context=json.loads(row['context']);formats['json']+=1
        except json.JSONDecodeError:context=ast.literal_eval(row['context']);formats['literal']+=1
        assert isinstance(context,list) and all(isinstance(x,list) and len(x)==2 and isinstance(x[0],str) and isinstance(x[1],list) and all(isinstance(s,str) for s in x[1]) for x in context)
        g=h.question_group_sha256(row['question']);groups.add(g);overlap+=int(g in oldgroups)
        counts[len(context)]+=1;length.append(sum(len(t)+sum(map(len,s)) for t,s in context))
receipt=dict(status='INPUT_ONLY_AUDIT_COMPLETE',dataset='xanhho/2WikiMultihopQA',revision=desc['revision'],source_sha256=desc['sha256'],rows=rows,unique_ids=len(seenids),unique_question_groups=len(groups),context_document_count_histogram=dict(counts),formats=dict(formats),context_character_quantiles=dict(zip(['min','p50','p90','max'],map(float,np.quantile(length,[0,.5,.9,1])))),exact_question_overlap_with_known_MAB2_exposures=overlap,projected_columns=['_id','question','context'],labels_projected=False,dev_test_downloaded=False,unknown_cross_project_exposure='UNKNOWN',semantic_paraphrase_overlap='NOT_ASSESSED',new_GPU_hours=0,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'QUESTION_GROUP_HASHES.json').write_text(json.dumps(sorted(groups))+'\n');print(json.dumps(receipt))
