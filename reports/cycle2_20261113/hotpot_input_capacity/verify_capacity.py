"""Independent recomputation from private indices/counts; no source question or label access."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description='Verify private derived count outputs; no benchmark text or labels read')
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
OUT=args.output.expanduser().resolve(strict=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(x):
    x=np.asarray(x,dtype=np.int64)
    return {'n':len(x),'min':int(min(x)),'max':int(max(x)),'mean':float(sum(x)/len(x)),'quantiles':{str(q):float(np.quantile(x,q)) for q in [0,.5,.9,.95,.99,1]}}
def check(x,reported):
    assert dist(x)==reported['tokens']
    assert dist(8192-np.asarray(x))==reported['headroom_to_8192']
    for r in reported['hypothetical_extra_token_reserves']:
        over=int(np.count_nonzero(np.asarray(x)+r['extra_tokens']>8192))
        assert over==r['over_8192'] and len(x)-over==r['within_8192']

protocol=json.loads((HERE/'PROTOCOL.json').read_text('utf-8'))
r=json.loads((OUT/'RESULTS.json').read_text())
selection=json.loads((OUT/'private/SELECTION.json').read_text())
manifest=[json.loads(line) for line in (OUT/'private/SAMPLE_MANIFEST.jsonl').read_text().splitlines()]
rows=[json.loads(line) for line in (OUT/'private/TOKEN_COUNTS.jsonl').read_text().splitlines()]
expected=np.sort(np.random.Generator(np.random.PCG64(protocol['sample']['seed'])).choice(90447,2000,replace=False)).tolist()
assert expected==selection['global_row_indices']==[x['global_row'] for x in manifest]==[x['global_row'] for x in rows]
assert len(set(expected))==2000
assert dict(Counter(x['type'] for x in manifest))==r['sample_types']
assert dict(Counter(x['level'] for x in manifest))==r['sample_levels']
assert all(x['payload_sha256']==y['payload_sha256'] for x,y in zip(manifest,rows))
assert sha(OUT/'private/SAMPLE_MANIFEST.jsonl')==r['private_sample_manifest_sha256']
assert sha(OUT/'private/TOKEN_COUNTS.jsonl')==r['private_token_counts_sha256']
for model in ('qwen','mistral'):
    d=r['models'][model]
    payload=[row[model]['payload_tokens'] for row in rows]
    workers=np.asarray([row[model]['round1_worker_tokens'] for row in rows])
    agg=np.asarray([row[model]['aggregate_empty_wire_tokens'] for row in rows])
    assert dist(payload)==d['payload_tokens']
    for w in range(4):check(workers[:,w],d['round1_workers'][f'w{w}'])
    check(workers.max(axis=1),d['round1_max_across_workers'])
    check(agg,d['aggregate_empty_wire_placeholder'])
    check(np.maximum(workers.max(axis=1),agg),d['max_base_across_round1_and_placeholder'])
comparison=r['models']['mistral']['local_auto_vs_explicit_equivalence']
assert comparison['payload_token_id_matches']==2000
assert comparison['same_backend_serialization'] and comparison['same_chat_template']
assert r['engineering_graph_config']['rounds']==3
assert not any(r[k] for k in ['answer_values_read','supporting_facts_values_read','validation_or_test_read','R1_results_read','weights_loaded','new_model_calls','new_gpu_hours','new_formal_rounds','context_truncated_or_filtered','cuda_initialized'])
receipt={'status':'PASS_DERIVED_COUNT_RECOMPUTATION','sample_size':2000,'both_models_verified':True,'seed_selection_reproduced':True,'all_reported_distributions_and_reserve_counts_exact':True,'mistral_local_loader_payload_ids_equal_rows':2000,'server_equivalence_still_not_established':True,'raw_data_read_by_verifier':False,'new_gpu_hours':0,'new_model_calls':0,'result_sha256':sha(OUT/'RESULTS.json'),'verifier_sha256':sha(Path(__file__))}
with (OUT/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
