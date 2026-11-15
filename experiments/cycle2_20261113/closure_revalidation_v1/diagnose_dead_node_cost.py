"""Descriptive cost-only counterfactual on exposed formal4; no new generations."""
from pathlib import Path
import argparse,json,hashlib,sys
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists();manifest=json.loads((a.archive/'ARCHIVE_MANIFEST.json').read_bytes())
 assert hashlib.sha256((a.archive/'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()=='6873e0b42c8170c861151d007fc3150450552de61ed78e0c1409a2e41bc81338'
 rows=[]
 for f in sorted(a.archive.glob('worker_*/unit_*/MATRIX.json')):
  assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[str(f.relative_to(a.archive)).replace('\\','/')]
  m=json.loads(f.read_bytes());shock=m['removed_upstream']
  for c in m['conditions']:
   if c['condition']!='G1_MINUS_E'+shock+'__Q_FULL':continue
   ancestors={'M'}
   while True:
    nex=ancestors|{u for u,v in c['edges'] if v in ancestors}
    if nex==ancestors:break
    ancestors=nex
   assert set(c['nodes'])-ancestors=={shock}
   assert all(note['source']!=shock for n,r in c['nodes'].items() if n!=shock for note in json.loads(r['request']['messages'][1]['content'])['evidence_notes'])
   vals=lambda field:sum(v['generation'][field] for v in c['nodes'].values())
   dead=c['nodes'][shock]['generation']
   rows.append({'total_tokens':vals('input_tokens')+vals('output_tokens'),'dead_tokens':dead['input_tokens']+dead['output_tokens'],
    'total_kernel':vals('gpu_seconds'),'dead_kernel':dead['gpu_seconds']})
 assert len(rows)==192
 totals={k:sum(r[k] for r in rows) for k in rows[0]}
 result={'status':'COMPLETE_COST_ONLY_EXPOSED_DIAGNOSTIC','observations':192,'questions':96,'generation_seeds':2,
  'condition':'G1 minus affected direct edge; Q_FULL receiver','dead_nodes_per_condition':1,'original_calls_per_condition':4,'compiled_calls_per_condition':3,
  'cost_sums':totals,'potential_token_fraction_saved':totals['dead_tokens']/totals['total_tokens'],'potential_kernel_fraction_saved':totals['dead_kernel']/totals['total_kernel'],
  'no_claim_of_measured_compiled_latency':True,'new_GPU_hours':0,'new_model_calls':0,'labels_read':False,'quality_improvement_from_compiler':False,
  'interpretation':'Request/graph-level redundant call; exact output preservation requires node-seed and side-effect-free backend assumptions. Same compiler must serve every policy. Does not rescue formal4.',
  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'archive_manifest_sha256':'6873e0b42c8170c861151d007fc3150450552de61ed78e0c1409a2e41bc81338'}
 a.output.mkdir(parents=True);(a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
