from pathlib import Path
import json
root=Path('experiments/e0_real_workflow_20261126')
full=json.loads((root/'results/E0_FULL_32_SUMMARY.json').read_text(encoding='utf-8'))
strict=json.loads((root/'results/E0_STRICT_GATE_SUMMARY.json').read_text(encoding='utf-8'))
replay=json.loads((root/'results/E0_BASELINE_REPLAY.json').read_text(encoding='utf-8'))
result={'schema':'mab2_e0_baseline_comparison_v1','status':'DEVELOPMENT_FULL_WITH_STRICT_GATE','methods':{
 'new_cold_actual':{'events':full['new_cold']['events'],'input_tokens':full['new_cold']['input_tokens'],'posthoc_accuracy':full['posthoc']['new_cold']['accuracy'],'decision_errors_vs_human':7},
 'new_warm_actual':{'events':full['new_warm']['events'],'cache_hits':full['new_warm']['cache_hits'],'input_tokens':full['new_warm']['input_tokens'],'posthoc_accuracy':full['posthoc']['new_warm']['accuracy'],'decision_errors_vs_human':7},
 'strict_gate_cold_actual':strict['cold'],
 'strict_gate_canonical_actual':strict['canonical'],
 'canonical_full_replay':{'input_tokens':sum(x['costs']['canonical_full_input_tokens_replay'] for x in replay['rows'])},
 'stale_replay':{'decision_errors':'not yet scored','cost_tokens':0}},
 'interpretation':'Strict gates reduce calls sharply but introduce more action errors than full cold on these development groups. Canonical strict gate is a replay-compatible implementation, but the comparison remains development-only.',
 'next':'Run strong flat/RAG and stale quality scoring under the same target before any claim.'}
(root/'results/E0_BASELINE_COMPARISON_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

