from pathlib import Path
import json
root=Path('experiments/e0_real_workflow_20261126')
run=json.loads((root/'results/E0_SMOKE_4_RERUN_001.json').read_text(encoding='utf-8'))
post=json.loads((root/'results/E0_SMOKE_4_RERUN_001_POSTHOC.json').read_text(encoding='utf-8'))
def summ(name):
 ev=run[name]
 return {'events':len(ev),'cache_hits':sum(e['cache_hit'] for e in ev),'input_tokens':sum(e['input_tokens'] for e in ev),'output_tokens':sum(e['output_tokens'] for e in ev),'latency_seconds':sum(e['latency_seconds'] for e in ev),'unresolved':sum(e['label'] is None for e in ev)}
summary={'schema':'mab2_e0_smoke_summary_v1','status':'DEVELOPMENT_ONLY','run':'E0_SMOKE_4_RERUN_001','invalid_prior_run':'E0_SMOKE_4','groups':4,'model':'Qwen2.5-3B-Instruct','knowledge_source':'enwiki-20230401.db','old':summ('old_events'),'new_cold':summ('new_cold_events'),'new_warm':summ('new_warm_events'),'posthoc':post,'decisions':{'old':run['old_decisions'],'new_cold':run['new_cold_decisions'],'new_warm':run['new_warm_decisions']},'warning':'Four-group development smoke only. Not full E0, not independent confirmation, and not an ACL claim.'}
(root/'results/E0_SMOKE_4_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

