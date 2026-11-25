from pathlib import Path
import hashlib,json
root=Path('experiments/e0_real_workflow_20261126')
run=json.loads((root/'results/E0_FULL_32.json').read_text(encoding='utf-8'))
post=json.loads((root/'results/E0_FULL_32_POSTHOC.json').read_text(encoding='utf-8'))

def summ(name):
 ev=run[name]
 return {'events':len(ev),'cache_hits':sum(e['cache_hit'] for e in ev),'input_tokens':sum(e['input_tokens'] for e in ev),'output_tokens':sum(e['output_tokens'] for e in ev),'latency_seconds':sum(e['latency_seconds'] for e in ev),'unresolved':sum(e['label'] is None for e in ev)}
valid=[]; invalid=[]; changed=[]
for gid,old in run['old_decisions'].items():
 new=run['new_cold_decisions'][gid]
 if old['status']=='INVALID' or new['status']=='INVALID': invalid.append(gid); continue
 valid.append(gid)
 if old['decision']!=new['decision']: changed.append({'group_id':gid,'old':old['decision'],'new':new['decision'],'old_status':old['status'],'new_status':new['status']})
old_s,new_s,warm_s=summ('old_events'),summ('new_cold_events'),summ('new_warm_events')
summary={'schema':'mab2_e0_full_summary_v1','status':'DEVELOPMENT_FULL_RUN_PARTIAL_BASELINES','groups_requested':32,'groups_valid_for_pairwise_decision':len(valid),'invalid_groups':invalid,'model':'Qwen2.5-3B-Instruct','knowledge_source_document_count':6187531,'old':old_s,'new_cold':new_s,'new_warm':warm_s,'warm_input_token_saving_fraction':1-warm_s['input_tokens']/new_s['input_tokens'],'warm_latency_saving_fraction':1-warm_s['latency_seconds']/new_s['latency_seconds'],'posthoc':post,'decision_changes':changed,'raw_cache_reuse_fraction':warm_s['cache_hits']/max(warm_s['events'],1),'caveats':['No strong flat/RAG baseline yet','No canonical cache rerun yet','No strict gate or stale action replay yet','All groups are development data, not independent confirmation','No confidence intervals per project-wide instruction'],'interpretation':'Observed raw cache reuse is limited. Continue with matched canonical/strict-gate/stale baselines before selecting a direction.'}
(root/'results/E0_FULL_32_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'valid':len(valid),'invalid':invalid,'raw_cache_reuse_fraction':summary['raw_cache_reuse_fraction'],'token_saving':summary['warm_input_token_saving_fraction'],'decision_changes':len(changed)},ensure_ascii=False))


