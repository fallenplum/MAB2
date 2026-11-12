"""Read accepted historical cost summaries; never equate CUDA events to residency."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous cost extraction')
    rows={};pins={}
    for stage,name in [('G1','INDEPENDENT_G1_ACCEPTANCE.json'),('G2','G2_INDEPENDENT_ACCEPTANCE.json')]:
        path=a.root/'artifacts/audit'/name;raw=path.read_bytes();x=json.loads(raw);pins[name]=hashlib.sha256(raw).hexdigest();c=x['costs']
        if stage=='G1':
            rows[stage]={'canonical_model_calls':c['canonical_g1'],'engineering_summary':c['engineering']['summary'],
                         'engineering_scope':c['engineering']['scope'],'engineering_missing_ledgers':c['engineering']['missing_ledgers'],
                         'migration_unrecorded_inflight':c['migration_unrecorded_inflight'],
                         'orphaned_success_calls_already_in_canonical':c['orphaned_success_calls']}
        else:
            rows[stage]={'formal_model_calls':c['formal'],'qualification_model_calls':c['qualification'],
                         'complete_cpu_attempt_seconds':c['complete_cpu_attempt_seconds'],
                         'independent_audit_cpu_seconds':x['audit_cpu_seconds'],
                         'preprocessing_cpu_seconds':c['preprocessing']['cpu_seconds'],
                         'preprocessing_cpu_status':c['preprocessing']['cpu_status'],
                         'observed_preprocessing_launch_to_receipt_wall_interval_seconds':c['preprocessing']['observed_launch_to_feature_receipt_wall_interval_seconds'],
                         'unrecorded_inflight_call_upper_bound':c['unrecorded_inflight_call_upper_bound']}
        rows[stage]['source_status']=x['status']
        rows[stage]['full_gpu_occupancy_hours']=None
        rows[stage]['occupancy_status']='NOT_PRESENT_IN_ACCEPTED_SUMMARY; CUDA-event time is not a substitute'
    interrupted_path=a.root/'artifacts/audit/v1_interrupted_costs.json'
    interrupted=json.loads(interrupted_path.read_bytes());pins[interrupted_path.name]=hashlib.sha256(interrupted_path.read_bytes()).hexdigest()
    report={'schema':'mab2_historical_cost_provenance_v1','evidence':'accepted historical summaries, not a fresh per-call re-audit',
        'source_sha256':pins,'stages':rows,'entire_project_cost_complete':False,
        'missing_or_nonidentifiable':['G1 migration in-flight call costs','Earlier v1/v2 engineering coverage beyond their retained trial receipt','G2 initial feature-preparation CPU duration','Historical full GPU load-to-exit occupancy absent from these accepted summaries','Some audit/engineering/human/agent CPU costs'],
        'earlier_trial_receipt_present':True,'earlier_trial_receipt_top_level_keys':list(interrupted),
        'budget_boundary':'These historical G1/G2 jobs precede the authorized new autonomous 160 GPU-hour cap. They remain research costs but are not charged again to this cycle.',
        'do_not_sum_as_one_unit':'Historical CUDA-event seconds, historical per-call elapsed seconds and new GPU residency are distinct measures; retain separate columns.',
        'unknowns_imputed_as_zero':False,'duplicate_backup_aliases_added':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
