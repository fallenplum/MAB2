"""Arithmetic reconciliation of authorized occupancy ledger, not source re-audit."""
from pathlib import Path
import argparse,json,hashlib,math

def reconcile(b):
    assert b['cap_gpu_occupancy_hours']==160 and b['budget_reset_on_new_cycle'] is False
    closed=[];active=[];seen=set()
    for e in b['candidate_reservations']:
        name=e['candidate'];assert name not in seen;seen.add(name)
        limit=e['max_gpu_hours'];assert math.isfinite(limit) and limit>=0
        if e.get('cost_closed'):
            cost=e['used_gpu_hours_estimate'];assert math.isfinite(cost) and cost>=0 and e.get('released')
            closed.append(dict(candidate=name,actual_GPU_hours=cost,reservation_GPU_hours=limit,status=e.get('phase'),qualification_only=e.get('qualification_only')))
        else:
            assert not e.get('released');active.append(dict(candidate=name,reserved_GPU_hours=limit,actual_cost_pending=True,status=e.get('phase')))
    used=sum(e['actual_GPU_hours'] for e in closed);reserved=sum(e['reserved_GPU_hours'] for e in active)
    assert math.isclose(used,b['used_gpu_occupancy_hours'],rel_tol=0,abs_tol=1e-9),'Closed costs do not reconcile'
    assert math.isclose(reserved,b['reserved_gpu_occupancy_hours'],rel_tol=0,abs_tol=1e-9),'Active reservations do not reconcile'
    assert used+reserved<=160+1e-9,'Global cap exceeded'
    cycle2=sum(e['actual_GPU_hours'] for e in closed if e['candidate'].startswith('C2_'))
    assert math.isclose(cycle2,b['final_audit_gpu_hours'],rel_tol=0,abs_tol=1e-9),'Prior closed cycle changed'
    engineering=sum(e['actual_GPU_hours'] for e in closed if e['candidate'].startswith('C3_ANNOTATION_ENGINEERING'))
    formal7=sum(e['actual_GPU_hours'] for e in closed if e['candidate'].startswith('C3_FORMAL7_'))
    formal7_reserve=sum(e['reserved_GPU_hours'] for e in active if e['candidate'].startswith('C3_FORMAL7_'))
    engineering_reserve=sum(e['reserved_GPU_hours'] for e in active if e['candidate'].startswith('C3_ANNOTATION_ENGINEERING'))
    assert engineering+engineering_reserve<=.5+1e-9 and formal7+formal7_reserve<=8+1e-9
    formal8=sum(e['actual_GPU_hours'] for e in closed if e['candidate'].startswith('C3_FORMAL8_'))
    formal8_reserve=sum(e['reserved_GPU_hours'] for e in active if e['candidate'].startswith('C3_FORMAL8_'))
    assert formal8+formal8_reserve<=8+1e-9,'Formal8 all-attempt cap exceeded'
    return dict(formal8_all_attempts_actual_GPU_hours=formal8,formal8_active_reserved_GPU_hours=formal8_reserve,formal8_all_attempts_cap_GPU_hours=8,status='PASS_ARITHMETIC_RECONCILIATION_ACTUAL_COST_PENDING' if active else 'PASS_ARITHMETIC_RECONCILIATION_CLOSED',cap_GPU_hours=160,closed_actual_GPU_hours=used,active_reserved_GPU_hours=reserved,unreserved_remaining_GPU_hours=160-used-reserved,prior_cycle_closed_GPU_hours=cycle2,formal7_engineering_actual_GPU_hours=engineering,formal7_formal_actual_closed_GPU_hours=formal7,formal7_active_reserved_GPU_hours=formal7_reserve,formal_candidates_admitted=b['formal_candidate_rounds_admitted'],formal_candidates_started=b['formal_candidate_rounds_started'],candidate_retry_not_new_scientific_round=True,old_five_round_cap_scope=b['formal_round_cap_scope'],closed_entries=closed,active_entries=active,live_running_estimates_charged=False,source_archives_reaudited_here=False,limitation='Reconciles ledger arithmetic and unchanged prior-cycle aggregate; archive-level cost evidence remains in each closure receipt. Active reservations are not actual spend.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ledger',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.ledger);raw=source.read_bytes();b=json.loads(raw);result=reconcile(b);result['ledger_sha256']=hashlib.sha256(raw).hexdigest();out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['closed_entries','active_entries','limitation','old_five_round_cap_scope']}))
