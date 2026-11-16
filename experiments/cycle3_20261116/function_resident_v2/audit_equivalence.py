"""Exact artificial cross-profile equivalence; timings are not quality evidence."""
from pathlib import Path
import json,hashlib,argparse
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(old,new):
 old=Path(old);run=Path(new);archive=run/"archive";newcost=read(run/"FULL_COST_SUMMARY.json");cost=newcost["actual_GPU_residence_hours"]
 assert read(archive/'worker_0/REPLAY.json')['status']=='PASS_EXACT_SEEDED_REPLAY';checks=0
 for model in ['qwen','mistral']:
  a=read(old/('archive/worker_0/LOAD_'+model+'.json'))['identity'];z=read(archive/('worker_0/LOAD_'+model+'.json'))['identity'];assert a.keys()==z.keys();assert a['allocator_cap_bytes']==20*1024**3 and z['allocator_cap_bytes']==26*1024**3
  for key in a:
   if key not in ['backend_code_sha256','allocator_cap_bytes']:assert a[key]==z[key];checks+=1
 for i in range(3):
  a=read(old/f'archive/worker_0/unit_{i:03d}/MATRIX.json');z=read(archive/f'worker_0/unit_{i:03d}/MATRIX.json');assert a['question_fingerprint']==z['question_fingerprint'];assert a['protocol'].keys()==z['protocol'].keys()
  for key in a['protocol']:
   if key!='model_identities':assert a['protocol'][key]==z['protocol'][key];checks+=1
  assert len(a['request_trace'])==len(z['request_trace'])
  for x,y in zip(a['request_trace'],z['request_trace']):
   assert x['request'].keys()==y['request'].keys()
   for key in x['request']:
    if key not in ['model_identity_sha256','protocol_sha256']:assert x['request'][key]==y['request'][key];checks+=1
   for key in ['text','input_tokens','output_tokens','finish_reason']:assert x['generation'][key]==y['generation'][key];checks+=1
  assert len(a['conditions'])==len(z['conditions'])==30
  for x,y in zip(a['conditions'],z['conditions']):
   for key in ['key','condition','execution_seed','edges','executed_nodes','selected_program','answer','cache_reuse_count']:assert x[key]==y[key];checks+=1
   assert len(x['programs'])==len(y['programs'])
   for u,v in zip(x['programs'],y['programs']):
    for key in ['code','program_sha256','execution_reused']:assert u[key]==v[key];checks+=1
    assert len(u['visible']['cases'])==len(v['visible']['cases'])==2
    for uu,vv in zip(u['visible']['cases'],v['visible']['cases']):
     for key in ['status','passed','returned','typed_return']:assert uu.get(key)==vv.get(key);checks+=1
 oldcost=read(old/'FULL_COST_SUMMARY.json');comparison=dict(status='PASS_EXACT_CROSS_PROFILE_ARTIFICIAL_EQUIVALENCE',scalar_checks=checks,old_actual_GPU_hours=oldcost['actual_GPU_residence_hours'],new_actual_GPU_hours=cost,old_placement_seconds=oldcost['model_placement_wall_seconds_sum'],new_placement_seconds=newcost['model_placement_wall_seconds_sum'],old_physical_tokens=oldcost['physical_tokens'],new_physical_tokens=newcost['physical_tokens'],old_archive_sha256=sha(old/'archive/ARCHIVE_MANIFEST.json'),new_archive_sha256=sha(archive/'ARCHIVE_MANIFEST.json'),scientific_samples=0,source_sha256=sha(__file__),limitation='Nonrandomized concurrent-load timing; exact artificial outputs establish execution equivalence only, not general throughput or scientific effectiveness. Current formal pilot unchanged.')
 return comparison
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--out',required=True);a=p.parse_args();result=compare(a.old,a.new)
 with open(a.out,'x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
