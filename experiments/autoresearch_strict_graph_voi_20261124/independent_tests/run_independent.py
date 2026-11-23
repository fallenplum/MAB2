from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent
files=['test_contract_properties.py','test_certificate_exhaustive.py','test_information_ledger.py','test_cycle_005_h2.py','test_cycle_006_impl.py']
records=[]
for name in files:
    r=subprocess.run([sys.executable,str(root/name)],capture_output=True,text=True)
    records.append({'test':name,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps({'schema':'p3_independent_test_run_v1','tests':records},indent=2))
raise SystemExit(0 if all(x['returncode']==0 for x in records) else 1)



