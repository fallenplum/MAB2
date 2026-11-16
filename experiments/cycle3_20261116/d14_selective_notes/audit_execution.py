"""Independent terminal execution audit. No benchmark labels or quality scoring."""
import argparse,hashlib,json,math
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def audit(archive,plan,requests,expected=None):
    archive=Path(archive);p=read(plan);closure=read(archive/'CLOSURE.json');manifest=read(archive/'ARCHIVE_MANIFEST.json')
    assert closure['status']=='COMPLETE_UNSCORED' and closure['all_workers_exited'] and closure['plan_sha256']==sha(plan)
    actual={str(f.relative_to(archive)).replace('\\','/'):sha(f) for f in archive.rglob('*') if f.is_file() and f.name!='ARCHIVE_MANIFEST.json'}
    assert manifest==actual,'Incomplete or modified archive'
    assert sha(requests)==p['requests_sha256']
    jobs={j['job_id']:j for j in map(json.loads,Path(requests).read_bytes().splitlines())};assert len(jobs)==p['expected_requests']
    seen={};calls=set();peak=0;tokens=0
    for wid in range(len(p['gpus'])):
        w=archive/f'worker_{wid}';assert read(w/'COMPLETE.json')['requests']==len(p['assignments'][str(wid)])
        load=read(w/'LOAD.json');assert digest(load['identity'])==p['model_identity_sha256']
        journal=(w/'REQUEST_JOURNAL.jsonl').read_bytes();assert journal.endswith(b'\n');rows=list(map(json.loads,journal.splitlines()))
        assert rows[0]['event']=='HEADER' and digest(rows[0]['backend_identity'])==p['model_identity_sha256']
        assert len(rows)==1+2*len(p['assignments'][str(wid)])
        results=list(map(json.loads,(w/'RESULTS.jsonl').read_bytes().splitlines()));assert len(results)==len(p['assignments'][str(wid)])
        for n,result in enumerate(results):
            key=result['job_id'];assert key not in seen and key in jobs
            start,end=rows[1+2*n:3+2*n];req=jobs[key]['request'];g=result['generation']
            assert start['event']=='START' and end['event']=='RETURNED'
            assert all(end.get(k)==v for k,v in start.items() if k!='event')
            assert start['context']['job_id']==key and start['backend_identity_sha256']==p['model_identity_sha256']
            for field in ['messages','seed','max_new_tokens','config']:assert start[field]==req[field]
            assert end['generation']==g and len(end['native_events'])==1
            native=end['native_events'][0];assert native['status']=='SUCCESS' and native['call_id']==g['call_id'] and g['call_id'] not in calls
            for field in ['input_tokens','output_tokens']:assert native[field]==g[field]
            assert 0<=g['output_tokens']<=256 and g['input_tokens']==result['capacity']['input_tokens']
            calls.add(g['call_id']);seen[key]=g;tokens+=g['input_tokens']+g['output_tokens'];peak=max(peak,g['gpu_peak_allocated_bytes'])
    assert set(seen)==set(jobs)
    assert math.isclose(sum(closure['residence_seconds_by_worker'])/3600,closure['actual_GPU_hours'],abs_tol=1e-12)
    matches=0
    if expected:
        exp=read(expected);assert set(exp)==set(seen)
        for k,g in seen.items():
            assert all(g[f]==v for f,v in exp[k].items()),'Qualification scientific output mismatch'
            matches+=1
    return dict(status='PASS_COMPLETE_TERMINAL_EXECUTION',requests=len(seen),actual_GPU_hours=closure['actual_GPU_hours'],physical_tokens=tokens,peak_allocated_bytes=peak,exact_qualification_matches=matches,archive_sha256=sha(archive/'ARCHIVE_MANIFEST.json'),plan_sha256=sha(plan),quality_scored=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--plan',required=True);ap.add_argument('--requests',required=True);ap.add_argument('--expected');ap.add_argument('--output',required=True);a=ap.parse_args();result=audit(a.archive,a.plan,a.requests,a.expected)
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
