"""Local data owner: verified model transfer, remote launch, incremental result mirror.

Password comes only from GRAPH_CREDIT_SSH_PASSWORD and is never serialized.
Host/user/project paths are CLI arguments and must not be committed as private config.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import paramiko


def atomic(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--user', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--remote-python', default='python3')
    p.add_argument('--known-hosts', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    status = root/'COORDINATOR_STATUS.json'
    state = {'pid':os.getpid(), 'schema':'graph_credit_local_coordinator_v1','scientific_completion':False}
    def update(stage, **kw):
        state.update(stage=stage, updated_at_unix=time.time(), **kw)
        atomic(status,state)
        print(json.dumps({'stage':stage, **kw}),flush=True)
    def connect():
        c = paramiko.SSHClient()
        c.load_host_keys(str(args.known_hosts))
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        c.connect(args.host, port=args.port, username=args.user, password=os.environ['GRAPH_CREDIT_SSH_PASSWORD'],
                  look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
        c.get_transport().set_keepalive(20)
        return c
    def execute(client, code, timeout=60):
        command = shlex.quote(args.remote_python) + ' -c ' + shlex.quote(code)
        _,stdout,stderr = client.exec_command(command,timeout=timeout)
        result = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        rc = stdout.channel.recv_exit_status()
        if rc:
            raise RuntimeError(f'Remote operation failed code {rc}: '+error[-1000:])
        return result
    def mkdirs(sftp, path):
        parts = path.split('/')
        current = ''
        for part in parts:
            if not part:
                continue
            current += '/'+part
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)
    try:
        while True:
            model_status = root/'artifacts/audit/model_download_status.json'
            if model_status.exists():
                m = json.loads(model_status.read_text())
                if m.get('status') == 'PASS':
                    break
                if m.get('status') == 'FAILED':
                    raise RuntimeError('Local model download failed; no remote inference launched')
            update('WAITING_VERIFIED_LOCAL_MODEL')
            time.sleep(20)
        client = connect()
        sftp = client.open_sftp()
        remote = args.remote_root.rstrip('/')
        update('UPLOADING_CODE_AND_DEVELOPMENT_DATA')
        candidates = []
        for part in ['src','scripts','configs']:
            candidates += [f for f in (root/part).rglob('*') if f.is_file() and '__pycache__' not in f.parts]
        for name in ['qualification_questions.jsonl','motivation_questions.jsonl','motivation_labels.jsonl']:
            candidates.append(root/'data/derived'/name)
        candidates += [root/'artifacts/audit/model_source_manifest.json',root/'artifacts/audit/model_download_status.json']
        deployment = {}
        for local in candidates:
            relative = local.relative_to(root).as_posix()
            deployment[relative] = hashlib.sha256(local.read_bytes()).hexdigest()
            target = remote+'/'+relative
            mkdirs(sftp,target.rsplit('/',1)[0])
            sftp.put(str(local),target)
        atomic(root/'artifacts/audit/deployment_manifest.json', deployment)
        sftp.put(str(root/'artifacts/audit/deployment_manifest.json'),remote+'/artifacts/audit/deployment_manifest.json')
        source_manifest = json.loads((root/'artifacts/audit/model_source_manifest.json').read_text())
        model_dir = root/'models/Qwen2.5-3B-Instruct'
        for entry in source_manifest['files']:
            name = entry['rfilename']
            local = model_dir/name
            target = remote+'/models/Qwen2.5-3B-Instruct/'+name
            mkdirs(sftp,target.rsplit('/',1)[0])
            try:
                if sftp.stat(target).st_size == entry['size']:
                    continue
            except OSError:
                pass
            partial = target+'.uploading'
            try:
                offset = sftp.stat(partial).st_size
            except OSError:
                offset = 0
            if offset > entry['size']:
                raise RuntimeError('Oversize remote upload partial')
            update('UPLOADING_VERIFIED_MODEL', file=name, offset=offset, total=entry['size'])
            with local.open('rb') as inp, sftp.open(partial,'ab' if offset else 'wb') as out:
                out.set_pipelined(True)
                inp.seek(offset)
                for chunk in iter(lambda: inp.read(2*1024*1024),b''):
                    out.write(chunk)
            if sftp.stat(partial).st_size != entry['size']:
                raise RuntimeError('Remote upload size mismatch')
            sftp.posix_rename(partial,target)
        update('VERIFYING_REMOTE_MODEL')
        verify = ('import json,sys,hashlib; from pathlib import Path; '
                  'sys.path.insert(0,'+repr(remote+'/scripts')+'); from download_model import verified; '
                  'r=Path('+repr(remote)+'); m=json.loads((r/"artifacts/audit/model_source_manifest.json").read_text()); '
                  'ok=all(verified(r/"models/Qwen2.5-3B-Instruct"/e["rfilename"],e) for e in m["files"]); '
                  'assert ok,"Remote model checksum mismatch"; '
                  'd=json.loads((r/"artifacts/audit/deployment_manifest.json").read_text()); '
                  'assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items()),"Remote code/config/data checksum mismatch"; '
                  '(r/"artifacts/audit/remote_model_verified.json").write_text(json.dumps({"status":"PASS","revision":m["revision"]})); print("PASS")')
        execute(client,verify,timeout=180)
        update('LAUNCHING_REMOTE_S0_G1')
        launch = ('import subprocess,json,os; from pathlib import Path; r=Path('+repr(remote)+'); '
                  '(r/"logs").mkdir(parents=True,exist_ok=True); '
                  'o=(r/"logs/supervisor.stdout.log").open("ab"); e=(r/"logs/supervisor.stderr.log").open("ab"); '
                  'p=subprocess.Popen(['+repr(args.remote_python)+',"-u",str(r/"scripts/server_supervisor.py"),"--root",str(r),"--gpus","0,4,5,6"],'
                  'cwd=str(r),stdin=subprocess.DEVNULL,stdout=o,stderr=e,start_new_session=True); print(p.pid)')
        remote_pid = int(execute(client,launch,timeout=30).strip())
        update('REMOTE_RUNNING_AND_MIRRORING',remote_supervisor_pid=remote_pid)
        sftp.close()
        client.close()
        while True:
            try:
                client = connect()
                sftp = client.open_sftp()
                listing = ('import os,json; from pathlib import Path; r=Path('+repr(remote)+'); a=[]; '
                           '\nfor part in ["RUN_STATUS.json","results","logs","artifacts/audit"]:'
                           '\n p=r/part'
                           '\n if not p.exists(): continue'
                           '\n for f in ([p] if p.is_file() else p.rglob("*")):'
                           '\n  if f.name.endswith((".tmp",".remote_partial",".uploading")): continue'
                           '\n  try:'
                           '\n   if not f.is_file(): continue'
                           '\n   s=f.stat()'
                           '\n  except FileNotFoundError: continue'
                           '\n  a.append([f.relative_to(r).as_posix(),s.st_size,int(s.st_mtime)])'
                           '\nprint(json.dumps(a))')
                files = json.loads(execute(client,listing,timeout=60))
                for name,size,mtime in files:
                    local = root/name
                    if local.exists() and local.stat().st_size==size and int(local.stat().st_mtime)==mtime:
                        continue
                    local.parent.mkdir(parents=True,exist_ok=True)
                    temp = local.with_name(local.name+'.remote_partial')
                    sftp.get(remote+'/'+name,str(temp))
                    os.replace(temp,local)
                    os.utime(local,(mtime,mtime))
                server = json.loads((root/'RUN_STATUS.json').read_text()) if (root/'RUN_STATUS.json').exists() else {}
                update('REMOTE_RUNNING_AND_MIRRORING',remote_stage=server.get('stage'), mirrored_files=len(files))
                sftp.close();client.close()
                if server.get('stage') in {'FAILED','G1_PARTIAL_FAILED','G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED'}:
                    update('REMOTE_TERMINAL_MIRRORED',remote_stage=server.get('stage'),workflow_complete=False)
                    break
                time.sleep(60)
            except (OSError,EOFError,paramiko.SSHException,json.JSONDecodeError,RuntimeError) as exc:
                update('MIRROR_CONNECTION_RETRY',error_type=type(exc).__name__)
                time.sleep(30)
    except Exception as exc:
        update('FAILED',error_type=type(exc).__name__,error=str(exc),workflow_complete=False)
        raise


if __name__ == '__main__':
    main()
