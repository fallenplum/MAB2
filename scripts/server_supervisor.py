"""Run only S0 and the preregistered G1 audit, with real gates and no inference retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', default='0,4,5,6')
    args = parser.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    (root/'logs').mkdir(exist_ok=True)
    lock = (root/'artifacts/supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'schema':'graph_credit_supervisor_v1', 'pid':os.getpid(), 'stage':'ADMISSION', 'scientific_completion':False,
             'shared_gpu_authorized':True, 'inference_retries':0, 'scope':'S0_then_G1_only'}
    status = root/'RUN_STATUS.json'
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    children = []
    all_children = []
    handles = []
    def update(stage, **kw):
        state.update(stage=stage, updated_at_unix=time.time(), **kw)
        write(status, state)
    def start(command, gpu, label):
        run_env = env.copy()
        run_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        stdout = (root/'logs'/f'{label}.stdout.log').open('ab')
        stderr = (root/'logs'/f'{label}.stderr.log').open('ab')
        handles.extend([stdout,stderr])
        child = subprocess.Popen(command, env=run_env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        all_children.append(child)
        return child
    def command(data, labels, config, run):
        result = [sys.executable,'-u','scripts/run_pilot.py','--root',str(root),'--data',str(root/'data/derived'/data),
                  '--model',str(root/'models/Qwen2.5-3B-Instruct'),'--devices','0','--config',str(root/'configs'/config),'--run-id',run]
        if labels:
            result += ['--labels',str(root/'data/derived'/labels)]
        return result
    try:
        manifest = json.loads((root/'artifacts/audit/model_download_status.json').read_text())
        if manifest.get('status') != 'PASS':
            raise RuntimeError('Model verification has not passed')
        gpus = [int(x) for x in args.gpus.split(',')]
        while True:
            raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
            free = {int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.strip().splitlines()}
            admitted = [gpu for gpu in gpus if free.get(gpu,0) >= 12000]
            write(root/'artifacts/audit/gpu_admission.json', {'timestamp':time.time(), 'free_mib':free, 'requested':gpus, 'admitted':admitted, 'minimum_free_mib':12000, 'sharing':True})
            if admitted:
                break
            update('WAITING_GPU_MEMORY', free_mib=free)
            time.sleep(30)
        s0run = 's0_v2_20261103'
        child = start(command('qualification_questions.jsonl',None,'qualification_v2.json',s0run)+['--limit','1','--qualification-only'], admitted[0], 's0_v2')
        update('S0_REAL_QUALIFICATION', active_pids=[child.pid], gpus=[admitted[0]])
        result = child.wait()
        if result:
            raise RuntimeError(f'S0 failed, return code {result}; stop before G1')
        qualification = root/'results/pilot'/s0run/'QUALIFICATION.json'
        q = json.loads(qualification.read_text())
        if q.get('status') != 'PASS' or not q.get('eligible_for_G1'):
            raise RuntimeError('S0 artifact does not pass')
        sanity_run='message_sanity_v2_20261103'
        sanity_cmd=[sys.executable,'-u','scripts/run_message_sanity.py','--root',str(root),'--config',str(root/'configs/qualification_v2.json'),'--run-id',sanity_run]
        sanity=start(sanity_cmd,admitted[0],'message_sanity_v2')
        update('S0_16_ITEM_MESSAGE_SANITY',active_pids=[sanity.pid],s0_execution='PASS')
        if sanity.wait():
            raise RuntimeError('16-item message conformity qualification failed; stop before G1')
        sanity_report=json.loads((root/'results/pilot'/sanity_run/'MESSAGE_SANITY.json').read_text())
        if sanity_report.get('status')!='PASS':
            raise RuntimeError('16-item qualification artifact not PASS')
        # Recheck sharing admission after engineering calls; other jobs may
        # have changed memory occupancy while S0 was running.
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            current={int(row.split(',')[0]):int(row.split(',')[1]) for row in raw.strip().splitlines()}
            admitted=[gpu for gpu in gpus if current.get(gpu,0)>=12000]
            if admitted:
                break
            update('WAITING_GPU_MEMORY_AFTER_QUALIFICATION',free_mib=current)
            time.sleep(30)
        run = 'motivation_v2_20261103'
        for index,gpu in enumerate(admitted):
            cmd = command('motivation_questions.jsonl','motivation_labels.jsonl','motivation_v2.json',run)
            cmd += ['--shard-index',str(index),'--num-shards',str(len(admitted))]
            children.append(start(cmd,gpu,f'g1_shard_{index:03}'))
        update('G1_REAL_GENERATION', active_pids=[c.pid for c in children], gpus=admitted, s0='PASS', expected_questions=200,
               expected_graph_observations=60000, expected_seeds=5)
        while any(c.poll() is None for c in children):
            update('G1_REAL_GENERATION', child_returncodes=[c.poll() for c in children])
            time.sleep(20)
        codes = [c.returncode for c in children]
        if any(codes):
            update('G1_PARTIAL_FAILED', child_returncodes=codes, workflow_complete=False)
            return 4
        # Each shard uses the exact same global plan and model identity.
        parent = root/'results/pilot'/run
        paths = [parent/f'shard_{i:03d}' for i in range(len(admitted))] if len(admitted)>1 else [parent]
        plans = [(p/'frozen_plan.json').read_bytes() for p in paths]
        if any(x != plans[0] for x in plans):
            raise RuntimeError('Shards do not share a frozen plan')
        combined = parent/'all_observations.jsonl'
        with combined.open('wb') as out:
            for path in paths:
                out.write((path/'observations.jsonl').read_bytes())
        analysis_cmd = [sys.executable,'scripts/analyze_pilot.py','--observations',str(combined),'--plan',str(paths[0]/'frozen_plan.json'),'--config',str(root/'configs/gate_v1.json'),
                        '--output',str(parent/'G1_ANALYSIS.json')]
        analysis = subprocess.run(analysis_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (root/'logs/g1_analysis.stdout.log').write_text(analysis.stdout)
        (root/'logs/g1_analysis.stderr.log').write_text(analysis.stderr)
        if analysis.returncode:
            raise RuntimeError('Independent G1 analysis failed; see log')
        update('G1_GENERATION_AND_ANALYSIS_COMPLETE_REVIEW_REQUIRED', active_pids=[], workflow_complete=False,
               result=str(parent/'G1_ANALYSIS.json'), note='No acquisition algorithm or final test is launched automatically')
        return 0
    except Exception as exc:
        # Retain ownership of already-started project children and keep the
        # mirror alive until their paid work settles. Never touch other jobs.
        while any(c.poll() is None for c in all_children):
            update('FAILED_CHILDREN_SETTLING', error_type=type(exc).__name__, error=str(exc),
                   active_pids=[c.pid for c in all_children if c.poll() is None], workflow_complete=False)
            time.sleep(20)
        update('FAILED', error_type=type(exc).__name__, error=str(exc), workflow_complete=False)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
