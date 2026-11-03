"""Label-free engineering message qualification on every reserved qualification item."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from graph_credit.backend import HFBackend
from graph_credit.executor import Executor, GraphConfig, digest
from run_pilot import read_jsonl, write_json, source_manifest, evaluate_message_records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-id', default='message_sanity_v3_20261103')
    args=p.parse_args()
    root=args.root
    out=root/'results/pilot'/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PLAN.json').exists():
        raise FileExistsError('No silent qualification retry or overwrite')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    graph=GraphConfig(**config['graph'])
    data=root/'data/derived/qualification_questions.jsonl'
    questions=read_jsonl(data)
    if len(questions)!=16:
        raise ValueError('Qualification sanity must cover all 16 reserved items')
    plan={'purpose':'engineering message conformity; no quality labels read','question_ids':[q['id'] for q in questions],
          'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'config':config,'source_manifest':source_manifest(root),'seed':1701,
          'scientific_completion':False}
    write_json(out/'PLAN.json',plan)
    write_json(out/'STATUS.json',{'stage':'LOADING_MODEL','pid':os.getpid(),'scientific_completion':False})
    backend=HFBackend(root/'models/Qwen2.5-3B-Instruct','cuda:0',out/'physical_calls.jsonl')
    executor=Executor(backend,graph)
    write_json(out/'MANIFEST.json',{'protocol_hash':executor.protocol_hash,'model_identity':backend.identity,
                                  'source_manifest_sha256':digest(plan['source_manifest'])})
    records=[]
    per_question=[]
    for q in questions:
        result=executor.execute(q,[1]*12,1701)
        write_json(out/'executions'/(q['id']+'.json'),result.to_dict())
        records.extend(result.nodes.items())
        per_question.append({"question_id":q["id"],"message_report":evaluate_message_records(list(result.nodes.items()),config.get("qualification_message_gate"))})
        write_json(out/'STATUS.json',{'stage':'GENERATING','completed_questions':len(records)//9,
                                      'physical_calls':len(backend.physical_calls),'scientific_completion':False})
    report=evaluate_message_records(records,config.get('qualification_message_gate'))
    write_json(out/'PER_QUESTION.json',per_question)
    report.update(purpose='16-item label-free message engineering qualification',questions=16,
                  labels_read=False,scientific_completion=False,protocol_hash=executor.protocol_hash,
                  physical_calls=len(backend.physical_calls),timestamp=time.time())
    write_json(out/'MESSAGE_SANITY.json',report)
    write_json(out/'STATUS.json',{'stage':'MESSAGE_SANITY_'+report['status'],'questions':16,
                                  'physical_calls':len(backend.physical_calls),'scientific_completion':False})
    print(json.dumps({'status':report['status'],'physical_calls':len(backend.physical_calls)}),flush=True)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
