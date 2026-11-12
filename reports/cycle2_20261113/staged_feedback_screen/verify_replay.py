"""Compare aggregate outputs and validate the declared post-result source revision."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(replay):
    mapping=json.loads((HERE/'SOURCE_HASH_MAP.json').read_text('utf-8'))
    entries={x['original_name']:x for x in mapping['source_map']}
    for row in mapping['source_map']:
        assert sha(HERE/row['public_name'])==row['public_sha256'], 'Public lineage hash mismatch'
    original=json.loads((HERE/'HISTORICAL_RESULTS.json').read_text('utf-8'))
    actual=json.loads((replay/'RESULTS.json').read_text('utf-8'))
    timing={'historical':original.pop('elapsed_wall_seconds'),'replay':actual.pop('elapsed_wall_seconds')}
    original_pins=original.pop('source_sha256')
    actual_pins=actual.pop('source_sha256')
    assert original_pins.keys()==actual_pins.keys(), 'Input/source provenance keys changed'
    changes=[]
    for name, old_pin in original_pins.items():
        new_pin=actual_pins[name]
        if old_pin!=new_pin:
            row=entries[name]
            assert old_pin==row['original_sha256'] and new_pin==row['public_sha256'], 'Unexplained source pin change'
            changes.append({'name':name,'original_sha256':old_pin,'public_sha256':new_pin})
    assert [x['name'] for x in changes]==['run_screen.py'], 'Unexpected provenance difference'
    assert actual==original, 'Scientific JSON fields differ'
    assert (replay/'FRONTIER.csv').read_bytes()==(HERE/'HISTORICAL_FRONTIER.csv').read_bytes(), 'Frontier bytes differ'
    with (replay/'FRONTIER.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1248
    assert all(x['variance'] in ('','0.0') for x in rows if float(x['fraction'])==1.)
    zero=[x for x in rows if x['base_mode']=='lazy_baseline' and x['method']=='staged_Poisson_HT_oracle' and x['scenario'] in ('observed_token_score_equivalent_ratio_0','observed_token_score_equivalent_ratio_0.1') and float(x['fraction'])<1]
    assert len(zero)==48 and all(x['p1']=='1.0' and x['p2']=='1.0' and abs(float(x['variance_ratio_to_direct_Poisson'])-1)<1e-12 for x in zero)
    counter=[x for x in rows if x['split']=='tune' and x['base_mode']=='lazy_baseline' and x['method']=='staged_Poisson_HT_oracle' and x['scenario']=='observed_token_score_equivalent_ratio_1' and float(x['fraction'])==.9]
    assert len(counter)==1 and float(counter[0]['p1'])<1 and float(counter[0]['variance_ratio_to_direct_SRS'])<1
    before={p.name:sha(p) for p in replay.iterdir() if p.is_file()}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    guard=subprocess.run([sys.executable,str(HERE/'run_screen.py'),'--diagnostic-dir','NONEXISTENT_AUTHORIZED_INPUT','--d2-inputs','NONEXISTENT_AUTHORIZED_INPUT.npz','--output',str(replay)],capture_output=True,text=True,env=env)
    assert guard.returncode==2 and 'Output already exists; refusing to overwrite' in guard.stderr
    after={p.name:sha(p) for p in replay.iterdir() if p.is_file()}
    assert before==after, 'Overwrite guard modified output'
    return {'schema':'public_interface_replay_validation_v1','status':'PASS','frontier_rows':len(rows),'frontier_byte_identical':True,'frontier_sha256':sha(replay/'FRONTIER.csv'),'scientific_json_exact_match':True,'timing_field_excluded':timing,'expected_post_result_provenance_changes':changes,'other_input_and_source_hashes_identical':True,'overwrite_guard_passed':True,'zero_score_and_point_one_degeneracy_rows':len(zero),'score_ratio_one_counterexample_preserved':True,'new_gpu_hours':0,'new_model_calls':0,'new_formal_rounds':0,'raw_question_or_R1_or_official_test_read':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay-dir',required=True,type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    result=verify(args.replay_dir.resolve())
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.receipt:
        with args.receipt.open('x',encoding='utf-8') as f:
            f.write(text)
    print(text)
