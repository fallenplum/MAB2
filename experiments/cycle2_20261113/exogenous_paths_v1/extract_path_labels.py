"""Extract only the frozen train IDs after complete archive and CPU replay review."""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def check_release(approval):
    assert approval['decision']=='APPROVED_FIXED_PATH_TRAIN_LABEL_EXTRACTION'
    assert approval['source_sha256']==sha(__file__)
    evidence={}
    for name in ['selection','archive_manifest','replay_receipt','cost_closure','source_descriptor']:
        item=approval[name];assert sha(item['path'])==item['sha256'];evidence[name]=read(item['path'])
    selection=evidence['selection'];rows=selection['questions']
    ids=[r['id'] for r in rows]
    assert len(ids)==len(set(ids))==96
    assert [r['index'] for r in rows]==list(range(96))
    replay=evidence['replay_receipt']
    assert replay['status']=='PASS_CPU_EXACT_CLOSED_ARCHIVE_REPLAY' and replay['completed_units']==96
    assert replay['archive_manifest_sha256']==approval['archive_manifest']['sha256']
    assert replay['labels_read'] is False
    cost=evidence['cost_closure'];assert cost['actual_cost_closed'] is True
    assert cost['archive_manifest_sha256']==approval['archive_manifest']['sha256']
    root=Path(approval['archive_root']).resolve()
    for rel,pin in evidence['archive_manifest'].items():
        path=(root/rel).resolve();assert path.is_relative_to(root) and sha(path)==pin
    closure=read(root/'CLOSURE.json')
    assert closure['status']=='COMPLETE_UNSCORED' and closure['all_workers_exited']
    for i in range(96):
        units=list(root.glob(f'worker_*/unit_{i:03d}/COMPLETE.json'));assert len(units)==1
        unit=units[0].parent;inp=read(unit/'INPUT.json')
        assert inp['index']==i and inp['payload']['id']==ids[i]
        assert read(units[0])['terminal_conditions']==18
        assert read(units[0])['matrix_sha256']==sha(unit/'MATRIX.json')
    descriptor=evidence['source_descriptor']
    assert descriptor['dataset_id']=='hotpotqa/hotpot_qa'
    assert descriptor['revision']=='1908d6afbbead072334abe2965f91bd2709910ab'
    assert descriptor['config']=='distractor' and descriptor['split']=='train'
    for item in descriptor['parquet_files']:assert sha(item['path'])==item['sha256']
    return ids,descriptor

def extract(ids,descriptor):
    # Arrow may decode unselected row-group buffers; only selected values leave Arrow.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    wanted=pa.array(ids,type=pa.string());vault={};scanned=0
    for item in descriptor['parquet_files']:
        source=pq.ParquetFile(item['path']);assert source.metadata.num_rows==item['num_rows']
        for batch in source.iter_batches(columns=['id','answer'],batch_size=4096):
            scanned+=len(batch)
            table=pa.Table.from_batches([batch]);keep=pc.is_in(table['id'],value_set=wanted)
            for row in table.filter(keep).to_pylist():
                assert row['id'] not in vault and isinstance(row['answer'],str)
                vault[row['id']]=row['answer']
    assert set(vault)==set(ids)
    return vault,scanned

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--approval-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert sha(a.approval)==a.approval_sha
    out=Path(a.output);assert not out.exists()
    approval=read(a.approval);ids,descriptor=check_release(approval)
    vault,scanned=extract(ids,descriptor)
    out.mkdir(parents=True)
    (out/'VAULT.json').write_text(json.dumps(vault,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    receipt={'status':'FIXED_96_TRAIN_LABELS_EXTRACTED_AFTER_CLOSURE','rows_scanned':scanned,'labels_written':len(vault),
      'approval_sha256':a.approval_sha,'vault_sha256':sha(out/'VAULT.json'),'source_sha256':sha(__file__),
      'official_validation_or_test_opened':False,'scoring_performed':False,'new_GPU_calls':0,
      'limitation':'Arrow may decode nonselected answers in row-group buffers; Python materialization and persisted output contain selected IDs only.'}
    (out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='limitation'}))
if __name__=='__main__':main()
