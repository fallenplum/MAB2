"""Small CLI isolation test: intentionally nonexistent archive/question paths."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='synthetic_cli_',dir=ROOT) as td:
    temp=Path(td);raw=b'{"label_vault_sha256":null}\n';(temp/'contract.json').write_bytes(raw)
    missing=str(temp/'FORBIDDEN_NONEXISTENT_INPUT')
    io={'contract':str(temp/'contract.json'),'plan':missing,'admission':missing,'input_files':{'questions':missing},
        'auxiliary_files':{},'source_files':{},'archive_root':missing,'tokenizer_roots':{},'prior_receipt_files':{}}
    (temp/'io.json').write_text(json.dumps(io),encoding='utf-8')
    command=[sys.executable,'-B',str(ROOT/'audit_cli.py'),'--io',str(temp/'io.json'),'--contract-sha256',hashlib.sha256(raw).hexdigest()]
    result=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,(result.stdout,result.stderr)
    output=json.loads(result.stdout)
    assert output['status']=='PREFLIGHT_ONLY_NO_ARCHIVE_OR_QUESTIONS_OPENED' and not output['label_release_authorized']
    command[-1]='0'*64
    failed=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert failed.returncode!=0
receipt={'status':'PASS_CLI_DEFAULT_INPUT_ISOLATION','cases':2,'benchmark_or_labels_read':False,'GPU_calls':0,
         'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'cli_source_sha256':hashlib.sha256((ROOT/'audit_cli.py').read_bytes()).hexdigest()}
with (ROOT/'CLI_PREFLIGHT_TEST_RECEIPT.json').open('x',encoding='utf-8') as handle:json.dump(receipt,handle,indent=2);handle.write('\n')
print(json.dumps(receipt))
