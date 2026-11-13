"""Explicit CPU-only archive review. Default preflight does not open question files.

The IO descriptor maps original plan paths to local immutable copies. No network,
model loading, labels, retry, publication or automatic release is implemented.
"""
from collections.abc import Mapping
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
sys.dont_write_bytecode=True
import archive_audit as a

class ArchiveFiles(Mapping):
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True);self.paths={}
        for path in self.root.rglob('*'):
            if path.is_symlink():raise a.AuditError('Symlink in archive')
            if path.is_file():
                resolved=path.resolve(strict=True)
                if not resolved.is_relative_to(self.root):raise a.AuditError('Archive path escapes root')
                self.paths[path.relative_to(self.root).as_posix()]=path
    def __getitem__(self,name):return self.paths[name].read_bytes()
    def __iter__(self):return iter(self.paths)
    def __len__(self):return len(self.paths)

def load_map(spec):return {name:Path(path).read_bytes() for name,path in spec.items()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--io',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--audit',action='store_true',help='Explicitly open unlabelled collector archive and perform CPU review')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();io=a.decode(args.io.read_bytes());contract_raw=Path(io['contract']).read_bytes()
    a.require(a.sha(contract_raw)==args.contract_sha256,'Externally supplied contract pin differs')
    contract=a.decode(contract_raw)
    required={'contract','plan','admission','input_files','auxiliary_files','source_files','archive_root','tokenizer_roots','prior_receipt_files'}
    a.require(set(io)==required,'IO descriptor keys differ; label files are not supported')
    if not args.audit:
        print(a.canonical({'status':'PREFLIGHT_ONLY_NO_ARCHIVE_OR_QUESTIONS_OPENED','contract_sha256':a.sha(contract_raw),
             'label_release_authorized':False,'archive_review_performed':False}));return
    a.require(args.output is not None and not args.output.exists(),'New explicit output directory required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();stage='read frozen metadata'
    try:
        plan_raw=Path(io['plan']).read_bytes();plan=a.decode(plan_raw)
        a.require(a.sha(plan_raw)==contract['plan_sha256'],'plan bytes changed')
        from tokenizer_witness import OfflineTokenizerWitness
        stage='CPU tokenizer identity and chat template'
        tokenizers={m:OfflineTokenizerWitness(path,plan['runtime']['models'][m]['identity']) for m,path in io['tokenizer_roots'].items()}
        a.require(set(tokenizers)=={'qwen','mistral'},'two CPU tokenizer witnesses required')
        stage='closed archive/source/native/input/residence review'
        archive=ArchiveFiles(io['archive_root'])
        result=a.audit_bundle(plan_bytes=plan_raw,admission_bytes=Path(io['admission']).read_bytes(),contract_bytes=contract_raw,
            pinned_inputs=load_map(io['input_files']),auxiliary=load_map(io['auxiliary_files']),sources=load_map(io['source_files']),
            archive=archive,tokenizers=tokenizers,prior_receipts=load_map(io['prior_receipt_files']))
        # Bind the exact final metadata and revisit the immutable raw manifest
        # after CPU work; no in-place correction or silent snapshot change.
        final=ArchiveFiles(io['archive_root']);a.require(set(final)==set(archive),'Archive inventory changed during review')
        for name,pin in result['external_release_template']['archive_metadata_sha256'].items():a.require(a.sha(final[name])==pin,'Metadata changed during review')
        manifest=a.decode(final['RAW_OUTPUT_MANIFEST.json'])
        for name,pin in manifest['files'].items():a.require(a.sha(final[name])==pin,'Raw bytes changed during review')
        template=result.pop('external_release_template')
        (args.output/'REVIEW_REQUIRED_TEMPLATE.json').write_bytes((a.canonical(template)+'\n').encode())
        result['template_sha256']=a.sha((args.output/'REVIEW_REQUIRED_TEMPLATE.json').read_bytes())
    except Exception as exc:
        # Never print exception text, which could contain private paths or prompts.
        result={'status':'BLOCKED_EXTERNAL_ARCHIVE_AUDIT','stage':stage,'error_type':type(exc).__name__,
                'label_release_authorized':False,'GPU_admission_granted':False}
        (args.output/'PRIVATE_ERROR.txt').write_text(str(exc),encoding='utf-8')
    result.update(elapsed_seconds=time.monotonic()-started,contract_sha256=a.sha(contract_raw),
        audit_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('archive_audit.py','tokenizer_witness.py','audit_cli.py')})
    (args.output/'AUDIT.json').write_bytes((a.canonical(result)+'\n').encode())
    print(a.canonical({'status':result['status'],'label_release_authorized':False,
                      'audit_sha256':a.sha((args.output/'AUDIT.json').read_bytes())}))
    if result['status']!='REVIEW_REQUIRED':raise SystemExit(2)

if __name__=='__main__':main()
