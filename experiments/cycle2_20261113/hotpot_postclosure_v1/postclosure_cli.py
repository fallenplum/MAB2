"""Explicit postclosure actions. Default preflight opens no dataset/archive/labels."""
from collections.abc import Mapping
from pathlib import Path
import argparse
import json
import sys
import time

sys.dont_write_bytecode = True
import postclosure as p


class ArchiveSubset(Mapping):
    def __init__(self, root, template):
        self.root = Path(root)
        self.names = set(p.META_NAMES)
        for entry in template['units']:
            self.names.update(entry['directory'] + '/' + name for name in entry['sha256'])

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        if key not in self.names:
            raise KeyError(key)
        root = self.root.resolve(strict=True)
        path = root / key
        p.require(path.resolve(strict=True).is_relative_to(root), 'Archive path escapes root')
        p.require(not any(item.is_symlink() for item in (path, *path.parents) if item != root.parent),
                  'Symlink archive path not permitted')
        return path.read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'extract', 'score'), default='preflight')
    parser.add_argument('--io', type=Path)
    parser.add_argument('--extraction-approval-sha256')
    parser.add_argument('--score-review-sha256')
    args = parser.parse_args(argv)
    if args.mode == 'preflight':
        print(json.dumps({'status': 'PREFLIGHT_NO_DATA_IO', 'labels_read': False,
            'scoring_authorized': False, 'GPU_admission_granted': False}))
        return
    p.require(args.io is not None, 'Explicit private IO descriptor required')
    io = p.decode(args.io.read_bytes())
    common = {'approval', 'evidence', 'output'}
    extra = {'parquet_paths_by_sha256'} if args.mode == 'extract' else {
        'score_review', 'extraction_receipt', 'label_vault', 'archive_root', 'adapter_path'}
    p.require(set(io) == common | extra and set(io['evidence']) == p.EVIDENCE_NAMES, 'IO schema differs')
    def path(value):
        item = Path(value)
        return item if item.is_absolute() else args.io.parent / item
    approval_bytes = path(io['approval']).read_bytes()
    evidence = {key: path(value).read_bytes() for key, value in io['evidence'].items()}
    common_args = dict(approval_bytes=approval_bytes,
        expected_approval_sha256=args.extraction_approval_sha256, evidence=evidence)
    validated = p.validate_extraction_release(**common_args)
    output = path(io['output'])
    p.require(not output.exists(), 'Fresh private output directory required; no overwrite/resume')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    if args.mode == 'extract':
        vault, receipt = p.extract_vault(**common_args,
            paths_by_sha256={pin: path(value) for pin, value in io['parquet_paths_by_sha256'].items()})
        p.write_new(output / 'LABEL_VAULT.json', vault)
        p.require(p.sha((output / 'LABEL_VAULT.json').read_bytes()) == receipt['label_vault_sha256'],
                  'Persisted label vault differs')
        receipt['elapsed_seconds'] = time.monotonic() - started
        p.write_new(output / 'EXTRACTION_RECEIPT.json', p.canonical(receipt))
        print(json.dumps({'status': receipt['status'], 'selected_label_count': receipt['selected_label_count'],
            'receipt_sha256': p.sha((output / 'EXTRACTION_RECEIPT.json').read_bytes()),
            'scoring_authorized': False, 'publication_authorized': False}))
    else:
        # Merely constructing this mapping opens no raw file. The pinned adapter
        # calls the vault callback only after its complete 384/4224 bind succeeds.
        archive = ArchiveSubset(path(io['archive_root']), validated['template'])
        result = p.score_after_release(**common_args,
            extraction_receipt_bytes=path(io['extraction_receipt']).read_bytes(),
            score_review_bytes=path(io['score_review']).read_bytes(),
            expected_score_review_sha256=args.score_review_sha256, archive_files=archive,
            vault_loader=lambda request: path(io['label_vault']).read_bytes(), adapter_path=path(io['adapter_path']))
        p.write_new(output / 'PRIVATE_EVALUATION.json', p.canonical(result))
        receipt = {'status': 'COMPLETE_PRIVATE_FROZEN_EVALUATION',
            'postclosure_source_sha256': p.source_sha(), 'evaluation_adapter_sha256': p.ADAPTER_PIN,
            'extraction_approval_sha256': args.extraction_approval_sha256,
            'score_review_sha256': args.score_review_sha256,
            'evaluation_sha256': p.sha((output / 'PRIVATE_EVALUATION.json').read_bytes()),
            'elapsed_seconds': time.monotonic() - started, 'new_model_calls': 0,
            'publication_authorized': False, 'scientific_or_GPU_admission_granted': False}
        p.write_new(output / 'EVALUATION_RECEIPT.json', p.canonical(receipt))
        print(json.dumps({'status': receipt['status'], 'receipt_sha256': p.sha(p.canonical(receipt)),
            'publication_authorized': False, 'scientific_or_GPU_admission_granted': False}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Exceptions can carry private paths or answer data; print the class only.
        print(json.dumps({'status': 'BLOCKED_POSTCLOSURE', 'error_type': type(error).__name__,
                          'automatic_retry': False}), file=sys.stderr)
        raise SystemExit(2)
