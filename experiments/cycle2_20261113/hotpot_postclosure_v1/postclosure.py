"""Postclosure label extraction and frozen evaluation orchestration; no authority.

Only externally trusted root receipt hashes authorize the two separate actions.
Importing this module performs no dataset/archive IO or model loading. Arrow
column projection and ID filtering do NOT guarantee row-level byte isolation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
N = 384
TERMINALS = 4224
REVISION = '1908d6afbbead072334abe2965f91bd2709910ab'
ADAPTER_PIN = 'f8ba38be9ce1ef6bbfee66c17606685b117d25c52b947e66937ea5574a6105b8'
SELECTOR_PIN = '5edc8dc62b349be49c2c939c9c3f93ba1a8795b3c6fc7c951aec3ab9cd3d8bfd'
SOURCE_COLUMNS = {'id', 'question', 'context', 'type', 'level', 'answer', 'supporting_facts'}
BINDINGS = {'scientific_protocol_sha256', 'selection_manifest_sha256', 'plan_sha256',
    'source_manifest_sha256', 'matrix_protocol_hash', 'graph_protocol_hash',
    'independent12_protocol_hash', 'single_protocol_hash', 'scorer_sha256',
    'statistics_sha256', 'frontier_sha256', 'selector_sha256', 'evaluation_adapter_sha256'}
SCOPES = {'source_model_tokenizer_identity', 'native_durable_generation_and_cost_lineage',
    'cache_input_and_output_lineage', 'all_attempts_and_physical_residence_ledger',
    'no_labels_during_collection'}
META_NAMES = {'RAW_OUTPUT_MANIFEST.json', 'PARENT_RESIDENCE.json', 'ATTEMPT_RESULT.json'}
EVIDENCE_NAMES = {'audit', 'template', 'audit_contract', 'selection_manifest', 'source_descriptor'}
AUDIT_SOURCES = {'archive_audit.py', 'tokenizer_witness.py', 'audit_cli.py'}


class ReleaseError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ReleaseError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def sha(raw):
    require(type(raw) is bytes, 'Exact bytes required')
    return hashlib.sha256(raw).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def decode(raw):
    require(type(raw) is bytes, 'Exact JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ReleaseError('Nonfinite JSON')))


def source_sha():
    return sha(Path(__file__).read_bytes())


def validate_extraction_release(*, approval_bytes, expected_approval_sha256, evidence):
    """Validate only metadata bytes, before any answer reader can be invoked.

    The caller must obtain expected_approval_sha256 from an independent root
    decision. A self-computed hash or the word 'root' is not authentication.
    This function trusts the hash-bound external audit; it does not redo it.
    """
    require(is_hash(expected_approval_sha256) and sha(approval_bytes) == expected_approval_sha256,
            'External extraction approval SHA absent/different')
    approval = decode(approval_bytes)
    require(approval['schema'] == 'hotpot_postclosure_label_extraction_release_v1'
            and approval['decision'] == 'APPROVED_FOR_LABEL_EXTRACTION_ONLY', 'No extraction-only release')
    require(approval['postclosure_source_sha256'] == source_sha(), 'Wrapper source differs')
    require(approval['expected_questions'] == N and type(approval['expected_questions']) is int
            and approval['expected_terminal_conditions'] == TERMINALS
            and type(approval['expected_terminal_conditions']) is int, 'Fixed coverage differs')
    require(approval['allowed_columns'] == ['id', 'answer']
            and approval['output_purpose'] == 'PRIVATE_LABEL_VAULT'
            and approval['scoring_authorized'] is False
            and approval['publication_authorized'] is False, 'Extraction scope differs')
    require(set(evidence) == EVIDENCE_NAMES and set(approval['evidence_sha256']) == EVIDENCE_NAMES,
            'Complete evidence pins required')
    require(all(is_hash(approval['evidence_sha256'][k])
                and sha(evidence[k]) == approval['evidence_sha256'][k] for k in EVIDENCE_NAMES),
            'Extraction evidence bytes differ')
    audit, template, contract, manifest, descriptor = [decode(evidence[k]) for k in
        ('audit', 'template', 'audit_contract', 'selection_manifest', 'source_descriptor')]
    require(audit['status'] == 'REVIEW_REQUIRED' and audit['synthetic_tokenizer_witness'] is False,
            'Real tokenizer witness audit required')
    require(audit['no_label_vault_read'] is True and audit['label_vault_binding_pending'] is True
            and audit['label_release_authorized'] is False and audit['GPU_admission_granted'] is False,
            'Audit scope/label boundary differs')
    require(audit['template_sha256'] == sha(evidence['template'])
            and audit['contract_sha256'] == sha(evidence['audit_contract']), 'Audit ancestry differs')
    require(set(approval['audit_source_sha256']) == AUDIT_SOURCES
            and all(is_hash(v) for v in approval['audit_source_sha256'].values())
            and audit['audit_source_sha256'] == approval['audit_source_sha256'], 'Audit implementation pins differ')
    require(template['schema'] == 'hotpot_closed_archive_score_release_v1'
            and template['decision'] == 'REVIEW_REQUIRED' and template['label_vault_sha256'] is None,
            'Original unapproved audit template required')
    require(template['scientific_protocol_status'] == 'FROZEN'
            and set(template['reviewed_scopes']) == SCOPES, 'External audit scope incomplete')
    for key, value in (('unique_questions', N), ('terminal_conditions', TERMINALS), ('runtime_failure_count', 0)):
        require(type(template[key]) is int and template[key] == value, 'Incomplete terminal coverage')
    require(all(template[k] is True for k in ('all_attempts_closed', 'physical_ledger_closed', 'resident_ledger_closed')),
            'Unclosed archive/physical residence')
    bindings = template['bindings']
    require(set(bindings) == BINDINGS and all(is_hash(v) for v in bindings.values())
            and bindings == contract['bindings'], 'Audit contract bindings differ')
    require(bindings['evaluation_adapter_sha256'] == ADAPTER_PIN and bindings['selector_sha256'] == SELECTOR_PIN,
            'Frozen adapter/selector differs')
    require(bindings['selection_manifest_sha256'] == sha(evidence['selection_manifest']), 'Selection bytes differ')
    require(manifest['classification'] == 'PRIVATE_NO_LABELS'
            and manifest['pins']['protocol_sha256'] == bindings['scientific_protocol_sha256']
            and manifest['pins']['selector_sha256'] == SELECTOR_PIN
            and manifest['pins']['source_descriptor_sha256'] == sha(evidence['source_descriptor']),
            'Selection/source/scientific lineage differs')
    questions = manifest['questions']
    require(len(questions) == N and all(isinstance(q['id'], str) and q['id']
            and q['id'] == q['id'].strip() and not q['id'].startswith('hotpotqa_') for q in questions),
            'Fixed raw-ID selection required')
    ids = tuple(q['id'] for q in questions)
    require(len(set(ids)) == N and Counter(q['edge_index'] for q in questions) == Counter({e: 96 for e in (1, 5, 7, 9)}),
            'Duplicate IDs or changed edge allocation')
    require(all(type(q['generation_seed1']) is int and type(q['generation_seed2']) is int
                and (q['generation_seed1'], q['generation_seed2']) == (26111301, 26111302) for q in questions),
            'Selection generation seeds differ')
    entries = template['units']
    require(len(entries) == N and all(type(u['index']) is int for u in entries)
            and {u['index'] for u in entries} == set(range(N)), 'Missing/duplicate closed units')
    for entry in entries:
        require(re.fullmatch(r'worker_[0-3]/unit_' + f"{entry['index']:03d}", entry['directory']) is not None,
                'Closed unit directory/index differs')
        require(set(entry['sha256']) == {'INPUT.json', 'MATRIX.json', 'UNIT_RECEIPT.json'}
                and all(is_hash(v) for v in entry['sha256'].values()), 'Closed unit pins incomplete')
    require(set(template['archive_metadata_sha256']) == META_NAMES
            and all(is_hash(v) for v in template['archive_metadata_sha256'].values()), 'Closed metadata pins missing')
    unknown = template['unknown_preflight_component_requests']
    require(type(unknown) is int and unknown >= 0
            and type(template['input_overflow_conditions']) is int
            and 0 <= template['input_overflow_conditions'] <= TERMINALS
            and type(audit['preflight_rejections_with_no_native_record']) is int
            and audit['preflight_rejections_with_no_native_record'] == unknown,
            'Unknown preflight accounting differs')
    require(template['all_native_cost_fields_complete'] is (unknown == 0)
            and template['root_must_acknowledge_unknown_preflight_components_before_release'] is (unknown > 0),
            'Preflight closure flags differ')
    require(type(approval['unknown_preflight_components_acknowledged']) is bool
            and (unknown == 0 or approval['unknown_preflight_components_acknowledged'] is True),
            'Unknown preflight components need explicit root acknowledgement')
    identity = {k: descriptor[k] for k in ('dataset_id', 'revision', 'config', 'split')}
    require(identity == approval['source_identity'] and identity['revision'] == REVISION
            and identity['config'] == 'distractor' and identity['split'] == 'train'
            and isinstance(identity['dataset_id'], str) and identity['dataset_id'], 'Fixed train source identity required')
    require(descriptor['schema_version'] == '1.0' and descriptor['status'] == 'FROZEN'
            and is_hash(descriptor['source_audit_sha256'])
            and descriptor['read_columns'] == ['id', 'question', 'context', 'type', 'level'],
            'Original frozen unlabelled source descriptor required')
    shards = descriptor['parquet_files']
    require(len(shards) == 2 and len({s['sha256'] for s in shards}) == 2
            and all(is_hash(s['sha256']) and type(s['num_rows']) is int and s['num_rows'] > 0 for s in shards),
            'Exactly two pinned train shards required')
    return {'approval': approval, 'template': template, 'descriptor': descriptor, 'ids': ids,
            'approval_sha256': expected_approval_sha256, 'evidence_sha256': dict(approval['evidence_sha256'])}


def _stream_sha(stream):
    stream.seek(0)
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    stream.seek(0)
    return h.hexdigest()


def read_selected_parquet(*, shards, paths_by_sha256, selected_ids):
    """Called only after release; projects id/answer. No model/dataset fetches.

    Hashing reads whole opaque shard bytes. Arrow may decode id/answer for other
    rows or whole row groups before filtering. Only selected answers are returned
    to Python; no claim of byte isolation or absence of transient extra decoding.
    """
    require(set(paths_by_sha256) == {s['sha256'] for s in shards}, 'Parquet path map must exactly match the two pins')
    import pyarrow.parquet as pq
    rows = []
    for item in shards:
        path = Path(paths_by_sha256[item['sha256']])
        require(path.suffix.lower() == '.parquet', 'Pinned parquet path required')
        with path.open('rb') as stream:
            require(_stream_sha(stream) == item['sha256'], 'Train shard SHA differs before decode')
            parquet = pq.ParquetFile(stream)
            require(set(parquet.schema_arrow.names) == SOURCE_COLUMNS
                    and parquet.metadata.num_rows == item['num_rows'], 'Pinned source schema/footer differs')
            stream.seek(0)
            table = pq.read_table(stream, columns=['id', 'answer'], filters=[('id', 'in', list(selected_ids))])
            require(table.column_names == ['id', 'answer'], 'Label projection differs')
            rows.extend(table.to_pylist())
            require(_stream_sha(stream) == item['sha256'], 'Train shard changed during extraction')
    return rows


def extract_vault(*, approval_bytes, expected_approval_sha256, evidence, paths_by_sha256, reader=None):
    validated = validate_extraction_release(approval_bytes=approval_bytes,
        expected_approval_sha256=expected_approval_sha256, evidence=evidence)
    require(set(paths_by_sha256) == {s['sha256'] for s in validated['descriptor']['parquet_files']}, 'Source path mapping differs')
    # Reader cannot run on an unapproved or incorrectly pinned request.
    rows = (reader or read_selected_parquet)(shards=validated['descriptor']['parquet_files'],
        paths_by_sha256=paths_by_sha256, selected_ids=validated['ids'])
    labels = {}
    selected = set(validated['ids'])
    for row in rows:
        require(set(row) == {'id', 'answer'} and isinstance(row['id'], str) and row['id'] in selected
                and row['id'] not in labels and isinstance(row['answer'], str), 'Missing/extra/duplicate or malformed selected label')
        labels[row['id']] = row['answer']
    require(set(labels) == selected and len(labels) == N, 'Selected label coverage differs')
    require(source_sha() == validated['approval']['postclosure_source_sha256'], 'Wrapper changed during extraction')
    vault = canonical(labels)
    receipt = {'schema': 'hotpot_postclosure_label_vault_receipt_v1', 'status': 'COMPLETE_PRIVATE_LABEL_EXTRACTION',
        'extraction_approval_sha256': expected_approval_sha256, 'postclosure_source_sha256': source_sha(),
        'evidence_sha256': validated['evidence_sha256'], 'label_vault_sha256': sha(vault), 'selected_label_count': N,
        'source_shard_sha256': [s['sha256'] for s in validated['descriptor']['parquet_files']],
        'source_split': 'train', 'source_revision': REVISION, 'projected_columns': ['id', 'answer'],
        'other_rows_may_be_decoded_by_arrow': True, 'row_level_byte_isolation': False,
        'supporting_facts_projected': False, 'validation_or_test_opened': False,
        'scoring_performed': False, 'scoring_authorized': False, 'publication_authorized': False}
    return vault, receipt


def _load_adapter(path):
    path = Path(path)
    require(sha(path.read_bytes()) == ADAPTER_PIN, 'Frozen evaluation adapter bytes differ')
    name = '_hotpot_postclosure_frozen_evaluation'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def score_after_release(*, approval_bytes, expected_approval_sha256, evidence,
                        extraction_receipt_bytes, score_review_bytes, expected_score_review_sha256,
                        archive_files, vault_loader, adapter_path):
    """Delegate to unchanged evaluate_after_release, with its full bind before IO.

    No scientific thresholds/statistics or condition selection live here. Root
    supplies a second independently trusted approval hash after vault creation.
    """
    require(is_hash(expected_score_review_sha256) and sha(score_review_bytes) == expected_score_review_sha256,
            'External scoring approval SHA absent/different')
    review = decode(score_review_bytes)
    require(review['decision'] == 'APPROVED_FOR_SCORING', 'Separate score release required')
    validated = validate_extraction_release(approval_bytes=approval_bytes,
        expected_approval_sha256=expected_approval_sha256, evidence=evidence)
    receipt = decode(extraction_receipt_bytes)
    require(receipt['schema'] == 'hotpot_postclosure_label_vault_receipt_v1'
            and receipt['status'] == 'COMPLETE_PRIVATE_LABEL_EXTRACTION'
            and receipt['extraction_approval_sha256'] == expected_approval_sha256
            and receipt['postclosure_source_sha256'] == source_sha()
            and receipt['evidence_sha256'] == validated['evidence_sha256']
            and type(receipt['selected_label_count']) is int and receipt['selected_label_count'] == N
            and is_hash(receipt['label_vault_sha256']), 'Label extraction receipt lineage differs')
    require(receipt['source_shard_sha256'] == [s['sha256'] for s in validated['descriptor']['parquet_files']]
            and receipt['source_split'] == 'train' and receipt['source_revision'] == REVISION
            and receipt['projected_columns'] == ['id', 'answer']
            and all(receipt[k] is False for k in ('supporting_facts_projected', 'validation_or_test_opened',
                'scoring_performed', 'scoring_authorized', 'publication_authorized')), 'Extraction scope receipt differs')
    # Root copies the original template exactly, changes only decision/vault pin,
    # and appends this linkage. No repinned scientific fields or unit selections.
    expected = dict(validated['template'])
    expected.update(decision='APPROVED_FOR_SCORING', label_vault_sha256=receipt['label_vault_sha256'],
        postclosure={'schema': 'hotpot_postclosure_scoring_binding_v1',
            'extraction_approval_sha256': expected_approval_sha256,
            'extraction_receipt_sha256': sha(extraction_receipt_bytes),
            'audit_sha256': sha(evidence['audit']), 'template_sha256': sha(evidence['template']),
            'postclosure_source_sha256': source_sha(),
            'unknown_preflight_components_acknowledged': validated['approval']['unknown_preflight_components_acknowledged']})
    require(canonical(review) == canonical(expected), 'Scoring release changed audited metadata or lacks extraction lineage')
    adapter = _load_adapter(adapter_path)
    def guarded_loader(request):
        require(request['archive_review_sha256'] == expected_score_review_sha256
                and request['label_vault_sha256'] == receipt['label_vault_sha256']
                and request['raw_question_ids'] == validated['ids'], 'Adapter label request differs')
        require(source_sha() == validated['approval']['postclosure_source_sha256'], 'Wrapper source changed before label IO')
        raw = vault_loader(request)
        require(sha(raw) == receipt['label_vault_sha256'], 'Label vault bytes differ')
        return raw
    return adapter.evaluate_after_release(label_loader=guarded_loader,
        selection_manifest_bytes=evidence['selection_manifest'], files=archive_files,
        review_bytes=score_review_bytes, expected_review_sha256=expected_score_review_sha256,
        expected_bindings=validated['template']['bindings'])


def write_new(path, raw):
    """Private no-overwrite file; a partial file/absent receipt is not completion."""
    require(type(raw) is bytes, 'Output bytes required')
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
