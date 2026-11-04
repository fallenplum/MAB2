"""Build a label-free G2 design and frozen train-only question representation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import time
import uuid

RUN_ID = 'g2_v1_20261104'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ENCODER_WEIGHT_SHA = '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'
SEEDS = [2701, 2702, 2703, 2704, 2705]
SPLIT_DOMAIN = 'graph_credit/G2/split/v1'
ANCHOR_DOMAIN = 'graph_credit/G2/anchor/v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as handle:
        handle.write((canonical(value) + '\n').encode('utf-8'))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_questions(path, expected_count):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows) != expected_count or len({row['id'] for row in rows}) != expected_count:
        raise ValueError('Question count or uniqueness differs from the contract')
    forbidden = {'answer', 'label', 'gold', 'target', 'solution', 'quality'}
    for row in rows:
        if forbidden.intersection(row) or not isinstance(row.get('question'), str):
            raise ValueError('A question contains forbidden quality information')
    return rows


def payload_sha(row):
    return digest({'id': row['id'], 'question': row['question'], 'context': row.get('context', '')})


def mask_id(mask):
    return ''.join(str(bit) for bit in mask)


def question_design(qid):
    requests, contrasts, anchors = [], [], []
    previous_anchor = None
    for family, threshold in [('bernoulli_025', 64), ('bernoulli_075', 192)]:
        anchor = [int(bytes.fromhex(digest([ANCHOR_DOMAIN, qid, family, edge]))[0] < threshold) for edge in range(12)]
        anchors.append({'anchor_id': family, 'mask_id': mask_id(anchor), 'mask': anchor})
        requests.append({'mask': anchor, 'reference_mask': previous_anchor,
                         'aliases': [{'anchor_id': family, 'kind': 'anchor'}]})
        for edge in range(12):
            flipped = list(anchor)
            flipped[edge] = 1 - flipped[edge]
            requests.append({'mask': flipped, 'reference_mask': anchor,
                             'aliases': [{'anchor_id': family, 'kind': 'single_flip', 'edge': edge}]})
            off, on = (anchor, flipped) if anchor[edge] == 0 else (flipped, anchor)
            contrasts.append({'question_id': qid, 'anchor_id': family, 'edge': edge,
                              'mask0_id': mask_id(off), 'mask1_id': mask_id(on)})
        previous_anchor = anchor
    return {'question_id': qid, 'anchors': anchors, 'mask_requests': requests, 'contrasts': contrasts}


def build_design(root, review_config):
    root = Path(root).resolve()
    accepted = json.loads((root / 'artifacts/audit/INDEPENDENT_G1_ACCEPTANCE.json').read_text(encoding='utf-8'))
    if accepted.get('status') != 'ACCEPTED_WITH_DOCUMENTED_LIMITATIONS' or accepted.get('scientific_gate') != 'PASS':
        raise ValueError('G1 independent gate must pass before G2 preparation')
    original = json.loads((root / 'results/pilot/motivation_v3_20261103/shard_000/frozen_plan.json').read_text(encoding='utf-8'))
    questions_path = root / 'data/derived/calibration_questions.jsonl'
    questions = read_questions(questions_path, 400)
    ordered = sorted(questions, key=lambda row: digest([SPLIT_DOMAIN, row['id']]))
    partitions = {row['id']: ('train' if index < 200 else 'tune' if index < 280 else 'test')
                  for index, row in enumerate(ordered)}
    if set(partitions).intersection(original['question_ids']):
        raise ValueError('G2 calibration overlaps G1 motivation')
    for relative, expected in original['source_manifest'].items():
        if file_sha(root / relative) != expected['sha256']:
            raise ValueError('G1 frozen source changed ' + relative)
    designs = {row['id']: question_design(row['id']) for row in ordered}
    masks = {mask_id(request['mask']): request['mask'] for design in designs.values() for request in design['mask_requests']}
    units = [{'question_id': row['id'], 'seed': seed, 'partition': partitions[row['id']],
              'mask_requests': designs[row['id']]['mask_requests']}
             for row in ordered for seed in SEEDS]
    qualification_path = root / 'data/derived/qualification_questions.jsonl'
    qualification_questions = read_questions(qualification_path, 16)
    qualification_row = min(qualification_questions, key=lambda row: digest(['graph_credit/G2/qualification/v1', row['id']]))
    config = json.loads(json.dumps(review_config))
    config['feature_contract']['revision'] = ENCODER_REVISION
    config['split']['sizes'] = {'train': 200, 'tune': 80, 'test': 120}
    actual_outputs = sum(len({mask_id(request['mask']) for request in unit['mask_requests']}) for unit in units)
    plan = {
        'schema': 'graph_credit_g2_plan_v1', 'run_id': RUN_ID, 'status': 'DESIGN_DRAFT_NO_FORMAL_GENERATION',
        'created_unix': time.time(), 'labels_read': False, 'scientific_completion': False,
        'output_dir': 'results/g2/' + RUN_ID, 'source_manifest': original['source_manifest'],
        'source_manifest_sha256': original['source_manifest_sha256'],
        'pipeline_source_manifest': {},
        'model': {'path': 'models/Qwen2.5-3B-Instruct', 'identity': original['model_identity'],
                  'manifest': original['model_manifest']},
        'graph': original['config']['graph'], 'protocol_hash': original['protocol_hash'],
        'questions': {'path': 'data/derived/calibration_questions.jsonl', 'sha256': file_sha(questions_path),
                      'payload_sha256': {row['id']: payload_sha(row) for row in ordered}},
        'partitions': partitions, 'designs': designs, 'masks': masks, 'units': units,
        'statistical_contract': config, 'expected_outputs': actual_outputs, 'expected_units': len(units),
        'expected_credit_targets': 400 * 2 * 12, 'expected_seed_blocks': 400 * 2 * 12 * 5,
        'formal_call_upper_bound': 124000,
        'qualification': {
            'questions': {'path': 'data/derived/qualification_questions.jsonl', 'sha256': file_sha(qualification_path),
                          'payload_sha256': {row['id']: payload_sha(row) for row in qualification_questions}},
            'units': [{'question_id': qualification_row['id'], 'seed': 2700, 'partition': 'qualification',
                       'mask_requests': question_design(qualification_row['id'])['mask_requests']}],
            'independent_full_replay_request_indices': [0, 1, 13],
            'worker_ids': ['gpu' + str(index) for index in range(8)],
            'prerequisites': [
                {'path': relative, 'sha256': file_sha(root / relative)} for relative in [
                    'results/pilot/s0_v3_20261103/QUALIFICATION.json',
                    'results/pilot/message_sanity_v3_20261103/MESSAGE_SANITY.json']],
            'prior_s0_receipt': 'results/pilot/s0_v3_20261103/QUALIFICATION.json',
            'prior_s0_receipt_sha256': file_sha(root / 'results/pilot/s0_v3_20261103/QUALIFICATION.json'),
        },
        'features': {'path': 'artifacts/private/g2/' + RUN_ID + '/question_features.npz',
                     'receipt': 'artifacts/private/g2/' + RUN_ID + '/FEATURES_FROZEN.json'},
        'label_release': {'training_after': 'GLOBAL_OUTPUTS_FROZEN', 'heldout_after': 'PREDICTIONS_FROZEN',
                          'train': 'data/derived/g2_v1/train_labels.jsonl',
                          'tune': 'data/derived/g2_v1/tune_labels.jsonl',
                          'test': 'data/derived/g2_v1/heldout_labels.jsonl'},
    }
    if actual_outputs > 52000:
        raise ValueError('Graph output budget cap exceeded')
    return plan


def copy_encoder(root, source):
    root, source = Path(root), Path(source)
    destination = root / 'models/sentence-transformers--all-MiniLM-L6-v2'
    expected = ENCODER_WEIGHT_SHA
    if file_sha(source / 'model.safetensors') != expected:
        raise ValueError('Existing encoder differs from the official pinned weight SHA')
    files = [path for path in source.rglob('*') if path.is_file() and '.cache' not in path.relative_to(source).parts]
    manifest = {}
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or file_sha(target) != file_sha(path):
            shutil.copyfile(path, target)
        manifest[relative.as_posix()] = {'sha256': file_sha(target), 'bytes': target.stat().st_size}
    return {'path': destination.relative_to(root).as_posix(), 'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
            'revision': ENCODER_REVISION, 'manifest': manifest, 'declared_license': 'apache-2.0',
            'source_url': 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/' + ENCODER_REVISION,
            'weight_source_url': 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/' + ENCODER_REVISION + '/model.safetensors'}


def encode_features(root, plan):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.decomposition import PCA

    root = Path(root)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261104)
    torch.use_deterministic_algorithms(True)
    encoder = plan['encoder']
    for relative, expected in encoder['manifest'].items():
        if file_sha(root / encoder['path'] / relative) != expected['sha256']:
            raise ValueError('Encoder bytes changed')
    tokenizer = AutoTokenizer.from_pretrained(root / encoder['path'], local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(root / encoder['path'], local_files_only=True, trust_remote_code=False).cpu().eval()
    questions = {row['id']: row for row in read_questions(root / plan['questions']['path'], 400)}
    qids = list(plan['partitions'])
    matrices = []
    with torch.inference_mode():
        for start in range(0, len(qids), 32):
            texts = [questions[qid]['question'] for qid in qids[start:start + 32]]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            outputs = model(**tokens).last_hidden_state
            attention = tokens['attention_mask'].unsqueeze(-1).to(outputs.dtype)
            pooled = (outputs * attention).sum(1) / attention.sum(1).clamp(min=1e-9)
            matrices.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    embeddings = np.concatenate(matrices).astype(np.float64)
    train = np.asarray([plan['partitions'][qid] == 'train' for qid in qids])
    pca = PCA(n_components=8, svd_solver='full', whiten=False)
    pca.fit(embeddings[train])
    phi = np.column_stack([pca.transform(embeddings), np.ones(400)])
    pairs = list(itertools.combinations(range(12), 2))
    design_blocks = []
    for index, qid in enumerate(qids):
        if not train[index]:
            continue
        seen = set()
        rows = []
        for request in plan['designs'][qid]['mask_requests']:
            mask = request['mask']
            key = mask_id(mask)
            if key in seen:
                continue
            seen.add(key)
            h = np.asarray([1, *mask, *(mask[left] * mask[right] for left, right in pairs)], dtype=np.float64)
            rows.append(np.kron(phi[index], h))
        design_blocks.append(np.asarray(rows) / np.sqrt(len(rows)))
    design = np.vstack(design_blocks)
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int((singular > tolerance).sum())
    if rank != 711:
        raise ValueError('Unlabeled train design does not span the declared 711 features')
    target = root / plan['features']['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, question_ids=np.asarray(qids), embeddings=embeddings, phi=phi,
                 pca_mean=pca.mean_, pca_components=pca.components_, explained_variance=pca.explained_variance_)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    receipt = {'schema': 'graph_credit_g2_features_v1', 'status': 'PASS_LABEL_FREE', 'labels_read': False,
               'created_unix': time.time(), 'feature_sha256': file_sha(target),
               'train_ids_sha256': digest([qid for qid in qids if plan['partitions'][qid] == 'train']),
               'pca_fit_questions': 200, 'transformed_questions': 400, 'dimensions': 9,
               'unlabeled_design_shape': list(design.shape), 'unlabeled_design_rank': rank,
               'unlabeled_design_smallest_singular': float(singular[-1]),
               'encoder': encoder, 'feature_contract': plan['statistical_contract']['feature_contract'],
               'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scikit-learn', 'torch', 'transformers']}}
    atomic_json(root / plan['features']['receipt'], receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['design', 'features'])
    parser.add_argument('--review-config', type=Path)
    parser.add_argument('--encoder-source', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.mode == 'design':
        review = json.loads(args.review_config.read_text(encoding='utf-8'))['approved_config']
        plan = build_design(args.root, review)
        plan['encoder'] = copy_encoder(args.root, args.encoder_source)
        path = args.root / 'artifacts/private/g2' / RUN_ID / 'DESIGN_DRAFT.json'
        atomic_json(path, plan)
        print(canonical({'status': plan['status'], 'plan': str(path), 'outputs': plan['expected_outputs'], 'units': plan['expected_units']}))
    else:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        receipt = encode_features(args.root, plan)
        print(canonical({key: receipt[key] for key in ['status', 'unlabeled_design_rank', 'unlabeled_design_shape', 'pca_fit_questions']}))


if __name__ == '__main__':
    main()
