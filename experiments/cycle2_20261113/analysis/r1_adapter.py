"""Raw collector -> staged R1 analysis. Owner splitting is a separate CLI mode.

No data are read on import. All generated scores, plans and arrays stay private;
only the explicit public aggregate is suitable for the repository.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Optional

import numpy as np
import r1_policy as policy

SCORER_SHA256 = 'da315ade4be7362ca81fc7ed41252cdccaa9271dc32f9a20531356587bb5f37e'
NODE_ORDER = tuple(f'r{r}:w{w}' for r in (1, 2) for w in range(4)) + ('aggregate',)
INTEGER_COSTS = ('calls', 'input_tokens', 'output_tokens', 'received_message_tokens', 'self_state_tokens')
ADDITIVE_COSTS = INTEGER_COSTS + ('model_elapsed_seconds', 'gpu_seconds')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key')
        out[key] = value
    return out


def parse(raw):
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object)


def read_pinned(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, 'Artifact byte hash mismatch')
    return parse(raw)


def safe(root, rel):
    require(isinstance(rel, str) and rel and '\\' not in rel and ':' not in rel,
            'Expected relative POSIX artifact path')
    require(all(x not in ('', '.', '..') for x in rel.split('/')), 'Unsafe artifact path')
    root = Path(root).resolve()
    target = root.joinpath(*rel.split('/'))
    require(target.resolve().is_relative_to(root), 'Artifact escapes root')
    cur = root
    for part in rel.split('/'):
        cur /= part
        require(not cur.is_symlink(), 'Symlink evidence is not accepted')
    return target


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical(value) + '\n').encode('utf-8')
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return sha(raw)


def _jsonl(raw):
    require(raw.endswith(b'\n'), 'Truncated JSONL')
    require(all(line.strip() for line in raw.splitlines()), 'Blank JSONL row')
    return [parse(line) for line in raw.splitlines()]


def validate_plan(plan, science_specs):
    require(plan.get('schema') == 'mab2_cycle2_collection_plan_v1' and plan.get('status') == 'FROZEN'
            and plan.get('labels_read') is False, 'Unfrozen/non-label-free collection plan')
    require(plan.get('scientific_protocol_sha256') == policy.PROTOCOL_SHA256, 'Scientific protocol binding differs')
    require(len(plan['episodes']) == 8, 'Wrong episode count')
    for ep, spec in zip(plan['episodes'], science_specs):
        require((ep['episode'], ep['seed'], ep['permutation'], ep['complement']) ==
                (spec.episode, spec.seed, [(j + spec.shift) % 4 for j in range(4)], bool(spec.complement)),
                'Episode/seed/role permutation binding differs')
        menu = ep['menu']
        require(len(menu) == 14 and menu[0]['mask'] == list(spec.A) and menu[7]['mask'] == list(spec.B),
                'Episode backgrounds differ')
        for background, mask in [('A', spec.A), ('B', spec.B)]:
            rows = [r for r in menu if r['background'] == background]
            require(len(rows) == 7 and sum(r['action'] == 'stay' for r in rows) == 1, 'Incomplete graph menu')
            toggles = [r for r in rows if r['action'] == 'toggle']
            require(sorted(r['edge'] for r in toggles) == list(spec.editable), 'Physical edge menu mismatch')
            for r in toggles:
                target = list(mask); target[r['edge']] ^= 1
                require(r['mask'] == target, 'Toggle metadata/mask disagree')
    units = plan['units']
    require(len(units) == 390 and len({u['unit_id'] for u in units}) == 390
            and len({u['question_id'] for u in units}) == 390, 'Require complete 390 disjoint units/questions')
    worker_ids = {w['index'] for w in plan['workers']}
    for u in units:
        require(u['worker_index'] in worker_ids, 'Unknown assigned worker')
        if u['phase'] == 'qualification':
            require(u['seed'] == 3799, 'Qualification seed differs')
        else:
            require(u['phase'] in ('I', 'C', 'S') and u['episode'] in range(8)
                    and u['seed'] == science_specs[u['episode']].seed, 'Wrong scientific unit role/seed')
    require(sum(u['phase'] == 'qualification' for u in units) == 6, 'Qualification coverage differs')
    for e in range(8):
        for block, n in [('I', 12), ('C', 12), ('S', 24)]:
            require(sum(u.get('episode') == e and u['phase'] == block for u in units) == n,
                    'Incomplete fixed I/C/S allocation')
    require(set(plan['questions']['payload_sha256']) == {u['question_id'] for u in units}, 'Question hash map differs')
    require(plan['graph']['rounds'] == 2 and plan['graph']['message_tokens'] == 128
            and plan['graph']['aggregate_tokens'] == 256 and plan['graph']['max_input_tokens'] == 8192,
            'Cost/operator configuration differs')


def scientific_unit_digest(plan):
    return digest(sorted([(u['question_id'], u['phase'], u.get('episode'), u['seed'])
                          for u in plan['units']], key=lambda x: x[0]))


def owner_split_labels(source, plan_path, execution_pins, protocol_path, data_receipt_path, selection_path, out_dir):
    """Separate data-owner process; full gold access is explicitly recorded."""
    pins = parse(Path(execution_pins).read_bytes())
    plan = read_pinned(plan_path, pins['plan_sha256'])
    protocol_raw = Path(protocol_path).read_bytes()
    specs = policy.specs_from_protocol_bytes(protocol_raw)
    data_binding = parse(protocol_raw)['data']
    original_receipt = read_pinned(data_receipt_path, data_binding['data_receipt_sha256'])
    selection = read_pinned(selection_path, data_binding['source_selection_sha256'])
    validate_plan(plan, specs)
    selected = {(u['question_id'], 'qualification' if u['phase'] == 'qualification' else u['block'],
                 None if u['phase'] == 'qualification' else u['episode'], u['seed']) for u in selection['units']}
    assigned = {(u['question_id'], u['phase'], u.get('episode'), u['seed']) for u in plan['units']}
    require(len(selection['units']) == len(selected) == 390 and selected == assigned,
            'Collection assignment differs from before-payload scientific selection')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    immutable_json(out / 'OWNER_ACCESS_STARTED.json', {
        'schema': 'mab2_r1_owner_label_access_v1', 'event': 'OWNER_WILL_READ_ALL_SELECTED_LABELS',
        'source_sha256_expected_from_data_receipt': original_receipt['labels_sha256'],
        'plan_sha256': pins['plan_sha256'], 'scientific_protocol_sha256': policy.PROTOCOL_SHA256,
        'policy_feedback_access_granted_by_owner_read': False, 'time_unix': time.time()})
    raw = Path(source).read_bytes()
    require(sha(raw) == original_receipt['labels_sha256'], 'Owner source labels differ from original pinned data receipt')
    rows = _jsonl(raw)
    labels = {row['id']: row for row in rows}
    require(len(labels) == len(rows) == 390 and set(labels) == {u['question_id'] for u in plan['units']},
            'Owner labels do not exactly match frozen selected IDs')
    phases = {}
    for phase in ('I', 'C', 'S'):
        ids = [u['question_id'] for u in plan['units'] if u['phase'] == phase]
        content = ('\n'.join(canonical(labels[q]) for q in ids) + '\n').encode('utf-8')
        name = phase + '.labels.jsonl'
        with (out / name).open('xb') as f:
            f.write(content); f.flush(); os.fsync(f.fileno())
        phases[phase] = {'path': name, 'sha256': sha(content), 'count': len(ids), 'ids_sha256': digest(sorted(ids))}
    descriptor = {'schema': 'mab2_r1_phase_labels_v1', 'scientific_protocol_sha256': policy.PROTOCOL_SHA256,
                  'scientific_unit_digest': scientific_unit_digest(plan),
                  'owner_collection_plan_sha256': pins['plan_sha256'], 'source_labels_sha256': sha(raw),
                  'source_selection_sha256': data_binding['source_selection_sha256'],
                  'source_data_receipt_sha256': data_binding['data_receipt_sha256'],
                  'owner_read_all_390_labels': True, 'qualification_labels_in_analysis': False,
                  'owner_access_is_not_policy_access': True, 'phases': phases}
    pin = immutable_json(out / 'PHASE_LABELS.json', descriptor)
    immutable_json(out / 'OWNER_SPLIT_COMPLETE.json', {'status': 'COMPLETE_OWNER_ONLY', 'descriptor_sha256': pin,
                   'phase_counts': {p: x['count'] for p, x in phases.items()}, 'gold_displayed': False})
    return {'status': 'COMPLETE_OWNER_ONLY', 'descriptor_sha256': pin,
            'phase_counts': {p: x['count'] for p, x in phases.items()}, 'gold_displayed': False}


def legacy_float_sum(values):
    """Emulate producer Python3.11 sum in native node order, not Python3.12+ sum."""
    total = 0
    for value in values:
        total = total + value
    return total


def sum_cost(nodes):
    values = [nodes[name] for name in NODE_ORDER if name in nodes]
    return {'calls': len(values),
            'input_tokens': sum(x['generation']['input_tokens'] for x in values),
            'output_tokens': sum(x['generation']['output_tokens'] for x in values),
            'received_message_tokens': sum(x['received_message_tokens'] for x in values),
            'self_state_tokens': sum(x['self_state_tokens'] for x in values),
            'model_elapsed_seconds': legacy_float_sum(x['generation']['elapsed_seconds'] for x in values),
            'gpu_seconds': legacy_float_sum(x['generation']['gpu_seconds'] for x in values),
            'peak_allocated_bytes': max((x['generation']['gpu_peak_allocated_bytes'] for x in values), default=0)}


def compare_cost(actual, expected):
    require(set(actual) == set(expected), 'Cost fields differ')
    for key in actual:
        require(isinstance(actual[key], (int, float)) and math.isfinite(actual[key]) and actual[key] >= 0,
                'Invalid cost value')
        if key in INTEGER_COSTS or key == 'peak_allocated_bytes':
            require(actual[key] == expected[key], 'Integer graph cost mismatch')
        else:
            require(actual[key] == expected[key], 'Floating graph cost mismatch under producer-order arithmetic')


@dataclass(frozen=True)
class GraphValue:
    request: dict
    raw_aggregate_text: str
    logical: dict
    increment: dict
    node_call_ids: tuple[str, ...]
    node_call_tokens: tuple[tuple[str, int, int], ...]


def validate_graph(record, unit, request, plan, plan_sha, previous, success_calls):
    require(record.get('schema') == 'mab2_cycle2_graph_checkpoint_v1'
            and record.get('plan_sha256') == plan_sha and record.get('unit_id') == unit['unit_id']
            and record.get('request') == request and record.get('labels_read') is False
            and record.get('quality') is None, 'Graph checkpoint binding/scoring state mismatch')
    x = record['execution']
    require(digest(x) == record['execution_value_sha256'], 'Execution payload digest differs')
    require(x['question_id'] == unit['question_id'] and x['seed'] == unit['seed']
            and x['mask'] == request['mask'] and x['protocol_hash'] == plan['executor_protocol_hash']
            and x['question_fingerprint'] == plan['questions']['payload_sha256'][unit['question_id']],
            'Graph question/seed/mask/protocol mismatch')
    require(set(x['nodes']) == set(NODE_ORDER), 'Nine-node coverage differs')
    base = previous.get(request.get('base'))
    if base is None:
        closure = set(NODE_ORDER)
    else:
        changed = [e for e in range(12) if base['mask'][e] != x['mask'][e]]
        closure = {f'r2:w{policy.EDGES[e][1]}' for e in changed}
        if closure:
            closure.add('aggregate')
    require(x['closure'] == sorted(closure), 'Two-round closure mismatch')
    integrity_keys = ('node', 'input_fingerprint', 'seed', 'messages', 'generation', 'wire_message',
                      'parsed', 'message_metadata', 'received_message_tokens', 'self_state_tokens')
    for name in NODE_ORDER:
        node = x['nodes'][name]; generation = node['generation']
        require(node['node'] == name and digest({k: node[k] for k in integrity_keys}) == node['output_digest'],
                'Node integrity digest differs')
        expected_seed = int(digest([unit['question_id'], unit['seed'], name])[:16], 16) % (2**31 - 1)
        cap = 256 if name == 'aggregate' else 128
        require(node['seed'] == expected_seed and node['input_fingerprint'] == digest({
            'question': x['question_fingerprint'], 'node': name, 'messages': node['messages'],
            'seed': expected_seed, 'protocol': x['protocol_hash'], 'max_new_tokens': cap}), 'Node prompt/seed binding differs')
        require(type(generation['input_tokens']) is int and 0 <= generation['input_tokens'] <= 8192
                and type(generation['output_tokens']) is int and 0 <= generation['output_tokens'] <= cap,
                'Actual node token counts exceed frozen quote')
        require(node['reused'] == (name not in closure), 'Incorrect node cache status')
        if name not in closure:
            require(node['source_execution_id'] == base['execution_id']
                    and node['output_digest'] == base['nodes'][name]['output_digest'], 'Cache source differs')
        else:
            require(node['source_execution_id'] is None, 'Fresh node has cache source')
        call = success_calls.get(generation['call_id'])
        require(call is not None and call['seed'] == node['seed'] and call['prompt_sha256'] == digest(node['messages'])
                and call['plan_sha256'] == plan_sha,
                'Node call lacks pre-journalled successful evidence')
        for key in ('input_tokens', 'output_tokens', 'gpu_seconds', 'gpu_peak_allocated_bytes'):
            require(call[key] == generation[key], 'Node call cost differs from journal')
    require(x['answer'] == x['nodes']['aggregate']['parsed']['answer'], 'Parsed answer binding differs')
    logical = sum_cost(x['nodes'])
    increment = sum_cost({n: x['nodes'][n] for n in NODE_ORDER if n in closure})
    compare_cost(x['logical_service_cost'], logical); compare_cost(x['physical_cost'], increment)
    require(x['cache_reuse_count'] == 9 - len(closure), 'Cache reuse count differs')
    return x, GraphValue(request, x['nodes']['aggregate']['generation']['text'], logical, increment,
                         tuple(x['nodes'][n]['generation']['call_id'] for n in NODE_ORDER),
                         tuple((x['nodes'][n]['generation']['call_id'], x['nodes'][n]['generation']['input_tokens'],
                                x['nodes'][n]['generation']['output_tokens']) for n in NODE_ORDER))


def read_worker_costs(root, worker, file_pins, unit_plan_pins, residence_plan_pins):
    def rows(name):
        path = safe(root, f'worker_{worker}/{name}'); raw = path.read_bytes()
        file_pins[f'worker_{worker}/{name}'] = sha(raw)
        return _jsonl(raw)
    journal, native, residence = rows('call_journal.jsonl'), rows('physical_calls.jsonl'), rows('resident_wall.jsonl')
    starts, terminal, success = {}, {}, {}
    for row in journal:
        event, key = row['event'], row['reservation_id']
        require(row.get('unit_id') in unit_plan_pins and row.get('plan_sha256') == unit_plan_pins[row['unit_id']],
                'Call ledger unit/phase uses an unapproved plan')
        target = starts if event == 'CALL_START' else terminal
        require(event in ('CALL_START', 'CALL_SUCCESS', 'CALL_FAILED') and key not in target, 'Invalid/duplicate call journal')
        target[key] = row
    require(set(terminal) <= set(starts), 'Call terminal lacks reservation')
    for key, row in terminal.items():
        require(all(row.get(field) == starts[key].get(field) for field in ('plan_sha256', 'unit_id', 'request_name')),
                'Call reservation/terminal plan, unit or request differs')
        if row['event'] == 'CALL_SUCCESS':
            cid = row['call_id']; require(cid not in success, 'Duplicate successful call ID')
            success[cid] = {**row, 'seed': starts[key]['seed'], 'prompt_sha256': starts[key]['prompt_sha256']}
    native_success = {}; all_native = set()
    for row in native:
        cid = row['call_id']; require(cid not in all_native, 'Duplicate native call ID'); all_native.add(cid)
        if row['status'] == 'SUCCESS':
            native_success[cid] = row
            if cid in success:
                for field in ('input_tokens', 'output_tokens', 'gpu_seconds'):
                    require(row[field] == success[cid][field], 'Native/wrapper physical costs differ')
    require(set(success) <= set(native_success), 'Wrapper success lacks native physical ledger')
    rs, re = {}, {}
    for row in residence:
        target = rs if row['event'] == 'START' else re
        require(row['event'] in ('START', 'END') and row['attempt_id'] not in target, 'Invalid residence ledger')
        target[row['attempt_id']] = row
    require(set(rs) == set(re), 'Residence ledger not closed; no analysis of running job')
    require(all(rs[k].get('phase') in residence_plan_pins
                and rs[k].get('plan_sha256') == residence_plan_pins[rs[k]['phase']] for k in rs),
            'Residence ledger phase uses an unapproved plan')
    require(all(rs[k]['plan_sha256'] == re[k]['plan_sha256'] for k in rs), 'Residence start/end plan differs')
    require(all(math.isfinite(x['resident_wall_seconds']) and x['resident_wall_seconds'] >= 0 for x in re.values()), 'Invalid residence cost')
    pending = len(set(starts) - set(terminal)); failed = sum(x['event'] == 'CALL_FAILED' for x in terminal.values())
    return success, {'worker': worker, 'attempted_calls': len(starts), 'success_calls': len(native_success),
                     'failed_calls': failed, 'pending_calls': pending,
                     'input_tokens': sum(x['input_tokens'] for x in native_success.values()),
                     'output_tokens': sum(x['output_tokens'] for x in native_success.values()),
                     'known_cuda_event_seconds': sum(x['gpu_seconds'] for x in native_success.values()),
                     'resident_wall_gpu_seconds': sum(x['resident_wall_seconds'] for x in re.values()),
                     'qualification_resident_wall_seconds': sum(re[k]['resident_wall_seconds'] for k in rs if rs[k]['phase'] == 'qualification'),
                     'unmeasured_failure_gpu_seconds': None if pending or failed else 0,
                     'unmeasured_failure_tokens': None if pending or failed else 0,
                     'successful_orphan_calls': len(set(native_success) - set(success))}


def require_attempt_caps(worker_costs, workers, budget):
    caps = {w['index']: w['max_call_attempts'] for w in workers}
    require(all(c['worker'] in caps and c['attempted_calls'] <= caps[c['worker']] for c in worker_costs),
            'Worker call-attempt budget exceeded')
    require(sum(c['attempted_calls'] for c in worker_costs) <= budget['max_call_attempts'],
            'Global call-attempt budget exceeded')


@dataclass
class Archive:
    plan: dict
    plan_sha: str
    root: Path
    specs: tuple
    graphs: dict
    unit_order: dict
    file_pins: dict
    physical_costs: dict
    scientific_data_bindings: dict


def validate_resource_amendment(path, data_root, plan, plan_sha, qa):
    raw = Path(path).read_bytes(); cert = parse(raw)
    require(cert['schema'] == 'mab2_cycle2_resource_only_amendment_v1' and cert['status'] == 'FROZEN'
            and cert['scientific_protocol_sha256'] == policy.PROTOCOL_SHA256
            and cert['formal_rounds_started_at_amendment'] == 0
            and cert['scientific_outcomes_read'] is False
            and cert['old_qualification_costs_remain_charged'] is True, 'Invalid resource-only amendment')
    bound = {}
    for key in ('old_plan', 'new_plan', 'qualification_evidence', 'qualification_acceptance_v1'):
        d = cert[key]; bound[key] = read_pinned(safe(data_root, d['path']), d['sha256'])
    old, new = bound['old_plan'], bound['new_plan']
    require(new == plan and cert['new_plan']['sha256'] == plan_sha, 'Amendment new plan differs')
    old_norm, new_norm = copy.deepcopy(old), copy.deepcopy(new)
    changes = []
    require(len(old['workers']) == len(new['workers']), 'Resource amendment changes workers')
    old_caps, new_caps = [], []
    for index in range(len(old['workers'])):
        a = old_norm['workers'][index].pop('resident_wall_cap_seconds')
        b = new_norm['workers'][index].pop('resident_wall_cap_seconds')
        old_caps.append(a); new_caps.append(b)
        if a != b:
            changes.append({'path': f'workers[{index}].resident_wall_cap_seconds', 'old': a, 'new': b})
    require(old_norm == new_norm and digest(old_norm) == cert['unchanged_fields_sha256'],
            'Amendment changes science, data, assignment, source or any non-cap field')
    require(changes == cert['changes'] and sum(old_caps) == sum(new_caps) == cert['both_total_seconds'] == 86400,
            'Resource amendment does not preserve exact total budget')
    require(len(cert['qualification_resident_seconds']) == len(new_caps)
            and all(0 <= used <= cap for used, cap in zip(cert['qualification_resident_seconds'], new_caps)),
            'New cap cannot cover already spent qualification')
    old_pin = cert['old_plan']['sha256']; evidence = bound['qualification_evidence']; acceptance = bound['qualification_acceptance_v1']
    require(evidence['schema'] == 'mab2_cycle2_qualification_evidence_audit_v1'
            and evidence['status'] == 'EVIDENCE_CHECKS_COMPLETE_REVIEW_REQUIRED'
            and evidence['plan_sha256'] == old_pin and evidence['runner_sha256'] == plan['runner_sha256']
            and evidence['executor_protocol_hash'] == plan['executor_protocol_hash'], 'Qualification audit is not bound to old operator')
    require(acceptance['status'] == 'PASS' and acceptance['plan_sha256'] == old_pin
            and acceptance['formal_admitted'] is False
            and acceptance['evidence_sha256'] == cert['qualification_evidence']['sha256']
            and acceptance['worker_reports'] == evidence['worker_report_hashes'] == qa['worker_reports'],
            'Old qualification acceptance/evidence/report pins differ')
    require(cert['phase_plan_binding'] == {'qualification': old_pin, 'I_C_S': plan_sha}, 'Wrong amendment phase binding')
    require(qa.get('resource_amendment_sha256', qa.get('amendment_sha256')) == sha(raw),
            'New qualification admission does not bind amendment certificate')
    return {'certificate_sha256': sha(raw), 'old_qualification_plan_sha256': old_pin,
            'new_formal_plan_sha256': plan_sha, 'new_worker_caps_seconds': new_caps,
            'qualification_resident_seconds': cert['qualification_resident_seconds'],
            'unchanged_fields_sha256': cert['unchanged_fields_sha256']}


def audit_archive(plan_path, execution_pins_path, protocol_path, run_root, *, resource_amendment=None, data_root=None):
    pins = parse(Path(execution_pins_path).read_bytes())
    plan = read_pinned(plan_path, pins['plan_sha256'])
    protocol_raw = Path(protocol_path).read_bytes()
    specs = policy.specs_from_protocol_bytes(protocol_raw)
    scientific_data_bindings = {k: parse(protocol_raw)['data'][k] for k in ('source_selection_sha256', 'data_receipt_sha256')}
    validate_plan(plan, specs)
    require(pins['scientific_protocol_sha256'] == policy.PROTOCOL_SHA256
            and pins['runner_sha256'] == plan['runner_sha256']
            and pins['executor_protocol_hash'] == plan['executor_protocol_hash'], 'Execution receipt binding differs')
    require(pins['source_manifest'] == plan['source_manifest'], 'Frozen execution source manifest differs')
    root = Path(run_root).resolve(); file_pins = {}; graphs = {}; worker_costs = []
    qualification_path = root / 'QUALIFICATION_ACCEPTED.json'
    qualification_raw = qualification_path.read_bytes(); qa = parse(qualification_raw)
    file_pins['QUALIFICATION_ACCEPTED.json'] = sha(qualification_raw)
    require(qa['status'] == 'PASS' and qa['plan_sha256'] == pins['plan_sha256']
            and qa['executor_protocol_hash'] == plan['executor_protocol_hash'], 'Qualification not bound/accepted')
    amendment = None
    if resource_amendment is not None:
        require(data_root is not None, 'Resource amendment needs explicit data root')
        amendment = validate_resource_amendment(resource_amendment, data_root, plan, pins['plan_sha256'], qa)
    qualification_plan_sha = amendment['old_qualification_plan_sha256'] if amendment else pins['plan_sha256']
    residence_plan_pins = {'qualification': qualification_plan_sha, 'development': pins['plan_sha256']}
    for w in plan['workers']:
        wi = w['index']
        unit_plan_pins = {u['unit_id']: qualification_plan_sha if u['phase'] == 'qualification' else pins['plan_sha256']
                          for u in plan['units'] if u['worker_index'] == wi}
        success, costs = read_worker_costs(root, wi, file_pins, unit_plan_pins, residence_plan_pins)
        worker_costs.append(costs)
        require_attempt_caps(worker_costs, plan['workers'], plan['budget'])
        if amendment:
            require(costs['qualification_resident_wall_seconds'] == amendment['qualification_resident_seconds'][wi],
                    'Old qualification expense was altered or omitted')
        require(costs['resident_wall_gpu_seconds'] <= w['resident_wall_cap_seconds'], 'Worker occupancy budget exceeded')
        for phase_name, selected in [('QUALIFICATION', lambda u: u['phase'] == 'qualification'),
                                     ('DEVELOPMENT', lambda u: u['phase'] != 'qualification')]:
            phase_plan_sha = qualification_plan_sha if phase_name == 'QUALIFICATION' else pins['plan_sha256']
            rel = f'worker_{wi}/{phase_name}_COMPLETE.json'; raw = safe(root, rel).read_bytes(); marker = parse(raw)
            file_pins[rel] = sha(raw)
            if phase_name == 'QUALIFICATION':
                require(qa['worker_reports'][str(wi)] == sha(raw), 'Qualification acceptance hash differs')
            require(marker['status'] == 'COMPLETE_UNSCORED' and marker['labels_read'] is False
                    and marker['plan_sha256'] == phase_plan_sha
                    and marker['executor_protocol_hash'] == plan['executor_protocol_hash'], 'Unsealed worker phase')
            expected = [u for u in plan['units'] if u['worker_index'] == wi and selected(u)]
            completed = {x['unit_id']: x['sha256'] for x in marker['completed_units']}
            require(len(completed) == len(marker['completed_units']) and set(completed) == {u['unit_id'] for u in expected},
                    'Worker unit coverage differs')
            for unit in expected:
                stem = f"worker_{wi}/units/{unit['unit_id']}/"
                rel = stem + 'UNIT_COMPLETE.json'; raw = safe(root, rel).read_bytes(); receipt = parse(raw)
                require(sha(raw) == completed[unit['unit_id']], 'Unit hash differs from phase manifest')
                file_pins[rel] = sha(raw)
                require(receipt['unit'] == unit and receipt['plan_sha256'] == phase_plan_sha
                        and receipt['labels_read'] is False and receipt['scored'] is False,
                        'Completed unit metadata differs')
                menu = (plan['qualification_menu'] if unit['phase'] == 'qualification' else
                        plan['episodes'][unit['episode']]['menu'][:7] if unit['phase'] == 'I' else
                        plan['episodes'][unit['episode']]['menu'])
                require(set(receipt['graphs']) == {r['name'] for r in menu}, 'Unit graph coverage differs')
                previous = {}; found = {}
                for request in menu:
                    rel = stem + request['name'] + '.json'; raw = safe(root, rel).read_bytes()
                    require(sha(raw) == receipt['graphs'][request['name']], 'Graph bytes changed after unit freeze')
                    file_pins[rel] = sha(raw)
                    execution, graph = validate_graph(parse(raw), unit, request, plan, phase_plan_sha, previous, success)
                    previous[request['name']] = execution
                    found[request['name']] = graph
                    if request.get('replay_of'):
                        original = previous[request['replay_of']]
                        fields = ('input_fingerprint', 'wire_message', 'seed')
                        for node in NODE_ORDER:
                            require(all(execution['nodes'][node][f] == original['nodes'][node][f] for f in fields)
                                    and execution['nodes'][node]['generation']['text'] == original['nodes'][node]['generation']['text'],
                                    'Qualification full/cache replay differs')
                graphs[unit['unit_id']] = found
    require(len(graphs) == 390, 'Global unit coverage differs')
    require(sum(x['resident_wall_gpu_seconds'] for x in worker_costs) <= plan['budget']['resident_wall_gpu_seconds'],
            'Global collection occupancy reservation exceeded')
    unit_order = {(e, phase): tuple(u for u in plan['units'] if u.get('episode') == e and u['phase'] == phase)
                  for e in range(8) for phase in ('I', 'C', 'S')}
    return Archive(plan, pins['plan_sha256'], root, specs, graphs, unit_order, file_pins,
                   {'current_collection_workers': worker_costs,
                    'resource_only_amendment': amendment,
                    'current_collection_occupancy_GPUh': sum(x['resident_wall_gpu_seconds'] for x in worker_costs) / 3600,
                    'includes_current_run_qualification_failures_and_orphans': True,
                    'prior_runs_or_other_engineering_GPUh': None,
                    'entire_project_cost_complete': False}, scientific_data_bindings)


def load_scorer(path):
    require(file_sha(path) == SCORER_SHA256, 'Original GSM8K scorer bytes changed')
    spec = importlib.util.spec_from_file_location('mab2_r1_original_scorer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.VERSION == 'gsm8k_explicit_answer_decimal_v2' and callable(module.score_item), 'Original scorer API differs')
    return module.score_item


def load_phase_labels(archive, descriptor_path, descriptor_sha256, phase, private, preceding_freeze):
    descriptor_path = Path(descriptor_path)
    descriptor = read_pinned(descriptor_path, descriptor_sha256)
    require(descriptor['schema'] == 'mab2_r1_phase_labels_v1'
            and descriptor['scientific_protocol_sha256'] == policy.PROTOCOL_SHA256
            and descriptor['scientific_unit_digest'] == scientific_unit_digest(archive.plan),
            'Phase label descriptor not bound to scientific allocation')
    require(descriptor['source_selection_sha256'] == archive.scientific_data_bindings['source_selection_sha256']
            and descriptor['source_data_receipt_sha256'] == archive.scientific_data_bindings['data_receipt_sha256'],
            'Phase descriptor source data/selection binding differs')
    require(phase in ('I', 'C', 'S'), 'Qualification gold is not used in analysis')
    d = descriptor['phases'][phase]
    expected_ids = {u['question_id'] for u in archive.plan['units'] if u['phase'] == phase}
    require(d['count'] == len(expected_ids) and d['ids_sha256'] == digest(sorted(expected_ids)), 'Phase label IDs differ')
    require(Path(preceding_freeze['path']).is_file() and file_sha(preceding_freeze['path']) == preceding_freeze['sha256'],
            'Preceding phase has not been durably frozen')
    path = safe(descriptor_path.parent, d['path'])
    with path.open('rb') as f:
        identity = os.fstat(f.fileno())
        immutable_json(Path(private) / f'FIRST_{phase}_LABEL_ACCESS.json', {
            'schema': 'mab2_r1_first_phase_label_access_v1', 'event': 'HANDLE_OPENED_BEFORE_FIRST_READ',
            'scope': 'this_analysis_run_only_owner_split_access_recorded_separately',
            'phase': phase, 'expected_sha256': d['sha256'], 'descriptor_sha256': file_sha(descriptor_path),
            'preceding_freeze_sha256': preceding_freeze['sha256'], 'handle_size': identity.st_size,
            'handle_inode': identity.st_ino, 'handle_device': identity.st_dev, 'time_unix': time.time()})
        raw = f.read()
    require(sha(raw) == d['sha256'], 'Released phase label bytes changed')
    rows = _jsonl(raw); labels = {r['id']: r for r in rows}
    require(len(rows) == len(labels) == len(expected_ids) and set(labels) == expected_ids, 'Released phase label set differs')
    return labels


class ScoringStore:
    def __init__(self, archive, scorer):
        self.archive = archive; self.scorer = scorer
        self.phase = None; self.labels = {}; self.cache = {}; self.score_log = []

    def open_phase(self, phase, labels):
        require((self.phase, phase) in ((None, 'I'), ('I', 'C'), ('C', 'S')), 'Invalid scoring phase transition')
        self.phase = phase; self.labels = labels

    def find(self, unit, background, action):
        found = [g for g in self.archive.graphs[unit['unit_id']].values()
                 if g.request['background'] == background and
                 ((action is None and g.request['action'] == 'stay') or
                  (action is not None and g.request['action'] == 'toggle' and g.request['edge'] == action))]
        require(len(found) == 1, 'Physical edge/background does not identify one menu graph')
        return found[0]

    def score(self, unit, background, action):
        require(unit['phase'] == self.phase, 'Attempt to score outside currently released phase')
        graph = self.find(unit, background, action)
        key = (unit['unit_id'], graph.request['name'])
        if key not in self.cache:
            before = time.process_time()
            quality = float(self.scorer(graph.raw_aggregate_text, self.labels[unit['question_id']], task='gsm8k'))
            elapsed = time.process_time() - before
            require(quality in (0., 1.) and math.isfinite(quality), 'Original scorer returned invalid quality')
            self.cache[key] = (quality, elapsed)
            self.score_log.append({'phase': self.phase, 'unit_id': unit['unit_id'], 'request': graph.request['name'],
                                   'quality': quality, 'scoring_cpu_seconds': elapsed})
        return self.cache[key][0]

    def measured_score_cpu(self, unit, background, action):
        graph = self.find(unit, background, action)
        entry = self.cache.get((unit['unit_id'], graph.request['name']))
        return entry[1] if entry is not None else float('nan')


def _add_cost(a, b):
    return np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def _cost_vector(graph_cost, score_cpu):
    return np.array([graph_cost['input_tokens'], graph_cost['output_tokens'], graph_cost['calls'], 1., score_cpu], dtype=np.float64)


def replay_cost_arrays(archive, store):
    """All policy charges are logical; C base/S include all nine service nodes."""
    initial = np.zeros((8, 5)); c_base = np.zeros((8, 2, 5)); c_flip = np.zeros((8, 2, 6, 5)); service = np.zeros((8, 2, 7, 5))
    for spec in archive.specs:
        e = spec.episode
        for unit in archive.unit_order[(e, 'I')]:
            base = store.find(unit, 'A', None)
            initial[e] += _cost_vector(base.logical, store.measured_score_cpu(unit, 'A', None))
            for edge in spec.editable:
                graph = store.find(unit, 'A', edge)
                require(graph.increment['calls'] == 2, 'Single-edge increment not two calls')
                initial[e] += _cost_vector(graph.increment, store.measured_score_cpu(unit, 'A', edge))
        for ci, background in enumerate(('B', 'A')):
            for unit in archive.unit_order[(e, 'C')]:
                graph = store.find(unit, background, None)
                # B_stay physically reused A; the policy never purchased A on C.
                c_base[e, ci] += _cost_vector(graph.logical, store.measured_score_cpu(unit, background, None))
                for j, edge in enumerate(spec.editable):
                    graph = store.find(unit, background, edge)
                    require(graph.increment['calls'] == 2, 'Single-edge increment not two calls')
                    c_flip[e, ci, j] += _cost_vector(graph.increment, store.measured_score_cpu(unit, background, edge))
            for slot, action in enumerate((None,) + spec.editable):
                for unit in archive.unit_order[(e, 'S')]:
                    graph = store.find(unit, background, action)
                    # S is direct deployment, not a two-call cached menu toggle.
                    service[e, ci, slot] += _cost_vector(graph.logical, store.measured_score_cpu(unit, background, action))
    require(np.isfinite(initial).all() and np.isfinite(c_base).all() and np.isfinite(c_flip).all(),
            'I/C score or cost coverage incomplete')
    return {'I_cost': initial, 'C_base_cost': c_base, 'C_increment_cost': c_flip, 'S_cost': service}


def logical_policy_costs(actions, cost_arrays):
    rows = []; replica_costs = []
    grouped = {(condition, k, method): np.zeros((8, 5)) for condition in policy.CONDITIONS
               for k in policy.K_VALUES for method in policy.METHODS}
    for record in actions.records:
        p = record.plan; ci = policy.CONDITIONS.index(p.condition)
        spec = actions.purchases.specs[p.episode]
        slot = 0 if record.action is None else 1 + spec.editable.index(record.action)
        actual = cost_arrays['S_cost'][p.episode, ci, slot].copy()
        if p.method != 'current_stay':
            actual += cost_arrays['I_cost'][p.episode]
        if p.purchased:
            actual += cost_arrays['C_base_cost'][p.episode, ci]
            for edge in p.purchased:
                actual += cost_arrays['C_increment_cost'][p.episode, ci, spec.editable.index(edge)]
        require(np.isfinite(actual).all(), 'Chosen policy has missing actual cost/score coverage')
        cap = policy.conservative_budget(p.k, p.method)
        require(actual[0] + actual[1] <= cap['total_token_cap'] and actual[3] == cap['score_count_cap'],
                'Logical actual spend exceeds conservative cap or score quota differs')
        grouped[(p.condition, p.k, p.method)][p.episode] += actual / p.weight_denominator
        replica_costs.append({'plan_key': list(p.key), 'cost_vector': actual.tolist()})
    for (condition, k, method), values in grouped.items():
        rows.append({'condition': condition, 'k': k, 'method': method,
                     'mean_episode_actual_cost': values.mean(axis=0).tolist(),
                     'all_eight_episode_actual_cost': values.sum(axis=0).tolist(),
                     'episode_actual_costs': values.tolist(),
                     'conservative_cap_per_episode': policy.conservative_budget(k, method),
                     'same_actual_expenditure_claim': False})
    return rows, replica_costs


def export_IC_only(archive, store, I_delta, actions, action_path, private, code_identity):
    """Root-authorized R2 diagnostic input, frozen after C and before S release."""
    private = Path(private)
    require(store.phase == 'C' and not (private / 'FIRST_S_LABEL_ACCESS.json').exists(), 'IC export must precede S release')
    require(Path(action_path).is_file(), 'Action freeze missing before IC export')
    arrays = {'I_A_gain': I_delta.copy(), 'C_A_quality': np.zeros((8, 12, 7)),
              'C_B_quality': np.zeros((8, 12, 7)),
              'I_A_menu_tokens': np.zeros((8, 12), dtype=np.int64),
              'C_A_menu_tokens': np.zeros((8, 12), dtype=np.int64),
              'C_B_menu_tokens': np.zeros((8, 12), dtype=np.int64),
              'C_AB_joint_tokens': np.zeros((8, 12), dtype=np.int64),
              'C_AB_joint_calls': np.zeros((8, 12), dtype=np.int64),
              'I_qid_hash': np.empty((8, 12), dtype='U64'), 'C_qid_hash': np.empty((8, 12), dtype='U64'),
              'I_seed': np.zeros((8, 12), dtype=np.int64), 'C_seed': np.zeros((8, 12), dtype=np.int64),
              'editable_edges': np.array([s.editable for s in archive.specs], dtype=np.int64),
              'dirty': np.array([s.dirty for s in archive.specs], dtype=np.int64),
              'A_masks': np.array([s.A for s in archive.specs], dtype=np.int64),
              'B_masks': np.array([s.B for s in archive.specs], dtype=np.int64)}
    ic_units = set()
    for spec in archive.specs:
        for phase in ('I', 'C'):
            for i, unit in enumerate(archive.unit_order[(spec.episode, phase)]):
                ic_units.add(unit['unit_id'])
                arrays[phase + '_qid_hash'][spec.episode, i] = sha(unit['question_id'].encode('utf-8'))
                arrays[phase + '_seed'][spec.episode, i] = unit['seed']
                for background in (('A',) if phase == 'I' else ('A', 'B')):
                    base = store.find(unit, background, None)
                    total_tokens = base.logical['input_tokens'] + base.logical['output_tokens']
                    for slot, edge in enumerate(spec.editable, 1):
                        graph = store.find(unit, background, edge)
                        total_tokens += graph.increment['input_tokens'] + graph.increment['output_tokens']
                    arrays[f'{phase}_{background}_menu_tokens'][spec.episode, i] = total_tokens
                    if phase == 'C':
                        for slot, edge in enumerate((None,) + spec.editable):
                            graph = store.find(unit, background, edge)
                            cached = store.cache.get((unit['unit_id'], graph.request['name']))
                            require(cached is not None, 'IC export cannot acquire previously unscored feedback')
                            arrays[f'C_{background}_quality'][spec.episode, i, slot] = cached[0]
                if phase == 'C':
                    calls = {}
                    for graph in archive.graphs[unit['unit_id']].values():
                        for call_id, input_tokens, output_tokens in graph.node_call_tokens:
                            require(call_id not in calls or calls[call_id] == (input_tokens, output_tokens), 'Joint-call cost disagreement')
                            calls[call_id] = (input_tokens, output_tokens)
                    arrays['C_AB_joint_tokens'][spec.episode, i] = sum(x + y for x, y in calls.values())
                    arrays['C_AB_joint_calls'][spec.episode, i] = len(calls)
                    require(len(calls) == 37, 'Frozen C A/B fourteen-menu union is not 37 physical calls')
    source_pins = {rel: pin for rel, pin in archive.file_pins.items()
                   if any('/units/' + unit + '/' in rel for unit in ic_units)}
    target = private / 'R2_IC_INPUTS.npz'
    with target.open('xb') as f:
        np.savez_compressed(f, **arrays)
    immutable_json(private / 'R2_IC_RECEIPT.json', {'schema': 'mab2_r2_ic_export_v1',
        'status': 'FROZEN_IC_ONLY_BEFORE_S_READ', 'R1_protocol_sha256': policy.PROTOCOL_SHA256,
        'R1_collection_plan_sha256': archive.plan_sha, 'R1_actions_file_sha256': file_sha(action_path),
        'R1_action_payload_digest': actions.digest, 'npz_sha256': file_sha(target),
        'executor_protocol_hash': archive.plan['executor_protocol_hash'],
        'model_identity_sha256': digest(archive.plan['model']['identity']), 'quality_scorer_sha256': SCORER_SHA256,
        'model_identity_digest': digest(archive.plan['model']['identity']),
        'source_manifest_digest': digest(archive.plan['source_manifest']),
        'analysis_source_sha256': code_identity.get('analysis_source_sha256', {}),
        'analysis_source_version': code_identity.get('git_commit', 'SYNTHETIC_FIXTURE_ONLY'),
        'analysis_source_commit_bytes_verified': code_identity.get('tracked_source_bytes_equal_HEAD', False),
        'array_keys': sorted(arrays), 'I_C_source_graphs_sha256': digest(source_pins),
        'I_C_source_artifact_hashes': source_pins, 'S_arrays_exported': False,
        'new_labels_or_outputs_scored_for_this_export': False, 'S_labels_accessed_at_export': False,
        'data_role': 'R1_development_I_C_only_not_independent_confirmation'})


def analyze_validated_archive(archive, label_descriptor, scorer, output, code_identity, *, label_descriptor_sha256):
    """Run after audit_archive only; test fixtures call this with synthetic data."""
    started = time.process_time(); wall = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    private = output / 'private'; public = output / 'public'; private.mkdir(); public.mkdir()
    start_pin = immutable_json(private / 'ANALYSIS_STARTED.json', {
        'schema': 'mab2_r1_analysis_start_v1', 'collection_plan_sha256': archive.plan_sha,
        'protocol_sha256': policy.PROTOCOL_SHA256, 'scorer_sha256': SCORER_SHA256,
        'code_identity': code_identity, 'time_unix': time.time(), 'labels_opened_by_analysis': []})
    freeze_path = private / 'OUTPUTS_FROZEN.json'
    freeze_pin = immutable_json(freeze_path, {'schema': 'mab2_r1_all_outputs_frozen_v1',
        'status': 'COMPLETE_LABEL_FREE', 'collection_plan_sha256': archive.plan_sha,
        'scientific_protocol_sha256': policy.PROTOCOL_SHA256, 'units': 390,
        'formal_questions': 384, 'formal_menu_outputs': 4704,
        'all_artifact_hashes': archive.file_pins, 'physical_research_costs': archive.physical_costs,
        'analysis_started_sha256': start_pin, 'first_I_read': False, 'first_C_read': False, 'first_S_read': False})
    store = ScoringStore(archive, scorer)
    labels = load_phase_labels(archive, label_descriptor, label_descriptor_sha256, 'I', private, {'path': freeze_path, 'sha256': freeze_pin})
    store.open_phase('I', labels)
    I_delta = np.zeros((8, 12, 6))
    for spec in archive.specs:
        columns = {}
        for edge in spec.editable:
            columns[edge] = [store.score(u, 'A', edge) - store.score(u, 'A', None)
                             for u in archive.unit_order[(spec.episode, 'I')]]
        I_delta[spec.episode] = policy.align_initial_columns(columns, spec)
    purchases = policy.freeze_purchases({e: I_delta[e] for e in range(8)}, archive.specs)
    purchase_path = private / 'PURCHASES_FROZEN.json'
    purchase_pin = immutable_json(purchase_path, policy.jsonable_freeze(purchases))
    immutable_json(private / 'I_SCORING_RECEIPT.json', {'purchases_file_sha256': purchase_pin,
        'purchase_payload_digest': purchases.digest, 'I_delta_sha256': policy._digest(I_delta.tolist()),
        'unique_scored_outputs': len(store.cache), 'scorer_CPU_seconds': sum(x['scoring_cpu_seconds'] for x in store.score_log)})
    labels = load_phase_labels(archive, label_descriptor, label_descriptor_sha256, 'C', private, {'path': purchase_path, 'sha256': purchase_pin})
    store.open_phase('C', labels); C_delta = np.full((8, 2, 12, 6), np.nan)
    accesses = []
    def read_C(plan, edge):
        require(edge in plan.purchased, 'Adapter refuses unpurchased feedback')
        spec = archive.specs[plan.episode]; ci = policy.CONDITIONS.index(plan.condition)
        background = 'B' if plan.condition == 'change' else 'A'
        result = np.array([store.score(u, background, edge) - store.score(u, background, None)
                           for u in archive.unit_order[(plan.episode, 'C')]])
        j = spec.editable.index(edge)
        prior = C_delta[plan.episode, ci, :, j]
        require(not np.isfinite(prior).any() or np.array_equal(prior, result), 'Common purchased C values differ')
        C_delta[plan.episode, ci, :, j] = result
        accesses.append({'replica_key': list(plan.key), 'physical_edge': edge,
                         'logical_paired_questions': 12, 'source': 'only_this_replica_purchased_C'})
        return result
    actions = policy.update_and_freeze_actions(purchases, read_C, persisted_purchase_digest=purchases.digest)
    action_path = private / 'ACTIONS_FROZEN.json'
    action_pin = immutable_json(action_path, policy.jsonable_freeze(actions))
    immutable_json(private / 'C_ACCESS_RECEIPT.json', {'actions_file_sha256': action_pin,
        'action_payload_digest': actions.digest, 'per_replica_accesses': accesses,
        'sharing_physical_scoring_cache_grants_extra_policy_information': False})
    require(np.isfinite(C_delta).all(), 'Full uniform enumeration did not cover C bank')
    export_IC_only(archive, store, I_delta, actions, action_path, private, code_identity)
    labels = load_phase_labels(archive, label_descriptor, label_descriptor_sha256, 'S', private, {'path': action_path, 'sha256': action_pin})
    store.open_phase('S', labels); S_quality = np.full((8, 2, 24, 7), np.nan)
    def read_S(episode, condition, action):
        ci = policy.CONDITIONS.index(condition); spec = archive.specs[episode]
        slot = 0 if action is None else 1 + spec.editable.index(action)
        background = 'B' if condition == 'change' else 'A'
        result = np.array([store.score(u, background, action) for u in archive.unit_order[(episode, 'S')]])
        S_quality[episode, ci, :, slot] = result
        return result
    evaluation = policy.grade_service(actions, read_S, persisted_action_digest=actions.digest)
    immutable_json(private / 'SERVICE_EVALUATION.json', policy.jsonable_freeze(evaluation))
    summary = policy.summarize(evaluation)
    costs = replay_cost_arrays(archive, store)
    logical_rows, replica_costs = logical_policy_costs(actions, costs)
    replay_path = private / 'REPLAY_INPUTS.npz'
    with replay_path.open('xb') as f:
        np.savez_compressed(f, I_delta=I_delta, C_delta=C_delta, S_quality=S_quality,
            editable_edges=np.array([s.editable for s in archive.specs], dtype=np.int64),
            dirty=np.array([s.dirty for s in archive.specs], dtype=np.int64),
            A_masks=np.array([s.A for s in archive.specs], dtype=np.int64),
            B_masks=np.array([s.B for s in archive.specs], dtype=np.int64), **costs)
    immutable_json(private / 'SCORING_LOG.json', {'observations': store.score_log,
        'total_measured_scoring_CPU_seconds': sum(x['scoring_cpu_seconds'] for x in store.score_log),
        'clock_resolution_seconds': time.get_clock_info('process_time').resolution,
        'logical_score_CPU_uses_per_observation_measured_reference_cost_not_new_physical_calls': True})
    immutable_json(private / 'REPLICA_LOGICAL_COSTS.json', {'cost_fields': ['input_tokens', 'output_tokens', 'calls', 'score_count', 'score_cpu_seconds'],
                                                         'replicas': replica_costs})
    # Original loaded bytes are the evidence. Recheck all source files before release.
    for rel, pin in archive.file_pins.items():
        require(file_sha(safe(archive.root, rel)) == pin, 'Raw archive changed during analysis')
    worker_costs = archive.physical_costs['current_collection_workers']
    summary['logical_policy_costs'] = logical_rows
    summary['cost_vector_fields'] = ['input_tokens', 'output_tokens', 'calls', 'score_count', 'score_cpu_seconds']
    summary['physical_research_costs'] = {'current_run_occupancy_GPUh': archive.physical_costs['current_collection_occupancy_GPUh'],
        'attempted_calls': sum(w['attempted_calls'] for w in worker_costs),
        'known_input_tokens': sum(w['input_tokens'] for w in worker_costs),
        'known_output_tokens': sum(w['output_tokens'] for w in worker_costs),
        'known_CUDA_event_seconds': sum(w['known_cuda_event_seconds'] for w in worker_costs),
        'failed_calls': sum(w['failed_calls'] for w in worker_costs),
        'pending_calls': sum(w['pending_calls'] for w in worker_costs),
        'prior_runs_GPUh': None, 'entire_project_cost_complete': False,
        'prior_costs_require_root_global_budget_ledger_merge': True}
    summary['analysis_costs'] = {'CPU_seconds_from_pipeline_entry': time.process_time() - started,
        'main_process_CPU_seconds_through_summary': time.process_time(),
        'wall_seconds_from_pipeline_entry': time.perf_counter() - wall,
        'archive_audit_and_import_CPU_seconds': None,
        'physical_unique_score_calls': len(store.cache),
        'measured_unique_scoring_CPU_seconds': sum(x['scoring_cpu_seconds'] for x in store.score_log)}
    summary['provenance'] = {'collection_plan_sha256': archive.plan_sha, 'scorer_sha256': SCORER_SHA256,
        'replay_inputs_sha256': file_sha(replay_path), 'code_identity': code_identity,
        'output_freeze_sha256': freeze_pin, 'purchases_file_sha256': purchase_pin, 'actions_file_sha256': action_pin}
    summary['provenance']['cost_arithmetic'] = 'producer_Python3.11_explicit_sequential_addition_native_r1_r2_aggregate_order_exact'
    for name, pin in code_identity.get('analysis_source_sha256', {}).items():
        require(file_sha(Path(__file__).resolve().parent / name) == pin, 'Analysis source changed while labels were open')
    result_pin = immutable_json(public / 'R1_RESULTS.json', summary)
    immutable_json(private / 'COMPLETE.json', {'status': 'COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION',
        'public_result_sha256': result_pin, 'replay_inputs_sha256': file_sha(replay_path),
        'source_archive_unchanged': True, 'S_released_after_all_actions_frozen': True,
        'new_model_calls': 0, 'global_prior_physical_cost_merge_pending': True})
    return summary


def code_identity(repo):
    repo = Path(repo).resolve()
    git_prefix = ['git', '-c', 'safe.directory=' + repo.as_posix(), '-C', str(repo)]
    def git(*args):
        result = subprocess.run(git_prefix + list(args), capture_output=True, text=True, check=True)
        return result.stdout.strip()
    head = git('rev-parse', 'HEAD'); dirty = git('status', '--porcelain')
    require(not dirty, 'Commit/review analysis source before opening labels; repository is dirty')
    here = Path(__file__).resolve().parent
    require(here.is_relative_to(repo), 'Run committed analysis files inside the supplied repository')
    require(Path(policy.__file__).resolve() == here / 'r1_policy.py', 'Imported policy is not the pinned adjacent source')
    for name in ('r1_adapter.py', 'r1_policy.py'):
        relative = (here / name).relative_to(repo).as_posix()
        committed = subprocess.run(git_prefix + ['show', 'HEAD:' + relative], capture_output=True, check=True).stdout
        require(sha(committed) == file_sha(here / name), 'Analysis source raw bytes differ from recorded HEAD')
    return {'git_commit': head, 'git_dirty': False,
            'tracked_source_bytes_equal_HEAD': True,
            'analysis_source_sha256': {n: file_sha(here / n) for n in ('r1_adapter.py', 'r1_policy.py')},
            'Python': os.sys.version.split()[0], 'numpy': np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    owner = sub.add_parser('split-owner')
    owner.add_argument('--source-labels', type=Path, required=True)
    owner.add_argument('--data-receipt', type=Path, required=True)
    owner.add_argument('--selection', type=Path, required=True)
    for cmd in (owner,):
        cmd.add_argument('--plan', type=Path, required=True); cmd.add_argument('--execution-pins', type=Path, required=True)
        cmd.add_argument('--protocol', type=Path, required=True); cmd.add_argument('--output', type=Path, required=True)
    analyze = sub.add_parser('analyze')
    for name in ('plan', 'execution-pins', 'protocol', 'run-root', 'labels-descriptor', 'scorer', 'repo-root', 'output'):
        analyze.add_argument('--' + name, type=Path, required=True)
    analyze.add_argument('--resource-amendment', type=Path)
    analyze.add_argument('--data-root', type=Path)
    analyze.add_argument('--labels-descriptor-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'split-owner':
        result = owner_split_labels(args.source_labels, args.plan, args.execution_pins, args.protocol, args.data_receipt, args.selection, args.output)
    else:
        identity = code_identity(args.repo_root)
        archive = audit_archive(args.plan, args.execution_pins, args.protocol, args.run_root,
                                resource_amendment=args.resource_amendment, data_root=args.data_root)
        scorer = load_scorer(args.scorer)
        result = analyze_validated_archive(archive, args.labels_descriptor, scorer, args.output, identity,
                                           label_descriptor_sha256=args.labels_descriptor_sha256)
        result = {'status': 'COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION',
                  'primary': result['primary'], 'point_rule_satisfied': result['point_rule_satisfied'],
                  'prior_global_cost_merge_pending': True}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
