"""Artificial collector-shaped archive; contains no research questions or labels."""
import copy
from pathlib import Path
import r1_adapter as a
import r1_policy as p


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (a.canonical(value) + '\n').encode('utf-8'); path.write_bytes(raw)
    return a.sha(raw)


def write_lines(path, values):
    raw = ('\n'.join(a.canonical(x) for x in values) + '\n').encode('utf-8')
    Path(path).write_bytes(raw)


def menu(spec):
    out = []
    for bg, mask in [('A', spec.A), ('B', spec.B)]:
        base_name = bg + '_stay'
        out.append({'name': base_name, 'mask': list(mask), 'base': None if bg == 'A' else 'A_stay',
                    'background': bg, 'action': 'stay'})
        # Intentionally reverse physical edge order to exercise metadata alignment.
        for edge in reversed(spec.editable):
            flipped = list(mask); flipped[edge] ^= 1
            out.append({'name': bg + '_e' + str(edge), 'mask': flipped, 'base': base_name,
                        'background': bg, 'action': 'toggle', 'edge': edge})
    return out


def make_fixture(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    specs = p.canonical_episode_specs(); units = []
    for j in range(6):
        units.append({'unit_id': f'qual_{j}', 'question_id': f'artificial_qual_{j}',
                      'phase': 'qualification', 'seed': 3799, 'worker_index': 0})
    for s in specs:
        for phase, n in [('I', 12), ('C', 12), ('S', 24)]:
            for j in range(n):
                units.append({'unit_id': f'ep{s.episode}_{phase}_{j:02}',
                              'question_id': f'artificial_ep{s.episode}_{phase}_{j:02}',
                              'phase': phase, 'seed': s.seed, 'episode': s.episode, 'worker_index': 0})
    qmenu = menu(specs[0]); replay = copy.deepcopy(qmenu[-1])
    replay.update(name='full_replay', base=None, replay_of=qmenu[-1]['name'], action='replay')
    qmenu.append(replay)
    plan = {'schema': 'mab2_cycle2_collection_plan_v1', 'status': 'FROZEN', 'labels_read': False,
            'scientific_protocol_sha256': p.PROTOCOL_SHA256, 'runner_sha256': 'a' * 64,
            'executor_protocol_hash': 'b' * 64, 'source_manifest': {'synthetic.py': 'c' * 64},
            'model': {'identity': {'fixture': 'artificial_no_model_calls'}},
            'graph': {'rounds': 2, 'message_tokens': 128, 'aggregate_tokens': 256, 'max_input_tokens': 8192},
            'episodes': [{'episode': s.episode, 'seed': s.seed, 'permutation': [(j + s.shift) % 4 for j in range(4)],
                          'complement': bool(s.complement), 'menu': menu(s)} for s in specs],
            'qualification_menu': qmenu, 'units': units,
            'questions': {'payload_sha256': {u['question_id']: a.digest({'id': u['question_id']}) for u in units}},
            'workers': [{'index': 0, 'resident_wall_cap_seconds': 86400, 'max_call_attempts': 12948}],
            'budget': {'resident_wall_gpu_seconds': 86400, 'max_call_attempts': 12948}}
    plan_path = root / 'plan.json'; plan_sha = write(plan_path, plan)
    pins = {k: plan[k] for k in ('scientific_protocol_sha256', 'runner_sha256', 'executor_protocol_hash', 'source_manifest')}
    pins['plan_sha256'] = plan_sha; write(root / 'execution_pins.json', pins)
    run = root / 'run'; worker = run / 'worker_0'; worker.mkdir(parents=True)
    journal = []; native = []; call_count = 0; phase_units = {'QUALIFICATION': [], 'DEVELOPMENT': []}
    for unit in units:
        requests = qmenu if unit['phase'] == 'qualification' else plan['episodes'][unit['episode']]['menu'][:7] if unit['phase'] == 'I' else plan['episodes'][unit['episode']]['menu']
        prior = {}; graph_pins = {}; full_logical_calls = 0; fresh_calls = 0
        for request in requests:
            base = prior.get(request['base']); execution_id = unit['unit_id'] + '_' + request['name']
            if base is None:
                closure = set(a.NODE_ORDER)
            else:
                closure = {f'r2:w{p.EDGES[e][1]}' for e in range(12) if base['mask'][e] != request['mask'][e]}
                if closure: closure.add('aggregate')
            nodes = {}
            for name in a.NODE_ORDER:
                if name not in closure:
                    node = copy.deepcopy(base['nodes'][name]); node['reused'] = True; node['source_execution_id'] = base['execution_id']
                    nodes[name] = node; continue
                call_count += 1; call_id = f'artificial_call_{call_count}'
                seed = int(a.digest([unit['question_id'], unit['seed'], name])[:16], 16) % (2**31 - 1)
                if name.startswith('r1:'):
                    content = unit['question_id'] + name
                elif name.startswith('r2:'):
                    destination = int(name[-1])
                    incoming = [request['mask'][e] for e in range(12) if p.EDGES[e][1] == destination]
                    content = unit['question_id'] + name + str(incoming)
                else:
                    content = unit['question_id'] + ''.join(nodes[f'r2:w{j}']['wire_message'] for j in range(4))
                messages = [{'role': 'user', 'content': content}]
                answer = str(int(a.digest([messages, seed])[-1], 16) % 2)
                parsed = {'answer': answer, 'evidence': 'artificial', 'uncertainty': ''}
                text = a.canonical(parsed)
                # Heterogeneous magnitudes reveal producer-order floating sums.
                elapsed = (1., 1e-16, .003, 1e-15)[call_count % 4]
                gpu = (0.1, 1e-16, .0003, 1e-15)[call_count % 4]
                generation = {'text': text, 'input_tokens': len(content), 'output_tokens': len(text),
                              'elapsed_seconds': elapsed, 'gpu_seconds': gpu,
                              'gpu_peak_allocated_bytes': 128, 'finish_reason': 'eos', 'call_id': call_id}
                cap = 256 if name == 'aggregate' else 128
                fp = a.digest({'question': plan['questions']['payload_sha256'][unit['question_id']], 'node': name,
                               'messages': messages, 'seed': seed, 'protocol': plan['executor_protocol_hash'], 'max_new_tokens': cap})
                integrity = {'node': name, 'input_fingerprint': fp, 'seed': seed, 'messages': messages,
                             'generation': generation, 'wire_message': text, 'parsed': parsed,
                             'message_metadata': {}, 'received_message_tokens': 0, 'self_state_tokens': 0}
                nodes[name] = {**integrity, 'output_digest': a.digest(integrity), 'reused': False, 'source_execution_id': None}
                journal.append({'event': 'CALL_START', 'reservation_id': call_id, 'seed': seed,
                                'prompt_sha256': a.digest(messages), 'plan_sha256': plan_sha,
                                'unit_id': unit['unit_id'], 'request_name': request['name']})
                costs = {k: generation[k] for k in ('input_tokens', 'output_tokens', 'gpu_seconds', 'gpu_peak_allocated_bytes')}
                journal.append({'event': 'CALL_SUCCESS', 'reservation_id': call_id, 'call_id': call_id,
                                'plan_sha256': plan_sha, 'unit_id': unit['unit_id'], 'request_name': request['name'], **costs})
                native.append({'status': 'SUCCESS', 'call_id': call_id, **costs})
            value = {'execution_id': execution_id, 'question_id': unit['question_id'],
                     'question_fingerprint': plan['questions']['payload_sha256'][unit['question_id']],
                     'seed': unit['seed'], 'mask': request['mask'], 'protocol_hash': plan['executor_protocol_hash'],
                     'nodes': nodes, 'closure': sorted(closure), 'answer': nodes['aggregate']['parsed']['answer'],
                     'cache_reuse_count': 9 - len(closure), 'elapsed_seconds': 0.01}
            value['logical_service_cost'] = a.sum_cost(nodes)
            value['physical_cost'] = a.sum_cost({n: nodes[n] for n in closure})
            prior[request['name']] = value
            checkpoint = {'schema': 'mab2_cycle2_graph_checkpoint_v1', 'unit_id': unit['unit_id'],
                          'plan_sha256': plan_sha, 'request': request, 'execution': value,
                          'execution_value_sha256': a.digest(value), 'labels_read': False, 'quality': None}
            graph_pins[request['name']] = write(worker / 'units' / unit['unit_id'] / (request['name'] + '.json'), checkpoint)
            full_logical_calls += 9; fresh_calls += len(closure)
        receipt = {'unit': unit, 'plan_sha256': plan_sha, 'graphs': graph_pins, 'labels_read': False, 'scored': False,
                   'logical_calls': full_logical_calls, 'physical_calls_referenced': fresh_calls}
        pin = write(worker / 'units' / unit['unit_id'] / 'UNIT_COMPLETE.json', receipt)
        phase_units['QUALIFICATION' if unit['phase'] == 'qualification' else 'DEVELOPMENT'].append({'unit_id': unit['unit_id'], 'sha256': pin})
    qpin = None
    for name, entries in phase_units.items():
        pin = write(worker / (name + '_COMPLETE.json'), {'status': 'COMPLETE_UNSCORED', 'labels_read': False,
                    'plan_sha256': plan_sha, 'executor_protocol_hash': plan['executor_protocol_hash'], 'completed_units': entries})
        if name == 'QUALIFICATION': qpin = pin
    write(run / 'QUALIFICATION_ACCEPTED.json', {'status': 'PASS', 'plan_sha256': plan_sha,
          'executor_protocol_hash': plan['executor_protocol_hash'], 'worker_reports': {'0': qpin}})
    write_lines(worker / 'call_journal.jsonl', journal); write_lines(worker / 'physical_calls.jsonl', native)
    write_lines(worker / 'resident_wall.jsonl', [
        {'event': 'START', 'attempt_id': 'q', 'phase': 'qualification', 'plan_sha256': plan_sha},
        {'event': 'END', 'attempt_id': 'q', 'resident_wall_seconds': 1.5, 'plan_sha256': plan_sha},
        {'event': 'START', 'attempt_id': 'd', 'phase': 'development', 'plan_sha256': plan_sha},
        {'event': 'END', 'attempt_id': 'd', 'resident_wall_seconds': 10.25, 'plan_sha256': plan_sha}])
    source_labels = root / 'labels.jsonl'
    write_lines(source_labels, [{'id': u['question_id'], 'answer': '0', 'task': 'gsm8k'} for u in units])
    data_pin = write(root / 'data_receipt.json', {'labels_sha256': a.file_sha(source_labels)})
    selection = {'units': [{'question_id': u['question_id'], 'phase': 'qualification' if u['phase'] == 'qualification' else 'development',
                           'block': 'Q' if u['phase'] == 'qualification' else u['phase'],
                           'episode': u.get('episode', -1), 'seed': u['seed']} for u in units]}
    selection_pin = write(root / 'selection.json', selection)
    write(root / 'protocol_fixture.json', {'data': {'data_receipt_sha256': data_pin, 'source_selection_sha256': selection_pin}})
    return {'root': root, 'run': run, 'plan_path': plan_path, 'pins_path': root / 'execution_pins.json',
            'protocol_path': root / 'protocol_fixture.json', 'data_receipt_path': root / 'data_receipt.json',
            'selection_path': root / 'selection.json',
            'labels': source_labels, 'specs': specs, 'plan': plan, 'plan_sha': plan_sha,
            'call_count': call_count, 'artificial_only': True}
