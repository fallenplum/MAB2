"""Streaming, aggregate-only post-release diagnostics. No answer scoring."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
PINS = {
    'evaluation_adapter.py': 'f8ba38be9ce1ef6bbfee66c17606685b117d25c52b947e66937ea5574a6105b8',
    'hotpot_support.py': '60d056c6c74080ff1186762074b6fb85b9ea7e229143bedd74299d87a1864c0e'}
TRUSTED = {
    'evaluation': '0f63c65f6d60d8ca924ff1c9e814425b4e906a1658f1ce81abf24530b9d32270',
    'score_receipt': '653a7df8c6eb0df7d6c7c23c1253c53219437578b6e8ea55e435e9c8254ca506',
    'score_review': 'f3482981cb9bbd11c08a746ae35620e7ccb5868ea31cfed46900ea1332887e36',
    'audit': 'fff1d22724f5d0607cf39cd53087d8bf1c57a6bc80ab088b7dffaa6d7d65afd4',
    'root_cost': '3fbd550f532435cad824f181bfa35991a32a0d788cb7489a9b99c461059a87b9',
    'prior_registry': 'e669d778a15c0b5dc11eef4c18205772f85d27d8a1251538a22eefb3662d2ccf'}
RAW_MANIFEST_PIN = 'f3487bb5d5602acdf2ddf57dea4a092c98bf7d0bd4e04522220ecda29a02acb3'
MODELS = ('qwen', 'mistral')
REASONS = ('eos', 'length', 'unknown')
MAIN_FIELDS = ('calls', 'input_tokens', 'output_tokens', 'CUDA_event_seconds')
EXTRA_FIELDS = ('model_elapsed_seconds', 'received_message_tokens', 'self_state_tokens', 'peak_allocated_bytes')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def decode(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, 'Duplicate JSON key'); obj[k] = v
        return obj
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read_pinned(path, pin):
    raw = Path(path).read_bytes()
    require(sha(raw) == pin, 'Input byte pin mismatch')
    return decode(raw)


def frozen_adapter():
    for name, pin in PINS.items():
        require(sha((ROOT / 'frozen_dependencies' / name).read_bytes()) == pin, 'Frozen source differs')
    name = '_descriptive_original_adapter'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'frozen_dependencies/evaluation_adapter.py')
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def strict_format_flag(record):
    """Exact format-only guards/return field of the pinned original scorer.

    Do not call its answer_metrics branch, pass dummy gold, or reparse raw text.
    The unchanged original source is kept and hash-verified by frozen_adapter.
    """
    meta = record['message_metadata']; parsed = record['parsed']
    require(type(meta['format_ok']) is bool, 'Explicit original format flag required')
    require(set(parsed) == {'answer', 'evidence', 'uncertainty'} and
        all(isinstance(v, str) for v in parsed.values()), 'Normalized parsed structure differs')
    return meta['format_ok']


def number(value, integer=False):
    require((type(value) is int if integer else type(value) in (int, float)) and
        math.isfinite(value) and value >= 0, 'Unknown/invalid observed numeric cost')
    return value


def add_cost(target, source):
    for field, value in source.items():
        if target[field] is None or value is None: target[field] = None
        elif field == 'peak_allocated_bytes': target[field] = max(target[field], value)
        else: target[field] += value


def zero_cost(): return {k: 0 for k in (*MAIN_FIELDS, *EXTRA_FIELDS)}


def close(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)


class Aggregator:
    def __init__(self, adapter):
        self.adapter = adapter; self.n = 0
        self.formats = {c: Counter() for c in adapter.ORDER}
        self.logical = {(c, m): zero_cost() for c in adapter.ORDER for m in MODELS}
        self.logical_reused = Counter(); self.finish = Counter(); self.unknown_finish_values = Counter()
        self.physical = {m: zero_cost() for m in MODELS}
        self.physical_elapsed_by_worker = Counter(); self.physical_by_worker = Counter()
        self.returned_count = 0; self.unreturned_requests = 0; self.seen_call_ids = set()

    def count_finish(self, scope, condition, model, node, generation, reused=False):
        role = 'aggregate' if node == 'aggregate' else 'worker'
        raw = generation.get('finish_reason')
        reason = raw if raw in ('eos', 'length') else 'unknown'
        self.finish[scope, condition, model, role, reason] += 1
        if raw not in REASONS: self.unknown_finish_values[scope] += 1
        if scope == 'logical' and reused: self.logical_reused[condition, model] += 1

    def consume(self, matrix, worker):
        require(matrix['status'] in ('COMPLETE', 'COMPLETE_WITH_INPUT_REJECTIONS'), 'Incomplete matrix')
        require(tuple(c['key'] for c in matrix['conditions']) == self.adapter.ORDER, 'Condition order differs')
        for condition in matrix['conditions']:
            key = condition['key']; name = condition['name']
            require(key.split('/')[1] == name, 'Condition name differs')
            self.formats[key]['conditions'] += 1
            if condition['status'] == 'INPUT_REJECTED':
                self.formats[key]['input_rejected'] += 1
                for model in MODELS:
                    if name == 'SINGLE_M' and model == 'qwen': continue
                    add_cost(self.logical[key, model], {k: None for k in self.logical[key, model]})
            else:
                require(condition['status'] == 'COMPLETE', 'Unsupported terminal condition')
                records = self.adapter._records(condition['execution'], name)
                costs, _ = self.adapter._node_costs(records)
                ok = strict_format_flag(records['aggregate'])
                self.formats[key]['aggregate_format_ok' if ok else 'aggregate_format_failed'] += 1
                extras = {m: {k: 0 for k in EXTRA_FIELDS} for m in MODELS}
                for node, record in records.items():
                    model = self.adapter._model(node); generation = record['generation']
                    extras[model]['model_elapsed_seconds'] += generation['elapsed_seconds']
                    for field in ('received_message_tokens', 'self_state_tokens'): extras[model][field] += record[field]
                    extras[model]['peak_allocated_bytes'] = max(extras[model]['peak_allocated_bytes'], generation['gpu_peak_allocated_bytes'])
                    reused = record.get('reused', False)
                    require(type(reused) is bool, 'Invalid reuse flag')
                    self.count_finish('logical', key, model, node, generation, reused)
                for model in MODELS: add_cost(self.logical[key, model], {**costs[model], **extras[model]})
            for trace in condition['request_trace']:
                generation = trace.get('generation')
                if generation is None:
                    self.unreturned_requests += 1; continue
                node = trace['node']; model = trace['model']
                require(model in MODELS and self.adapter._model(node) == model, 'Physical route differs')
                call_id = generation.get('call_id')
                require(isinstance(call_id, str) and call_id and call_id not in self.seen_call_ids, 'Missing/duplicate physical returned identity')
                self.seen_call_ids.add(call_id)
                cost = {'calls': 1,
                    'input_tokens': number(generation['input_tokens'], True),
                    'output_tokens': number(generation['output_tokens'], True),
                    'CUDA_event_seconds': number(generation['gpu_seconds']),
                    'model_elapsed_seconds': number(generation['elapsed_seconds']),
                    'received_message_tokens': 0, 'self_state_tokens': 0,
                    'peak_allocated_bytes': number(generation['gpu_peak_allocated_bytes'], True)}
                # Receiver/self token attribution is supplied by the external
                # audit, not guessed from request traces lacking normalized records.
                add_cost(self.physical[model], cost)
                self.physical_elapsed_by_worker[worker] += cost['model_elapsed_seconds']
                self.physical_by_worker[worker] += 1
                self.returned_count += 1
                self.count_finish('physical_returned', key, model, node, generation)
        self.n += 1

    def output_rows(self):
        formats = []
        for key in self.adapter.ORDER:
            counts = self.formats[key]
            formats.append({'condition': key, 'conditions': counts['conditions'],
                'aggregate_format_ok': counts['aggregate_format_ok'],
                'aggregate_format_failed': counts['aggregate_format_failed'],
                'input_rejected_not_format_tested': counts['input_rejected'],
                'format_denominator': counts['aggregate_format_ok'] + counts['aggregate_format_failed']})
        logical = [{'condition': c, 'model': m, **self.logical[c, m],
            'reused_logical_calls_included': self.logical_reused[c, m]} for c in self.adapter.ORDER for m in MODELS]
        finish = []
        for scope in ('logical', 'physical_returned'):
            for condition in self.adapter.ORDER:
                for model, role in (('qwen', 'worker'), ('mistral', 'worker'), ('mistral', 'aggregate')):
                    counts = {reason: self.finish[scope, condition, model, role, reason] for reason in REASONS}
                    finish.append({'scope': scope, 'condition': condition, 'model': model, 'role': role,
                        'observed_records': sum(counts.values()), **counts})
        return formats, logical, finish


def _relative_file(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Relative archive path required')
    result = (root / name).resolve()
    require(result.is_relative_to(root.resolve()), 'Archive path escapes root')
    return result


def run_real(pathmap, output):
    """One explicitly authorized stream. Metadata-only pins before MATRIX IO."""
    started = time.monotonic(); adapter = frozen_adapter()
    require(not output.exists(), 'Fresh output directory required')
    # All paths and provenance are private; only hash-bound objects are read.
    score_receipt = read_pinned(pathmap['score_receipt'], TRUSTED['score_receipt'])
    review = read_pinned(pathmap['score_review'], TRUSTED['score_review'])
    require(score_receipt['status'] == 'COMPLETE_PRIVATE_FROZEN_EVALUATION' and
        score_receipt['evaluation_sha256'] == TRUSTED['evaluation'] and
        score_receipt['score_review_sha256'] == TRUSTED['score_review'], 'Score release ancestry differs')
    require(review['decision'] == 'APPROVED_FOR_SCORING' and review['all_attempts_closed'] is True and
        review['physical_ledger_closed'] is True and review['unique_questions'] == 384 and
        review['terminal_conditions'] == 4224 and review['runtime_failure_count'] == 0, 'Full release absent')
    require(review['bindings']['evaluation_adapter_sha256'] == PINS['evaluation_adapter.py'] and
        review['bindings']['scorer_sha256'] == PINS['hotpot_support.py'], 'Frozen operator ancestry differs')
    evaluation = read_pinned(pathmap['evaluation'], TRUSTED['evaluation'])
    require(evaluation['archive_review_sha256'] == TRUSTED['score_review'] and evaluation['bindings'] == review['bindings'], 'Evaluation binding differs')
    # Only existing global format count and four-method cost table are retained;
    # main statistics and per-question quality arrays are never used.
    comparison = {'format': evaluation['format_failure_conditions'],
        'cost': evaluation['full_logical_costs_by_model'], 'status': evaluation['status']}
    del evaluation
    audit = read_pinned(pathmap['audit'], TRUSTED['audit'])
    root_cost = read_pinned(pathmap['root_cost'], TRUSTED['root_cost'])
    prior = read_pinned(pathmap['prior_registry'], TRUSTED['prior_registry'])
    require(root_cost['status'] == 'ROOT_REVIEWED_FULL_ARCHIVE_AND_ACTUAL_COST_CLOSURE' and
        root_cost['audit_sha256'] == TRUSTED['audit'] and root_cost['cap_used_as_cost'] is False and
        root_cost['prior_qualification_recharged'] is False, 'Root cost closure differs')
    require(review['postclosure']['audit_sha256'] == TRUSTED['audit'] and audit['synthetic_tokenizer_witness'] is False,
        'Real audit witness binding differs')
    require(review['archive_metadata_sha256'] == root_cost['metadata_sha256'], 'Archive metadata ancestry differs')
    archive = Path(pathmap['archive_root'])
    metadata = {name: read_pinned(archive / name, pin) for name, pin in review['archive_metadata_sha256'].items()}
    require(root_cost['metadata_sha256']['RAW_OUTPUT_MANIFEST.json'] == RAW_MANIFEST_PIN, 'Original manifest pin differs')
    manifest = metadata['RAW_OUTPUT_MANIFEST.json']; residence = metadata['PARENT_RESIDENCE.json']
    require(manifest['status'] == 'CLOSED_SNAPSHOT' and residence['all_workers_closed'] is True and
        residence['cap_used_as_cost'] is False, 'Archive not closed')
    units = {unit['index']: unit for unit in review['units']}
    require(len(units) == 384 and set(units) == set(range(384)), 'Fixed unit coverage differs')
    output.mkdir(parents=True, exist_ok=False); (output / 'private').mkdir(); (output / 'public').mkdir()
    pins_before = {name: sha((ROOT / name).read_bytes()) for name in ('supplement.py', 'DESCRIPTIVE_SCOPE.md')}
    aggregate = Aggregator(adapter)
    try:
        for index in range(384):
            entry = units[index]; prefix = entry['directory']
            require(re.fullmatch(r'worker_[0-3]/unit_' + f'{index:03d}', prefix) is not None, 'Unit path differs')
            name = prefix + '/MATRIX.json'; pin = entry['sha256']['MATRIX.json']
            require(manifest['files'][name] == pin, 'Review/original manifest MATRIX pin differs')
            matrix = read_pinned(_relative_file(archive, name), pin)
            require(matrix['matrix_protocol_hash'] == review['bindings']['matrix_protocol_hash'], 'Matrix operator differs')
            aggregate.consume(matrix, int(prefix[7])); del matrix
        formats, logical, finish = aggregate.output_rows()
        require(aggregate.n == 384 and all(row['conditions'] == 384 for row in formats), 'Coverage differs')
        require(sum(row['aggregate_format_failed'] for row in formats) == comparison['format'], 'Frozen global format count differs')
        for name, models in comparison['cost'].items():
            for model, costs in models.items():
                for field, value in costs.items():
                    require(aggregate.logical['S1/' + name, model][field] == value, 'Frozen four-method cost not exactly equal')
        require(aggregate.returned_count == audit['returned_generations'] == audit['physical_request_count'], 'Audit request count differs')
        require(aggregate.unreturned_requests == audit['preflight_rejections_with_no_native_record'], 'Unreturned request count differs')
        for field, audit_field in (('calls', 'calls'), ('input_tokens', 'input_tokens'), ('output_tokens', 'output_tokens'),
                ('CUDA_event_seconds', 'gpu_seconds'), ('model_elapsed_seconds', 'model_elapsed_seconds')):
            require(close(sum(row[field] for row in aggregate.physical.values()), audit['physical_success_cost'][audit_field]), 'Physical audit aggregate differs')
            require(close(sum(row[field] for row in aggregate.logical.values()), audit['full_logical_cost'][audit_field]), 'Full logical audit aggregate differs')
        require(root_cost['physical_success_cost'] == audit['physical_success_cost'] and
            root_cost['full_logical_research_cost'] == audit['full_logical_cost'], 'Root and audit costs differ')
        loads = Counter()
        for worker in range(4):
            for model in MODELS:
                name = f'worker_{worker}/MODEL_LOAD_COMPLETE_{model}.json'
                record = read_pinned(_relative_file(archive, name), manifest['files'][name])
                require(record['worker'] == worker and record['model'] == model, 'Model load record differs')
                loads[worker] += number(record['wall_seconds'])
        time_rows = []
        for worker, seconds in enumerate(residence['observed_seconds_by_worker']):
            require(number(seconds) == root_cost['seconds_by_worker'][worker], 'Worker residence differs')
            elapsed = aggregate.physical_elapsed_by_worker[worker]
            time_rows.append({'worker_index': worker, 'physical_returned_requests': aggregate.physical_by_worker[worker],
                'parent_observed_residence_seconds': seconds, 'observed_model_load_section_wall_seconds': loads[worker],
                'sum_returned_backend_elapsed_seconds': elapsed,
                'unattributed_residence_balance_seconds': seconds - loads[worker] - elapsed,
                'exact_barrier_wait_seconds': None, 'exact_GPU_initialization_seconds': None})
        prior_ids = prior['candidate_prior_transaction_ids']
        require(len(prior_ids) == 1 and prior['all_known_attempts_enumerated'] is True, 'Prior candidate ledger differs')
        prior_seconds = sum(row['seconds'] for row in prior['transactions'] if row['transaction_id'] in prior_ids)
        require(close(prior_seconds / 3600, root_cost['prior_candidate_GPU_hours_already_charged']), 'Already charged qualification differs')
        require(close(residence['closed_sum_seconds'] / 3600, root_cost['formal_attempt_GPU_hours']), 'Residence conversion differs')
        physical_rows = []
        for model, row in aggregate.physical.items():
            physical_rows.append({'scope': 'physical_returned_generations', 'model': model,
                **{k: v for k, v in row.items() if k not in ('received_message_tokens', 'self_state_tokens')}})
        summary = {'status': 'COMPLETE_DESCRIPTIVE_SUPPLEMENT_REQUIRES_ROOT_REVIEW',
            'questions': aggregate.n, 'conditions': sum(row['conditions'] for row in formats),
            'format_rows': formats, 'logical_cost_rows': logical, 'finish_reason_rows': finish,
            'physical_by_model': physical_rows, 'physical_audit_total': audit['physical_success_cost'],
            'full_logical_audit_total': audit['full_logical_cost'], 'time_components_by_worker': time_rows,
            'total_observed_model_load_section_wall_seconds': sum(loads.values()),
            'total_returned_backend_elapsed_seconds': sum(aggregate.physical_elapsed_by_worker.values()),
            'total_unattributed_residence_balance_seconds': sum(row['unattributed_residence_balance_seconds'] for row in time_rows),
            'formal_attempt_GPU_hours': root_cost['formal_attempt_GPU_hours'],
            'qualification_GPU_hours_already_charged': root_cost['prior_candidate_GPU_hours_already_charged'],
            'candidate_total_GPU_hours': root_cost['candidate_total_GPU_hours'],
            'global_closed_GPU_hours_after': root_cost['global_closed_GPU_hours_after'],
            'cache_avoided_new_requests': audit['full_logical_cost']['calls'] - aggregate.returned_count,
            'observed_reused_logical_calls': sum(aggregate.logical_reused.values()),
            'unknown_finish_values_count': dict(aggregate.unknown_finish_values),
            'exact_barrier_wait_seconds': None, 'failure_cost_separately_identified': None,
            'new_scores_or_statistical_tests': False, 'project_stop_inferred': False,
            'qualification_recharged': False, 'scientific_or_GPU_admission_granted': False}
        require(summary['cache_avoided_new_requests'] == summary['observed_reused_logical_calls'], 'Reuse count reconciliation differs')
        require(all(sha((ROOT / name).read_bytes()) == pin for name, pin in pins_before.items()), 'Source changed during stream')
        write_outputs(summary, output)
        receipt = {'status': summary['status'], 'source_sha256': pins_before, 'frozen_sources': PINS,
            'input_sha256': TRUSTED, 'archive_metadata_sha256': review['archive_metadata_sha256'],
            'matrix_files_read_once': aggregate.n, 'unit_outputs_written': 0, 'gold_or_INPUT_read': False,
            'scorer_answer_metrics_called': False, 'main_statistics_called': False,
            'four_method_logical_cost_exact_match': True, 'global_format_count_exact_match': True,
            'physical_and_full_logical_audit_match': True,
            'public_files_sha256': {p.name: sha(p.read_bytes()) for p in sorted((output / 'public').iterdir())},
            'elapsed_seconds': time.monotonic() - started, 'new_model_calls': 0, 'GPU_hours_added': 0,
            'publication_authorized': False}
        (output / 'private/RECEIPT.json').write_bytes(canonical(receipt))
        return receipt
    except Exception as error:
        (output / 'private/FAILED.json').write_bytes(canonical({'status': 'FAILED_DESCRIPTIVE_SUPPLEMENT',
            'completed_unit_stream_count': aggregate.n, 'error_type': type(error).__name__,
            'error_code': str(error) if str(error) in ('Unsupported terminal condition',) else 'INSPECT_PRIVATE_SOURCE_GUARD',
            'new_model_calls': 0, 'no_scoring_or_statistics_change': True}))
        raise


def csv_write(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: 'unknown' if v is None else v for k, v in row.items()} for row in rows)


def write_outputs(summary, output):
    public = output / 'public'
    (public / 'SUMMARY.json').write_bytes(canonical(summary))
    for filename, key in (('aggregate_format.csv', 'format_rows'), ('logical_costs_all_conditions.csv', 'logical_cost_rows'),
        ('finish_reasons.csv', 'finish_reason_rows'), ('physical_returned_by_model.csv', 'physical_by_model'),
        ('residence_components.csv', 'time_components_by_worker')):
        csv_write(public / filename, summary[key])
    (public / 'REPORT.md').write_text(report(summary), encoding='utf-8')
    for path in public.iterdir():
        text = path.read_text(encoding='utf-8')
        require(re.search(r'\b[0-9a-fA-F]{64}\b', text) is None and 'hotpotqa_' not in text and
            'E:\\' not in text and 'C:\\Users\\' not in text, 'Private identity in public output')


def report(s):
    lines = ['# 已解封 Hotpot 描述性补充', '',
        '本补充覆盖原固定 384 题 × 11 条件；只统计已冻结格式状态、结束原因与费用。没有 gold 读取、原文重解析、EM/F1 重评分或主统计重算。B 的科学结论仍以原冻结评价为准；本补充不救援候选、不推断项目终止。', '',
        '| 条件 | 严格 aggregate 格式成功 | 格式失败 | 输入拒绝未检查格式 |', '|---|---:|---:|---:|']
    lines += [f"| {r['condition']} | {r['aggregate_format_ok']} | {r['aggregate_format_failed']} | {r['input_rejected_not_format_tested']} |" for r in s['format_rows']]
    lines += ['', '格式字段沿用冻结 scorer 的 bool 与已解析结构检查；结构合法不等于 format_ok。格式失败数与原评价全局计数精确一致。', '',
        '结束原因详见 finish_reasons.csv：全部条件按模型和 worker/aggregate 分开；完整逻辑计数包含 reused，物理返回计数每个实际返回请求一次。eos 与 length 是原 backend 的记录值，不据 token 长度补判。unknown 表示字段缺失/未知，不能解释为零或未发生。', '',
        f"完整逻辑调用 {s['full_logical_audit_total']['calls']:,}；物理返回调用 {s['physical_audit_total']['calls']:,}；复用避免的新请求 {s['cache_avoided_new_requests']:,}。这些是本次执行几何下的计数，不是测得的驻留节省。", '',
        '全 11 条件的完整服务逻辑费用见 logical_costs_all_conditions.csv，按模型列 calls、输入/输出 token、CUDA 秒、backend elapsed、接收/自状态 token、峰值分配与 reused 数。四个原 S1 比较方法的主要费用逐字段精确匹配冻结评价；其余七条件是同口径描述补全。峰值取 max，时间和 token 求和。', '',
        '| 物理费用/区段 | 数值 |', '|---|---:|',
        f"| 正式采集实际驻留 GPUh（四 worker 求和） | {s['formal_attempt_GPU_hours']:.9f} |",
        f"| qualification 已收费 GPUh，仅单列 | {s['qualification_GPU_hours_already_charged']:.9f} |",
        f"| 当前候选含 qualification 已闭账 GPUh | {s['candidate_total_GPU_hours']:.9f} |",
        f"| 全项目已闭账 GPUh，非本方法费用 | {s['global_closed_GPU_hours_after']:.9f} |",
        f"| 实际返回请求 CUDA 事件秒之和 | {s['physical_audit_total']['gpu_seconds']:.6f} |",
        f"| 返回请求 backend elapsed 秒之和 | {s['total_returned_backend_elapsed_seconds']:.6f} |",
        f"| 8 次模型装载区段 wall 秒之和 | {s['total_observed_model_load_section_wall_seconds']:.6f} |",
        f"| 驻留减装载与 backend elapsed 的未分解余额（秒） | {s['total_unattributed_residence_balance_seconds']:.6f} |",
        '| 精确 barrier 等待、GPU 初始化内核、失败分项费用 | unknown |', '',
        '实际驻留来自 parent 从启动前到观测退出的 monotonic 区段，并行 worker 相加；不是并行 wall，也没有把 cap 作为花费。qualification 已计过一次，本报告不再次收费。', '',
        '模型装载计时覆盖原 factory 记录范围，不能等同于纯 GPU 初始化。backend elapsed 是原模型生成和同步区段，未包含全部分词、解码、序列化、fsync、Python 编排等时间。余额可能包含这些工作、屏障等待、调度、启动/退出及观测误差；不能称精确等待或 CPU/GPU 并行开销。CUDA 事件秒也不等于驻留。', '',
        '优势是补齐原报告的缺项并让完整逻辑与物理费用可核查；限制是格式、结束原因和成本都是描述，不能定位质量变化的因果原因。没有新阈值、显著性检验、选优或自动准入。来源与校验指纹仅保存于私有回执，公开表不含逐题数据或原文。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-authorized-real', action='store_true')
    parser.add_argument('--private-pathmap', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.run_authorized_real:
        print(json.dumps({'status': 'PREFLIGHT_NO_DATA_IO'})); return
    require(args.private_pathmap is not None and args.output is not None, 'Explicit private paths and fresh output required')
    receipt = run_real(decode(args.private_pathmap.read_bytes()), args.output)
    print(json.dumps({'status': receipt['status'], 'units_streamed': receipt['matrix_files_read_once'],
        'receipt_sha256': sha(canonical(receipt)), 'new_model_calls': 0}))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps({'status': 'FAILED_DESCRIPTIVE_SUPPLEMENT', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
