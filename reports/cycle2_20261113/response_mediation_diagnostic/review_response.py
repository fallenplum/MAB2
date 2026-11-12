"""Independent arithmetic audit of frozen, exposed historical derived arrays.

Does not import production statistics/scoring, open raw executions, questions or
labels, or access models. D2 contains 400 questions; only train/tune 280 enter
array indexing and numerical reconstruction. All outputs are aggregate-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

D2_PIN = '7639dad2581ea52536b7c274d33fd019ad726a9bce9e8e4ec6792c68b77f4963'
PROTOCOL_PIN = '0bb50c240bcdb69ff409dccca3adc384d6c4b83f741efb87304d1543ee1976fe'
PRODUCTION_SOURCE_PIN = '35725f660fdda9f4020b0ff9ebab35d6c71035c810448101b342da7802144acd'
CATEGORIES = ('same_receiver_wire', 'changed_wire_same_receiver_answer', 'changed_receiver_answer')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def linear_percentile(values, probability):
    """Order-statistic interpolation, independent of np.quantile implementation."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (len(ordered) - 1) * probability
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def inspect(protocol_dir, d2_path):
    started = time.monotonic()
    paths = {name: protocol_dir / name for name in (
        'RESULTS.json', 'PRIVATE_MEASUREMENT_ROWS.npy', 'HASHED_NODE_CACHE.npz',
        'PROTOCOL.json', 'FREEZE_COMMIT.json', 'SCAN_STARTED.json',
        'INTERPRETATION_ADDENDUM.json', 'INTERPRETATION_FREEZE_RECEIPT.json')}
    pins = {name: file_sha(path) for name, path in paths.items()}
    d2_pin = file_sha(d2_path)
    require(d2_pin == D2_PIN, 'D2 input pin differs from frozen historical input')
    require(pins['PROTOCOL.json'] == PROTOCOL_PIN, 'Protocol pin mismatch')
    report = read_json(paths['RESULTS.json'])
    protocol = read_json(paths['PROTOCOL.json'])
    frozen = read_json(paths['FREEZE_COMMIT.json'])
    scan = read_json(paths['SCAN_STARTED.json'])
    interpretation = read_json(paths['INTERPRETATION_ADDENDUM.json'])
    interpretation_freeze = read_json(paths['INTERPRETATION_FREEZE_RECEIPT.json'])
    require(report['cache_sha256'] == pins['HASHED_NODE_CACHE.npz'], 'Result/cache hash mismatch')
    require(frozen['protocol_sha256'] == scan['protocol_sha256'] == PROTOCOL_PIN, 'Protocol freeze mismatch')
    require(frozen['code_sha256'] == scan['code_sha256'] == PRODUCTION_SOURCE_PIN, 'Recorded scan code pin mismatch')
    require(interpretation_freeze['sha256'] == pins['INTERPRETATION_ADDENDUM.json'], 'Interpretation binding mismatch')
    require(interpretation_freeze['result_existed'] is False and
            interpretation_freeze['running_code_or_protocol_modified'] is False and
            interpretation['original_protocol_changed'] is False and
            interpretation['running_source_changed'] is False, 'Unexpected interpretation amendment')
    require(protocol['fixed_categories'] == list(CATEGORIES), 'Category definitions changed')
    require(protocol['uncertainty']['seed'] == 97173 and protocol['uncertainty']['resamples'] == 2000,
            'Frozen bootstrap changed')

    with np.load(d2_path, allow_pickle=False) as source:
        partitions = source['partition']  # partition metadata for exclusion only
        row_question = source['row_question']
        require(len(partitions) == 400, 'Unexpected source population')
        selected_q = np.flatnonzero(np.isin(partitions, ('train', 'tune')))
        require(len(selected_q) == 280 and np.sum(partitions[selected_q] == 'train') == 200
                and np.sum(partitions[selected_q] == 'tune') == 80, 'Wrong permitted question subset')
        selected_graph = np.flatnonzero(np.isin(row_question, selected_q))
        graph_lookup = np.full(len(row_question), -1, dtype=np.int64)
        graph_lookup[selected_graph] = np.arange(len(selected_graph))
        # Materialize only selected outcomes; unused question data never enters calculations.
        quality = np.asarray(source['quality_by_seed'][selected_graph], dtype=np.float64)
        masks = source['masks'][selected_graph]
        left_global = source['left'].reshape(400, 2, 12)[selected_q]
        right_global = source['right'].reshape(400, 2, 12)[selected_q]
    left, right = graph_lookup[left_global], graph_lookup[right_global]
    require(np.all(left >= 0) and np.all(right >= 0), 'Excluded question reached pair mapping')
    expected_q = np.broadcast_to(selected_q[:, None, None], left_global.shape)
    require(np.array_equal(row_question[left_global], expected_q) and
            np.array_equal(row_question[right_global], expected_q), 'Cross-question pairing')
    require(np.isfinite(quality).all() and np.isin(quality, (0., 1.)).all(), 'Unexpected historical quality scale')
    require(np.isin(masks, (0, 1)).all(), 'Nonbinary masks')
    edge_index = np.arange(12)[None, None, :]
    changed_bits = masks[left] != masks[right]
    require(np.all(changed_bits.sum(axis=-1) == 1), 'Pair changes multiple edges')
    require(np.all(np.take_along_axis(masks[left], edge_index[..., None], axis=-1) == 0), 'Left edge is not absent')
    require(np.all(np.take_along_axis(masks[right], edge_index[..., None], axis=-1) == 1), 'Right edge is not present')

    with np.load(paths['HASHED_NODE_CACHE.npz'], allow_pickle=False) as source:
        present = source['present']
        require(present.shape == (len(row_question), 5), 'Cache seed dimensions differ')
        require(present[selected_graph].all() and not present[graph_lookup < 0].any(),
                'Cache coverage is not exactly the permitted old train/tune graphs')
        require(int(present.sum()) == report['raw_files'] == 36370, 'Cache execution coverage mismatch')
        cache = {key: source[key][selected_graph] for key in
                 ('wire', 'generation', 'prompt', 'answer', 'numeric', 'tokens')}
    for key in ('wire', 'generation', 'prompt', 'answer'):
        require(cache[key].shape == (len(selected_graph), 5, 9), 'Wrong node dimensions: ' + key)
        require(np.all(np.char.str_len(cache[key]) == 64), 'Missing node digest: ' + key)
    require(cache['numeric'].shape == (len(selected_graph), 5) and
            np.all(np.char.str_len(cache['numeric']) == 64), 'Missing numeric digests')
    require(np.issubdtype(cache['tokens'].dtype, np.integer) and
            cache['tokens'].shape == (len(selected_graph), 5, 9) and
            np.all(cache['tokens'] > 0), 'Invalid token counts')

    # Reconstruct four-dimensional [question, anchor, physical edge, seed] arrays.
    seeds = np.arange(5)[None, None, None, :]
    receivers = np.array([destination + 4 for sender in range(4)
                          for destination in range(4) if sender != destination])
    receiver_axis = receivers[None, None, :, None]
    li, ri = left[..., None], right[..., None]
    receiver_wire_same = cache['wire'][li, seeds, receiver_axis] == cache['wire'][ri, seeds, receiver_axis]
    receiver_answer_same = cache['answer'][li, seeds, receiver_axis] == cache['answer'][ri, seeds, receiver_axis]
    aggregate_changed = cache['numeric'][li, seeds] != cache['numeric'][ri, seeds]
    credit = quality[right] - quality[left]
    require(credit.shape == (280, 2, 12, 5), 'Reconstructed credit shape mismatch')
    category = np.where(receiver_wire_same, 0, np.where(receiver_answer_same, 1, 2))
    tokens = cache['tokens'].astype(np.int64)
    aggregate_tokens = tokens[ri, seeds, 8]
    suffix_tokens = tokens[ri, seeds, receiver_axis] + aggregate_tokens
    skipped_tokens = np.where(receiver_wire_same, aggregate_tokens, 0)

    all_nonaggregate_same = cache['wire'][li, seeds, :8] == cache['wire'][ri, seeds, :8]
    excluded_receiver = np.arange(8)[None, None, None, None, :] == receivers[None, None, :, None, None]
    require(np.all(all_nonaggregate_same | excluded_receiver), 'An unaffected node wire changed')
    aggregate_prompt_same = cache['prompt'][li, seeds, 8] == cache['prompt'][ri, seeds, 8]
    aggregate_generation_same = cache['generation'][li, seeds, 8] == cache['generation'][ri, seeds, 8]
    require(np.all(aggregate_prompt_same[receiver_wire_same]), 'Wire equality did not preserve aggregator prompt')
    require(np.all(aggregate_generation_same[receiver_wire_same]), 'Wire equality did not preserve generated text')
    require(np.all(credit[receiver_wire_same] == 0), 'Wire equality has nonzero credit')
    require(np.all(credit[~aggregate_changed] == 0), 'Numeric equality has nonzero credit')
    # This reproduces the documented assertion-counter convention; it is not an n of independent observations.
    chain_counts = {
        'one_unaffected_wire_array_check_per_cell': int(credit.size),
        'equal_receiver_wire_cells': int(receiver_wire_same.sum()),
        'three_checks_per_equal_wire_cell': int(3 * receiver_wire_same.sum()),
        'one_zero_credit_check_per_equal_aggregate_numeric_cell': int((~aggregate_changed).sum()),
    }
    chain_total = chain_counts['one_unaffected_wire_array_check_per_cell'] + chain_counts['three_checks_per_equal_wire_cell'] + chain_counts['one_zero_credit_check_per_equal_aggregate_numeric_cell']
    require(chain_total == report['deterministic_chain_checks'] == 79842, 'Chain-check counter mismatch')
    chain_counts['total'] = chain_total
    chain_counts['not_independent_sample_size'] = True

    axes = np.indices(credit.shape)
    reconstructed_rows = np.column_stack((
        selected_q[axes[0].ravel()], axes[1].ravel(), axes[2].ravel(), axes[3].ravel(),
        category.ravel(), aggregate_changed.ravel(), credit.ravel(), suffix_tokens.ravel(), skipped_tokens.ravel(),
    )).astype(np.float64)
    supplied_rows = np.load(paths['PRIVATE_MEASUREMENT_ROWS.npy'], allow_pickle=False)
    require(supplied_rows.shape == reconstructed_rows.shape == (33600, 9), 'Measurement row dimensions differ')
    require(np.array_equal(reconstructed_rows, supplied_rows), 'Independent cache/D2 reconstruction differs from measurement rows')
    del reconstructed_rows, supplied_rows

    # Same prespecified stream: train first, then tune without reseeding.
    rng = np.random.Generator(np.random.PCG64(97173))
    independent = {}
    numeric_comparisons = []

    def compare(label, actual, expected):
        if actual is None or expected is None:
            require(actual is expected, 'Null mismatch: ' + label)
            numeric_comparisons.append({'field': label, 'absolute_difference': 0.})
        else:
            difference = abs(float(actual) - float(expected))
            require(difference <= 2e-15, 'Reported arithmetic mismatch: ' + label)
            numeric_comparisons.append({'field': label, 'absolute_difference': difference})

    for partition in ('train', 'tune'):
        choose = partitions[selected_q] == partition
        classes = category[choose].reshape(int(choose.sum()), -1)
        events = (credit[choose] != 0).reshape(classes.shape)
        numerical = aggregate_changed[choose].reshape(classes.shape)
        signed = credit[choose].reshape(classes.shape)
        summarized = []
        counts_by_q = []
        event_by_q = []
        for number, name in enumerate(CATEGORIES):
            members = classes == number
            counts = members.sum(axis=1)
            events_per_question = (members & events).sum(axis=1)
            counts_by_q.append(counts)
            event_by_q.append(events_per_question)
            support = int(counts.sum())
            total_events = int(events_per_question.sum())
            numeric_changes = int((members & numerical).sum())
            value = {'category': name, 'n': support, 'questions': int(np.count_nonzero(counts)),
                     'event_rate': total_events / support if support else None,
                     'aggregate_numeric_answer_change_rate': numeric_changes / support if support else None,
                     'signed_credit': float(np.sum(np.where(members, signed, 0))) / support if support else None,
                     'event_count': total_events, 'aggregate_numeric_answer_change_count': numeric_changes,
                     'signed_credit_sum': int(np.sum(np.where(members, signed, 0)))}
            published = report['groups'][partition]['cells'][number]
            require(published['category'] == name, 'Category order differs')
            for key in ('n', 'questions', 'event_rate', 'aggregate_numeric_answer_change_rate', 'signed_credit'):
                compare(partition + '/' + name + '/' + key, value[key], published[key])
            summarized.append(value)
        overlap = (counts_by_q[1] > 0) & (counts_by_q[2] > 0)
        contrast = (event_by_q[2][overlap] / counts_by_q[2][overlap]
                    - event_by_q[1][overlap] / counts_by_q[1][overlap])
        if len(contrast):
            bootstrap_indices = rng.integers(0, len(contrast), size=(2000, len(contrast)))
            sampled = np.mean(contrast[bootstrap_indices], axis=1)
            interval = [linear_percentile(sampled, .025), linear_percentile(sampled, .975)]
            estimate = float(np.mean(contrast))
        else:
            interval = estimate = None
        total_suffix = int(suffix_tokens[choose].sum())
        possible_skip = int(skipped_tokens[choose].sum())
        ratio = possible_skip / total_suffix
        published = report['groups'][partition]
        compare(partition + '/paired_questions', int(overlap.sum()), published['paired_questions'])
        compare(partition + '/paired_mean', estimate, published['paired_answer_change_vs_answer_preserving_event_delta'])
        if interval is None:
            compare(partition + '/paired_ci95', None, published['paired_95_percentile_ci'])
        else:
            for i in (0, 1):
                compare(partition + '/paired_ci95/' + str(i), interval[i], published['paired_95_percentile_ci'][i])
        compare(partition + '/suffix_token_ratio', ratio, published['generic_exact_coalescence_suffix_token_saving_fraction'])
        independent[partition] = {'cells': summarized, 'question_count': int(choose.sum()),
            'single_seed_cells': int(classes.size), 'paired_questions': int(overlap.sum()),
            'paired_question_equal_weight_mean': estimate, 'paired_ci95_linear_percentile': interval,
            'suffix_token_denominator': total_suffix, 'potential_skipped_aggregate_token_numerator': possible_skip,
            'suffix_ratio': ratio, 'public_interval_supported_by_at_least_two_clusters': int(overlap.sum()) >= 2}

    # Detect any input mutation while reading; no producer or original artifact is modified.
    require(pins == {name: file_sha(path) for name, path in paths.items()}, 'Audit input changed during reconstruction')
    require(file_sha(d2_path) == d2_pin, 'D2 input changed during reconstruction')
    return {
        'status': 'PASS_INDEPENDENT_DERIVED_ARRAY_ARITHMETIC_REPLAY',
        'audit_script_sha256': file_sha(__file__), 'input_sha256': pins,
        'd2_input_sha256': d2_pin, 'source_code_imported': False,
        'scope': {'selected_questions': 280, 'train_questions': 200, 'tune_questions': 80,
                  'excluded_question_outcomes_used': False, 'source_container_questions': 400,
                  'raw_executions_opened': False, 'question_text_or_gold_opened': False,
                  'new_R1_accessed': False, 'official_test_accessed': False,
                  'new_GPU_calls': 0, 'new_candidates_screened': 0,
                  'reconstructed_measurement_rows': 33600, 'reconstructed_columns': 9,
                  'exact_measurement_array_equality': True, 'present_cached_executions': 36370},
        'chain_check_decomposition': chain_counts,
        'bootstrap': {'rng': 'NumPy PCG64', 'seed': 97173, 'resamples': 2000,
                      'stream_order': ['train', 'tune'], 'reseed_between_partitions': False,
                      'unit': 'question', 'contrast': 'per-question conditional event-rate difference; equal weight on overlapping questions',
                      'quantile': 'independent sorted order-statistic linear interpolation'},
        'independently_recomputed': independent,
        'numeric_fields_checked': len(numeric_comparisons),
        'maximum_reported_numeric_absolute_difference': max(x['absolute_difference'] for x in numeric_comparisons),
        'numeric_comparisons': numeric_comparisons,
        'interpretation_binding_verified': True,
        'limitations': [
            'Hashes are trusted derived evidence: no original wire, prompt, generation, numeric parse or gold was re-opened.',
            'The 79,842 value is a dependent assertion counter, not an independent sample size.',
            'Source index paths/raw-byte total and actual raw scan uniqueness were not independently rescanned.',
            'Receiver answer equality is literal parsed-string equality, not semantic correctness.',
            'Credit events are single-seed paired differences, not five-seed aggregated G2 risk.',
            'Categories condition on post-intervention output and do not identify a causal mediation effect.',
            'Receiver state is observed only after generating it; this is not a free acquisition signal.',
            'Token ratio excludes baseline bank construction and other full-budget costs; available to all methods.',
            'No candidate or model improvement is admitted by this audit.'
        ],
        'numpy_version': np.__version__, 'python_version': sys.version,
        'wall_seconds': time.monotonic() - started,
    }


def narrative(receipt):
    train, tune = (receipt['independently_recomputed'][x] for x in ('train', 'tune'))
    changed, same = tune['cells'][2], tune['cells'][1]
    lines = [
        '# 历史响应链诊断：独立派生数组算术复核', '',
        '**结论：复核通过。** 未导入生产统计模块；从固定 hash 缓存和旧 D2 的 train/tune 配对索引、图 mask、历史质量数组重新构造全部 33,600 行、9 列，与已保存测量数组逐元素完全相同。', '',
        f"全部 {receipt['numeric_fields_checked']} 个报告数值字段通过；最大绝对差为 {receipt['maximum_reported_numeric_absolute_difference']:.3g}。未读取真实题目、gold、原始执行记录、新 R1 或官方测试，未运行 GPU、修改原报告或筛选候选。", '',
        '## 可复核的关键结果', '',
        '| 分组 | 题数 | 单 seed 配对单元 | 三类别支持数 | 重叠题数 | 题均条件事件率差 | 95% 探索性题级区间 |',
        '|---|---:|---:|---|---:|---:|---|',
    ]
    for name, values in (('train', train), ('tune', tune)):
        ci = values['paired_ci95_linear_percentile']
        lines.append(f"| {name} | {values['question_count']} | {values['single_seed_cells']} | "
                     + '/'.join(str(c['n']) for c in values['cells'])
                     + f" | {values['paired_questions']} | {values['paired_question_equal_weight_mean']:.12f} | [{ci[0]:.12f}, {ci[1]:.12f}] |")
    lines += ['',
        f"tune 的 receiver 答案字符串改变组为 {changed['n']} 个配对单元，其中 {changed['event_count']} 个信用差非零（{changed['event_rate'] * 100:.6f}%）；字符串不变但 wire 改变组为 {same['n']} 个，其中 {same['event_count']} 个信用差非零（{same['event_rate'] * 100:.6f}%）。这两个比例是单 seed 条件事件率，不是 G2 五 seed 聚合风险。", '',
        'paired 指标先在每道题内分别计算两类的事件率之差，再在同时拥有两类的题上等权平均。因此它不是两组总体事件率直接相减。bootstrap 固定 PCG64 seed 97173，先使用 train 的抽样流，继续用于 tune，不重新设 seed；2,000 次按题重抽。独立复核用排序后的线性插值计算百分位。', '',
        '## 链检查与局部费用', '',
        '79,842 的计数拆解为：33,600 次非受影响节点 wire 数组检查 + 5,961 个 equal-wire 单元各 3 次 aggregate prompt／generation／zero-credit 检查 + 28,359 次 aggregate 数值不变时的 zero-credit 检查。它们相互依赖，不能当作 79,842 个独立样本。', '',
    ]
    for name, values in (('train', train), ('tune', tune)):
        lines.append(f"- {name}：潜在可省 aggregate token {values['potential_skipped_aggregate_token_numerator']:,} / "
                     f"receiver+aggregate 后缀 token {values['suffix_token_denominator']:,} = {values['suffix_ratio'] * 100:.9f}%。")
    lines += ['',
        '这个 token 比例只适用于 off-world baseline 已缓存的特定后缀。它没有计入银行建设、生成 receiver 之前的费用、反馈打分、完整服务、失败或驻留成本；不能宣称完整系统节省。', '',
        '## 科学解释和审计覆盖边界', '',
        '优势：数据绑定、配对方向、原数组一致性、支持数、条件事件率、题级 bootstrap 和局部成本均有独立实现支持。', '',
        '风险：分组条件是干预后才观察到的 receiver 输出；其关联不能识别因果中介效应，也不能把需要先生成 receiver 才获得的状态当作免费采样信号。相同 parsed answer 字符串只表示字面相同，不能替代语义或正确性等价。', '',
        '400 题源 NPZ 中仅 train/tune 的 280 题参与数值重建。未访问 question_ids、题目或外部 gold；另外 120 题没有进入结果计算。固定协议、扫描起始元数据、解释附录及其结果前保存回执之间的哈希绑定通过；这验证记录一致性，不独立证明人的历史访问行为。', '',
        '本复核不从摘要 hash 还原原文，因此不能重新核验 raw→wire、raw→numeric parser、实际 token 计数或 gold→quality 转换是否正确；这些保留为生产日志与先前验证的依赖。36,370 条 present 标记覆盖得到复核，885,991,882 原始字节总量及原始路径索引没有重扫。', '',
        '置信度：派生数组算术一致性为高；不对因果效应、泛化或候选方法收益给出概率。结果仅维持该旧开发诊断的探索性证据级别。', '',
        '复现入口：`review_response.py --diagnostic-dir <旧诊断目录> --d2-inputs <固定旧inputs.npz> --output <不存在的新审查目录>`。输入和代码 SHA 位于 `REVIEW_RECEIPT.json`。', '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--d2-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new audit output directory; do not overwrite a receipt')
    result = inspect(args.diagnostic_dir.resolve(strict=True), args.d2_inputs.resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'REVIEW.md'
    report_path.write_text(narrative(result), encoding='utf-8')
    result['review_md_sha256'] = file_sha(report_path)
    receipt = args.output / 'REVIEW_RECEIPT.json'
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'numeric_fields_checked': result['numeric_fields_checked'],
                      'maximum_difference': result['maximum_reported_numeric_absolute_difference'],
                      'chain_checks': result['chain_check_decomposition'],
                      'independently_recomputed': result['independently_recomputed'],
                      'receipt_sha256': file_sha(receipt), 'review_md_sha256': result['review_md_sha256'],
                      'wall_seconds': result['wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
