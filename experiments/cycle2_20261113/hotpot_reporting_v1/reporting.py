"""Aggregate-only descriptive reporting after a pinned frozen score release.

No archive/question/label reader, scorer, bootstrap, candidate choice or GPU code.
The sole evaluation reader is called after two externally trusted metadata pins.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.dont_write_bytecode = True
N = 384
GRAPH_NAMES = ('A', 'B', 'A_MINUS_E', 'B_MINUS_E')
CONDITIONS = tuple('S1/' + name for name in ('A', 'B', 'NO_EDGE', 'A_MINUS_E', 'B_MINUS_E', 'INDEPENDENT12', 'SINGLE_M')) + tuple('S2/' + name for name in GRAPH_NAMES)
PRIMARY = (
    ('D1', 'B minus no-edge', 'B_MINUS_NO_EDGE_EM'),
    ('D2', 'B minus independent12', 'B_MINUS_INDEPENDENT12_EM'),
    ('D3', 'Mean signed delta', 'MEAN_SIGNED_DELTA_EM'),
    ('D4', 'Mean cross-seed delta product', 'MEAN_CROSSSEED_DELTA_PRODUCT_EM'))
MODELS = ('qwen', 'mistral')
COST_FIELDS = ('calls', 'input_tokens', 'output_tokens', 'CUDA_event_seconds')
PLAN_PIN = 'fc1ec977b51a08ba19b39236409ed1c3361f3f58a348cc6e33df4e050f6a4d31'
POSTCLOSURE_PIN = '8f701f52ac7fa15ea39f55f2beb877ce400774cc88484e7280c45da322ccb544'
FROZEN_PINS = {'evaluation_adapter_sha256': 'f8ba38be9ce1ef6bbfee66c17606685b117d25c52b947e66937ea5574a6105b8',
    'statistics_sha256': '3417dc11bc583d72a2ec8c8cc139c6ee490aa039d8eb386c97f7e2aa52060bfe',
    'scorer_sha256': '60d056c6c74080ff1186762074b6fb85b9ea7e229143bedd74299d87a1864c0e',
    'frontier_sha256': '6f641a3ff9c00afc2b045bc08435e42c602df5756747c58a7852f017da7bf7f9',
    'selector_sha256': '5edc8dc62b349be49c2c939c9c3f93ba1a8795b3c6fc7c951aec3ab9cd3d8bfd'}
DECISION_ENUMS = {
    'L1': {'SUPPORTED_BOTH', 'B_DISFAVORED', 'INCONCLUSIVE'},
    'L2': {'GRAPH_CONDITIONAL_SIGNAL_SUPPORTED', 'INCONCLUSIVE'},
    'signed_direction': {'INCONCLUSIVE', 'REDUNDANCY_DIRECTION_SUPPORTED', 'REDUNDANCY_DIRECTION_REVERSED'},
    'action': {'NO_AUTOMATIC_UPGRADE_CAPACITY_FAILURES', 'STOP_CURRENT_B_DIRECTION',
        'CONSIDER_INDEPENDENT_MECHANISM_PROTOCOL', 'PENDING_COST_FRONTIER_REVIEW',
        'STATIC_COMMUNICATION_SUPPORTED_REVALIDATION_UNRESOLVED', 'CONDITIONAL_RESPONSE_ONLY_NO_SYSTEM_SUCCESS',
        'EVIDENCE_INSUFFICIENT_STOP_AUTOMATIC_UPGRADE'},
    'theta_small_effect_exclusion': {'NOT_PERMITTED_BY_THIS_BOOTSTRAP_CONTRACT'}}


class ReportError(ValueError):
    pass


def require(value, message):
    if not value: raise ReportError(message)


def sha(raw):
    require(type(raw) is bytes, 'Exact bytes required')
    return hashlib.sha256(raw).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def decode(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, 'Duplicate JSON key'); obj[key] = value
        return obj
    require(type(raw) is bytes, 'Exact JSON bytes required')
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ReportError('Nonfinite JSON')))


def _optional_number(value):
    if value is None: return None
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric summary')
    return value


def _optional_bool(value):
    require(value is None or type(value) is bool, 'Invalid Boolean summary')
    return value


def _optional_enum(value, allowed):
    require(value is None or value in allowed, 'Unexpected frozen decision value')
    return value


def load_released_evaluation(*, receipt_bytes, expected_receipt_sha256, score_review_bytes,
                             expected_score_review_sha256, evaluation_reader):
    """Gate BEFORE reader. Expected pins must come from root, not self-approval."""
    require(is_hash(expected_receipt_sha256) and sha(receipt_bytes) == expected_receipt_sha256,
            'Released evaluation receipt SHA absent/different')
    receipt = decode(receipt_bytes)
    require(receipt['status'] == 'COMPLETE_PRIVATE_FROZEN_EVALUATION'
            and receipt['evaluation_adapter_sha256'] == FROZEN_PINS['evaluation_adapter_sha256']
            and receipt['postclosure_source_sha256'] == POSTCLOSURE_PIN, 'Not the frozen completed evaluation')
    require(type(receipt['new_model_calls']) is int and receipt['new_model_calls'] == 0
            and receipt['scientific_or_GPU_admission_granted'] is False
            and receipt['publication_authorized'] is False and is_hash(receipt['evaluation_sha256']), 'Evaluation scope differs')
    require(is_hash(expected_score_review_sha256) and sha(score_review_bytes) == expected_score_review_sha256
            and receipt['score_review_sha256'] == expected_score_review_sha256, 'Score-review SHA absent/different')
    review = decode(score_review_bytes)
    require(review['schema'] == 'hotpot_closed_archive_score_release_v1'
            and review['decision'] == 'APPROVED_FOR_SCORING' and review['scientific_protocol_status'] == 'FROZEN',
            'Root scoring release required')
    require(type(review['unique_questions']) is int and review['unique_questions'] == N
            and type(review['terminal_conditions']) is int and review['terminal_conditions'] == 4224
            and type(review['runtime_failure_count']) is int and review['runtime_failure_count'] == 0
            and review['all_attempts_closed'] is True and review['physical_ledger_closed'] is True,
            'Full closure required before report read')
    require(all(review['bindings'].get(key) == value for key, value in FROZEN_PINS.items()), 'Frozen source bindings differ')
    require(sha(Path(__file__).with_name('FROZEN_REPORTING_PLAN.md').read_bytes()) == PLAN_PIN, 'Frozen reporting plan differs')
    raw = evaluation_reader()
    require(sha(raw) == receipt['evaluation_sha256'], 'Released evaluation bytes differ')
    result = decode(raw)
    require(result.get('synthetic_fixture') is not True, 'Synthetic results cannot use the formal mode')
    require(result['archive_review_sha256'] == expected_score_review_sha256
            and result['bindings'] == review['bindings'], 'Evaluation/release ancestry differs')
    require(result['scientific_or_GPU_admission_granted'] is False, 'Unexpected admission in evaluation')
    physical = result['physical_research_ledger']
    require(physical['recomputed_as_logical_cost'] is False
            and physical['archive_metadata_sha256'] == review['archive_metadata_sha256']
            and all(is_hash(x) for x in physical['archive_metadata_sha256'].values()), 'Physical ledger pin mismatch')
    private = {'evaluation_receipt_sha256': expected_receipt_sha256, 'evaluation_sha256': receipt['evaluation_sha256'],
        'score_review_sha256': expected_score_review_sha256, 'frozen_source_pins': dict(FROZEN_PINS),
        'physical_archive_metadata_sha256': physical['archive_metadata_sha256'],
        'physical_values_not_available_in_evaluation': True, 'reporting_plan_sha256': PLAN_PIN}
    return result, private


def _metric_arrays(metric, binary):
    require(set(metric) == {'graphs', 'no_edge', 'independent12', 'single'}, 'Metric schema differs')
    result = {}
    for key, shape in (('graphs', (N, 2, 4)), ('no_edge', (N,)), ('independent12', (N,)), ('single', (N,))):
        array = np.asarray(metric[key], dtype=float)
        require(array.shape == shape and np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(),
                'Invalid score array shape/range')
        if binary: require(((array == 0) | (array == 1)).all(), 'EM must remain binary')
        result[key] = array
    return result


def _column(arrays, condition):
    prefix, name = condition.split('/')
    if name in GRAPH_NAMES: return arrays['graphs'][:, int(prefix[1]) - 1, GRAPH_NAMES.index(name)]
    return arrays[{'NO_EDGE': 'no_edge', 'INDEPENDENT12': 'independent12', 'SINGLE_M': 'single'}[name]]


def aggregate_evaluation(result, *, synthetic=False):
    """Public whitelist projection. No raw text, IDs, hashes or per-question rows."""
    require(result.get('synthetic_fixture') is True if synthetic else result.get('synthetic_fixture') is not True,
            'Synthetic/formal mode differs')
    require(result['status'] in ('COMPLETE_BOUND_SCORING', 'COMPLETE_WITH_CAPACITY_FAILURES'), 'Incomplete evaluation')
    audit = result['audit_inputs']
    require(audit['integrity_passed'] is True and audit['unique_questions'] == N
            and audit['complete_conditions'] == 4224 and audit['runtime_failure_count'] == 0,
            'Incomplete evaluation coverage')
    axes = result['array_axes']
    require(axes['graphs'] == list(GRAPH_NAMES) and axes['seeds'] == [26111301, 26111302], 'Frozen axes differ')
    em = _metric_arrays(result['em'], True); f1 = _metric_arrays(result['f1'], False)
    edge = result['edge_indices']
    require(len(edge) == N and all(type(x) is int for x in edge)
            and Counter(edge) == Counter({x: 96 for x in (1, 5, 7, 9)}), 'Fixed equal edge allocation differs')
    statuses = result['condition_statuses']; seen = set(); rejected = Counter()
    for row in statuses:
        index, condition = row['selection_index'], row['condition_key']
        require(type(index) is int and 0 <= index < N and condition in CONDITIONS
                and (index, condition) not in seen, 'Condition coverage/index differs')
        seen.add((index, condition))
        require(row['status'] in ('INPUT_REJECTED_ZERO', 'SCORED_STRICT_AGGREGATE'), 'Unexpected terminal status')
        if row['status'] == 'INPUT_REJECTED_ZERO':
            rejected[condition] += 1
            require(_column(em, condition)[index] == 0 and _column(f1, condition)[index] == 0,
                    'Frozen rejection zero differs')
    require(len(seen) == 4224 and sum(rejected.values()) == audit['input_overflow_conditions'], 'Terminal counts differ')
    require(result['status'] == ('COMPLETE_WITH_CAPACITY_FAILURES' if sum(rejected.values()) else 'COMPLETE_BOUND_SCORING'),
            'Evaluation summary status differs')
    global_format = result.get('format_failure_conditions')
    require(global_format is None or type(global_format) is int and 0 <= global_format <= 4224 - sum(rejected.values()),
            'Invalid global format count')
    conditions = [{'condition': c, 'n': N, 'em': float(_column(em, c).mean()), 'f1': float(_column(f1, c).mean()),
        'em_success_count': int(_column(em, c).sum()), 'input_rejections': rejected[c],
        'format_failures': None, 'finish_reasons': None,
        'diagnostic_availability': 'NOT_PROVIDED_BY_FROZEN_EVALUATION'} for c in CONDITIONS]
    stats = result.get('statistics') or {}
    require(isinstance(stats, dict), 'Statistics must be an object or absent')
    primary = []
    for contrast, label, key in PRIMARY:
        row = (stats.get('primary_EM') or {}).get(key) or {}
        interval = row.get('interval')
        require(interval is None or isinstance(interval, list) and len(interval) == 2, 'Invalid frozen interval')
        lower, upper = (None, None) if interval is None else tuple(_optional_number(x) for x in interval)
        require(lower is None or upper is None or lower <= upper, 'Reversed frozen interval')
        projected = {'contrast': contrast, 'label': label, 'estimate': _optional_number(row.get('estimate')),
            'lower': lower, 'upper': upper, 'support_guard_pass': _optional_bool(row.get('support_guard_pass')),
            'bootstrap_nondegenerate': _optional_bool(row.get('bootstrap_nondegenerate')),
            'availability': 'COPIED_FROM_FROZEN_EVALUATION' if row else 'NOT_PROVIDED_BY_FROZEN_EVALUATION'}
        for field in ('nonzero_questions', 'positive_questions', 'negative_questions', 'sample_variance',
                      'bootstrap_estimator_variance', 'interval_width'):
            projected[field] = _optional_number(row.get(field))
        primary.append(projected)
    decision = stats.get('decision') or {}
    gates = {key: _optional_enum(decision.get(key), values) for key, values in DECISION_ENUMS.items()}
    for key in ('theta_nonzero_supported', 'theta_negative_point_retained', 'GPU_admission_granted', 'method_success_claim'):
        gates[key] = _optional_bool(decision.get(key))
    gates['theta_width_divided_by_0_0025'] = _optional_number(decision.get('theta_width_divided_by_0_0025'))
    frontier_status = _optional_enum(result.get('frontier', {}).get('status'), {'NOT_ASSESSED', 'DOMINATED', 'NOT_DOMINATED'})
    # Only descriptive distributions follow the already frozen gamma/delta
    # formulas. Primary estimates, intervals and gates above are never recomputed.
    signed = []; hist = []
    for metric, arrays in (('EM', em), ('F1', f1)):
        graphs = arrays['graphs']; ga = graphs[:, :, 0] - graphs[:, :, 2]
        gb = graphs[:, :, 1] - graphs[:, :, 3]; delta = gb - ga
        entries = [('gamma_A', 'S1', ga[:, 0]), ('gamma_A', 'S2', ga[:, 1]),
            ('gamma_B', 'S1', gb[:, 0]), ('gamma_B', 'S2', gb[:, 1]),
            ('delta', 'S1', delta[:, 0]), ('delta', 'S2', delta[:, 1]),
            ('delta_product', 'S1xS2', delta[:, 0] * delta[:, 1])]
        for quantity, seed, values in entries:
            signed.append({'metric': metric, 'quantity': quantity, 'seed': seed, 'n': N,
                'negative': int((values < 0).sum()), 'zero': int((values == 0).sum()),
                'positive': int((values > 0).sum()), 'mean': float(values.mean()),
                'interpretation': 'DESCRIPTIVE_ONLY_NOT_A_NEW_GATE'})
            if metric == 'EM':
                limit = 1 if quantity.startswith('gamma') else 2 if quantity == 'delta' else 4
                for value in range(-limit, limit + 1):
                    hist.append({'quantity': quantity, 'seed': seed, 'value': value, 'count': int((values == value).sum())})
    logical = []
    raw_costs = result.get('full_logical_costs_by_model', {})
    for condition in CONDITIONS:
        name = condition[3:]
        available = condition.startswith('S1/') and name in ('B', 'NO_EDGE', 'INDEPENDENT12', 'SINGLE_M') and name in raw_costs
        for model in MODELS:
            row = {'condition': condition, 'model': model,
                'availability': 'FROZEN_FULL_LOGICAL_SERVICE' if available else 'NOT_PROVIDED_BY_FROZEN_EVALUATION'}
            for field in COST_FIELDS:
                value = _optional_number(raw_costs[name].get(model, {}).get(field)) if available else None
                require(value is None or value >= 0, 'Negative logical cost')
                row[field] = value
            logical.append(row)
    bootstrap = stats.get('bootstrap') or {}
    bootstrap_summary = {field: _optional_number(bootstrap.get(field)) for field in
        ('repetitions', 'family_size', 'family_alpha')}
    # The draw hash and seed are deliberately omitted from public export.
    return {'synthetic': synthetic,
        'scenario': 'SYNTHETIC_DEMONSTRATION' if synthetic else 'FROZEN_RELEASED_EVALUATION',
        'n_questions': N, 'terminal_conditions': 4224,
        'conditions': conditions, 'primary': primary, 'gates': gates, 'bootstrap': bootstrap_summary,
        'point_frontier_status': frontier_status, 'global_format_failure_conditions': global_format,
        'format_failure_denominator_completed_conditions': 4224 - sum(rejected.values()),
        'input_overflow_conditions': sum(rejected.values()), 'signed': signed, 'em_hist': hist,
        'logical_costs': logical, 'physical_costs': [{'quantity': key, 'value': None,
            'availability': 'NOT_PROVIDED_BY_FROZEN_EVALUATION'} for key in (
                'resident_seconds', 'GPU_hours', 'physical_requests', 'cache_calls_saved',
                'initialization_seconds', 'waiting_seconds', 'failure_cost')],
        'claims': {'descriptive_only': True, 'complete_cost_coverage': False,
            'new_statistical_tests': False, 'population_dominance_claim': False,
            'active_feedback_beats_random_claim': False, 'heterogeneous_causal_advantage_claim': False,
            'method_success_claim': False, 'project_stop_inferred': False}}


def _csv(path, rows):
    require(rows, 'Nonempty fixed table required')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: ('NA' if v is None else v) for k, v in row.items()} for row in rows)


def _display(value):
    if value is None: return 'NA'
    if type(value) is float: return f'{value:.6g}'
    return str(value)


def render_markdown(summary):
    tag = 'SYNTHETIC DEMONSTRATION — 非正式结果' if summary['synthetic'] else '已解封冻结评价的聚合报告'
    lines = ['# ' + tag, '', '本报告仅按预定范围展示固定结果；不新增检验、择优条件或准入。全部 384 题、11 条件保留。', '',
        '## 冻结主要结果', '',
        'D1–D4 的估计、区间、支持数及门均直接引用冻结评估输出，缺失则为 NA；没有重新抽样或重算判据。区间是原 Bonferroni 近似 bootstrap，不是有限样本精确保证。', '',
        '| 对比 | 估计 | 原区间 | 非零支持 | 支持门 |', '|---|---:|---|---:|---|']
    for row in summary['primary']:
        lines.append(f"| {row['contrast']}: {row['label']} | {_display(row['estimate'])} | [{_display(row['lower'])}, {_display(row['upper'])}] | {_display(row['nonzero_questions'])} | {_display(row['support_guard_pass'])} |")
    lines += ['', 'D1–D3 使用 EM 比例差；D4 使用 EM 比例的平方单位，保留负值，不开平方。', '',
        '| 冻结判据字段 | 原结果 |', '|---|---|']
    lines += [f'| {key} | {_display(value)} |' for key, value in summary['gates'].items()]
    lines += [f"| observed_point_frontier | {_display(summary['point_frontier_status'])} |", '',
        'STOP_CURRENT_B_DIRECTION 等字段只约束当前 B 路线。即使门通过，也只允许讨论独立新合同下的复检机制；不证明主动反馈胜过随机、不证明异构模型因果优势、不代表论文证据齐备。失败后继续既有外循环，项目停止仍服从总合同。', '',
        '## 全部条件的描述性结果', '',
        '| 条件 | n | EM | F1 | 输入拒绝置零 | aggregate格式失败 | 生成结束原因 |', '|---|---:|---:|---:|---:|---|---|']
    for row in summary['conditions']:
        lines.append(f"| {row['condition']} | {row['n']} | {_display(row['em'])} | {_display(row['f1'])} | {row['input_rejections']} | NA | NA |")
    lines += ['', f"全局 aggregate 格式失败计数：{_display(summary['global_format_failure_conditions'])}；已完成评分条件数：{summary['format_failure_denominator_completed_conditions']}；输入拒绝数：{summary['input_overflow_conditions']}。",
        '逐条件格式失败及生成结束原因未由冻结评估输出提供。NA 表示不可用，不能解释为零或未发生；报告器不从原文补抽答案。', '',
        '## 有符号信用与成本', '',
        'gamma_A=A−(A−e)，gamma_B=B−(B−e)，delta=gamma_B−gamma_A。两个 seed 分列，跨 seed 乘积保留负值；全部题、四边等权。符号计数和固定离散直方图只作描述，不替代 D3/D4 或原支持门。F1 的零按原浮点值精确比较，不另调容差。', '',
        '完整逻辑成本仅有 S1 的 B、no-edge、independent12、single-M 四方法，按模型分别列调用、输入/输出 token 和 CUDA 事件秒；包括复用节点的完整服务费用。其余七条件为 NA。独立12匹配调用及模型分配，不保证 token 或设备时间相等。', '',
        '物理研究调用、缓存节省、驻留、初始化、等待和失败费用的数值未由冻结评估输出提供，全部另列 NA。相关证据仅保留在私有回执，本公开报告没有覆盖全部成本；CUDA 事件秒不是物理驻留，也不按逻辑调用估算 GPUh。', '',
        '固定长度层未出现在该输出中，因此不生成长度分组。两个 seed 不代表全部采样变化，单任务不支持普遍系统结论。', '',
        '## 图表来源与范围', '',
        '图只读取本次已净化的聚合表。所有条件顺序固定，负值与缺失值保留；描述性 EM/F1 和成本没有另造误差条。主要区间图只使用上述原区间。PNG/PDF 及 CSV/JSON 均不含逐题行、标识、原题、答案或来源指纹。', '']
    return '\n'.join(lines)


def write_reports(summary, output, private_provenance):
    """Fresh outputs only; the public subtree is an explicit aggregate whitelist."""
    output = Path(output)
    require(not output.exists(), 'New output directory required; no overwrite')
    output.mkdir(parents=True, exist_ok=False)
    public = output / 'public'; private = output / 'private'
    public.mkdir(); private.mkdir()
    (public / 'REPORT.md').write_text(render_markdown(summary), encoding='utf-8')
    (public / 'SUMMARY.json').write_bytes(encode(summary))
    for filename, rows in (('conditions.csv', summary['conditions']), ('primary_contrasts.csv', summary['primary']),
        ('signed_support.csv', summary['signed']), ('em_credit_histograms.csv', summary['em_hist']),
        ('logical_service_costs.csv', summary['logical_costs']), ('physical_research_costs.csv', summary['physical_costs'])):
        tagged = [{'scenario': 'SYNTHETIC_DEMONSTRATION' if summary['synthetic'] else 'FROZEN_RELEASED_EVALUATION', **row} for row in rows]
        _csv(public / filename, tagged)
    from plotting import draw_figures
    figure_names = draw_figures(summary, public, private)
    public_files = sorted(p for p in public.iterdir() if p.is_file())
    # No input strings flow to public output; additionally reject identifiers or
    # 64-hex fingerprints in public text. Binary figures are generated from this
    # same small whitelist of numeric tables and fixed captions.
    for path in public_files:
        if path.suffix in ('.md', '.json', '.csv'):
            text = path.read_text(encoding='utf-8')
            require(re.search(r'\b[0-9a-fA-F]{64}\b', text) is None, 'Fingerprint in public text')
            require('hotpotqa_' not in text, 'Question identifier in public text')
    receipt = {'status': 'SYNTHETIC_REPORT_ONLY' if summary['synthetic'] else 'COMPLETE_DESCRIPTIVE_REPORT_REQUIRES_ROOT_REVIEW',
        'input_provenance': private_provenance,
        'report_source_sha256': {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
            ('reporting.py', 'report_cli.py', 'plotting.py')},
        'reporting_plan_sha256': PLAN_PIN, 'public_whitelist_sha256': {p.name: sha(p.read_bytes()) for p in public_files},
        'figures': figure_names, 'new_statistics_run': False, 'new_model_calls': 0,
        'full_physical_cost_coverage': False, 'publication_authorized_by_generator': False,
        'scientific_or_GPU_admission_granted': False, 'project_stop_inferred': False}
    (private / 'REPORT_RECEIPT.json').write_bytes(encode(receipt))
    return receipt
