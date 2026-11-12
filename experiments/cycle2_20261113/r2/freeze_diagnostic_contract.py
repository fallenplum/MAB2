"""One-time prospective freeze. Reads design documents only, no experiment data."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    draft = ROOT / 'R2_DRAFT_CONTRACT.json'
    original = json.loads(draft.read_text(encoding='utf-8-sig'))
    contract = {
        'schema': 'mab2_r2_frozen_ic_diagnostic_v1',
        'status': 'FROZEN_BEFORE_REAL_IC_ACCESS',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'Root explicitly accepted this low-cost investment screen and requested freeze; no new GPU authorization requested or implied.',
        'candidate_id': original['candidate_id'],
        'protocol_sha256': original['sources']['r1_protocol_sha256'],
        'source_versions': {
            'draft_json_sha256': sha(draft),
            'draft_md_sha256': sha(ROOT / 'R2_DRAFT_CONTRACT.md'),
            'literature_sources_sha256': sha(ROOT / 'SOURCES.json'),
            'pure_math_source_sha256': sha(ROOT / 'r2_math.py'),
            'freeze_script_sha256': sha(Path(__file__)),
            'r1_commit_parent_reported': original['sources']['r1_commit_reported_by_parent'],
            'r1_git_commit_independently_verified_by_this_subagent': False,
            'exact_source_revision_and_model_manifest': 'Must be included and hash-bound in IC_EXPORT_RECEIPT; no unverified hardcoded current revision.',
        },
        'information_permissions': {
            'allowed': ['independent I/C-only scored numeric arrays', 'I/C integer token/call counts',
                        'hashed question identifiers and paired execution seeds', 'frozen graph metadata'],
            'forbidden': ['S arrays or summary', 'S costs', 'raw question text', 'gold labels', 'raw model answers'],
            'release': 'R1 actions already immutable; export receipt binds protocol, action freeze, model/source manifests and NPZ hash.',
            'extra_npz_keys': 'FAIL_CLOSED_INCLUDING_ANY_S_KEY',
            'real_arrays_read_when_frozen': False,
            'final_or_sealed_test_used': False,
        },
        'dimensions': {'episodes': 8, 'I_questions': 12, 'C_questions': 12,
                       'editable_edges': 6, 'quality_slots': 7,
                       'slot_0': 'stay', 'slots_1_to_6': 'physical editable edge ascending'},
        'input_schema': {
            'I_A_gain': {'shape': [8, 12, 6], 'dtype': 'float64', 'values': [-1, 0, 1]},
            'C_A_quality': {'shape': [8, 12, 7], 'dtype': 'float64', 'values': [0, 1]},
            'C_B_quality': {'shape': [8, 12, 7], 'dtype': 'float64', 'values': [0, 1]},
            'I_A_menu_tokens': {'shape': [8, 12], 'dtype': 'int64', 'bounds': [1, 175616]},
            'C_A_menu_tokens': {'shape': [8, 12], 'dtype': 'int64', 'bounds': [1, 175616]},
            'C_B_menu_tokens': {'shape': [8, 12], 'dtype': 'int64', 'bounds': [1, 175616]},
            'C_AB_joint_tokens': {'shape': [8, 12], 'dtype': 'int64', 'bounds': [1, 309632]},
            'C_AB_joint_calls': {'shape': [8, 12], 'dtype': 'int64', 'values': [37]},
            'I_qid_hash': {'shape': [8, 12], 'dtype': 'U64', 'unique': 'all192 I/C hashes globally distinct'},
            'C_qid_hash': {'shape': [8, 12], 'dtype': 'U64', 'unique': 'all192 I/C hashes globally distinct'},
            'I_seed': {'shape': [8, 12], 'dtype': 'int64', 'episode_e': '3701+e'},
            'C_seed': {'shape': [8, 12], 'dtype': 'int64', 'episode_e': '3701+e; same for C A/B worlds'},
            'editable_edges': {'shape': [8, 6], 'dtype': 'int64', 'values': 'exact frozen graph template'},
            'dirty': {'shape': [8, 6], 'dtype': 'int64', 'values': 'exact frozen graph template'},
            'A_masks': {'shape': [8, 12], 'dtype': 'int64', 'values': 'exact frozen graph template'},
            'B_masks': {'shape': [8, 12], 'dtype': 'int64', 'values': 'exact frozen graph template'},
        },
        'estimators': {
            'm': 12, 'n_transport': 3, 'n_direct': 5,
            'source': 'mean I_A_gain over all 12 independent I questions',
            'A_gain': 'C_A_quality[...,1:7] - C_A_quality[...,0:1]',
            'B_gain': 'C_B_quality[...,1:7] - C_B_quality[...,0:1]',
            'drift': 'B_gain - A_gain on identical C question/seed',
            'transport': 'source + mean first 3 training rows of drift',
            'direct': 'mean first 5 training rows of B_gain',
            'general_cv': 'mean first 3 B_gain - beta*(mean first 3 A_gain - source)',
            'no_bank_cold_direct': 'Same direct action, separate cost accounting; never replaces the primary baseline.',
            'editor': 'max gain strictly greater than 0; exact ties choose smallest physical edge; else stay',
        },
        'risk_estimation': {
            'v_A': 'I_A_gain sample variance over 12 I rows, ddof=1, per episode/edge',
            'v_B': 'B_gain sample variance over all 12 C rows, ddof=1, per episode/edge',
            'v_D': 'drift sample variance over all 12 C rows, ddof=1, per episode/edge',
            'transport_cell': 'v_A/12 + v_D/3', 'direct_cell': 'v_B/5',
            'primary_ratio': 'unweighted mean of all 48 transport cells / unweighted mean of all 48 direct cells',
            'zero_cells': 'Retain all cells and episodes; never divide each cell or drop zero cells.',
            'finite_population_correction': 'None; plug-in iid-source projection, not an unbiased finite-population guarantee.',
            'risk_definition': 'Conditional-graph repeated-sample variance proxy; not squared error against known population credit.',
            'bank_necessary_count': 'Count and report v_A/12 < v_B/5 across all 48 cells; zero=zero fails strict inequality.',
        },
        'cost_estimation': {
            'exact_label': '初始化+反馈的实测token成本',
            'includes_future_service': False,
            'I_bank_episode': 'sum observed 12 I cold-A-menu token counts',
            'A_single_question': 'cold A base logical tokens + six A toggle incremental tokens',
            'B_single_question': 'cold B base logical tokens + six B toggle incremental tokens, never cached B16-call surrogate',
            'AB_single_question': 'observed A/B union executed tokens, 37 unique calls per C question for fixed graph template',
            'transport_new_episode': '3 * mean all12 C AB union tokens',
            'direct_new_episode': '5 * mean all12 C cold-B-menu tokens',
            'primary_cost_ratio': 'mean_episode(mean_edge(transport_risk)*[I_bank+transport_new]) / mean_episode(mean_edge(direct_risk)*[I_bank+direct_new])',
            'incremental_ratio': 'Same products using only transport_new/direct_new; descriptive, not primary',
            'cold_direct_reference': 'Report direct new token costs without I bank separately; no equivalent-initialization claim',
            'future_service_sensitivity': 'Add the common prospective S24 cap 1800192 to both acquisition totals only as upper-cap sensitivity, not actual service cost.',
            'from_scratch_coefficient': 'mean cells (sqrt(v_A*c_A)+sqrt(v_D*c_D))^2 / mean cells(v_B*c_B); c are episode mean per-question token costs; continuous allocation envelope, not implemented policy',
            'latency_claim': 'No cold-B GPU latency claim from cached outputs; no token-to-GPU conversion assumption.',
            'actual_policy_expenditure': 'May differ; common conservative cap is not equal actual expenditure.',
        },
        'folds': original['exploratory_R1_diagnostic']['folds'],
        'decision_estimation': {
            'methods': ['transport_beta1', 'direct_C5', 'old_bank', 'cv_leave_one_episode_out',
                        'beta_half', 'beta_zero_C3'],
            'per_fold': 'Build all method actions from I and fixed training rows before querying heldout quality.',
            'heldout': 'All 6 B-quality rows of the other fold; same rows across all methods.',
            'episode_metric': 'Mean of two fold accuracy means; all 12 C questions each validated once.',
            'population_metric': 'Unweighted mean over all 8 episode metrics.',
            'independence': 'Folds share I bank; episodes have graph symmetries; this is exploratory R1 development reuse, not new confirmation.',
            'beta_fit': 'One global scalar per heldout episode, estimated from the other 7 episodes only. Average ddof=1 C_A sample variances and C_A/C_B sample covariances over their 42 episode-edge cells (center within each episode and edge); beta=cov/[variance*(1+3/12)].',
            'beta_no_clipping': True,
            'beta_degenerate': 'If mean source variance <=1e-12 set beta=0 and flag degeneration; this secondary baseline cannot rescue primary gates.',
            'deployment_beta_proposal': 'Same formula fitted once to all 8 R1 episodes for a later independent pilot; report but never substitute for primary beta1.',
            'beta_uses_I_or_heldout_S': False,
            'decision_uncertainty': 'No confirmatory interval; report all 8 episode contrasts and two folds, no winner search.',
        },
        'bootstrap': {
            'seed': 72031, 'replicates': 10000, 'interval': [0.05, 0.95], 'quantile_method': 'linear',
            'rng': 'numpy.random.default_rng',
            'exact_order': 'For each replicate sequentially draw episode indices size8, then I-row indices shape(8,12), then C-row indices shape(8,12). Each selected episode occurrence gets independent row draws; C A/B use the same row indices.',
            'recomputed': 'ddof=1 v_A, v_B, v_D and ratio of their all-cell means',
            'costs_resampled_for_primary_interval': False,
            'undefined_denominator': 'Any replicate with mean(v_B/5)<=1e-4 is invalid, including near-zero, zero, nonpositive or nonfinite denominators. Record count/fraction; do not drop or regularize.',
            'invalid_effect': 'Any invalid replicate makes primary interval null and G2 INCONCLUSIVE, even if finite replicates look favorable.',
            'scope': 'Eight-cluster, small-question exploratory investment screen; no coverage or Type-I-error guarantee.',
        },
        'gates': {
            'G0': {'pass': 'Exact allowlisted complete arrays, immutable receipts/hash bindings, question disjointness, fixed seeds, matching metadata and cost provenance; no S inputs.',
                   'failure': 'QUARANTINE_INPUT_NO_METHOD_CONCLUSION'},
            'G1': {'nonzero_B_questions_min': 12, 'episodes_with_nonzero_B_variance_min': 4,
                   'mean_direct_risk_strict_min': 0.0001,
                   'failure': 'INCONCLUSIVE_LOW_SUPPORT'},
            'G2': {'risk_ratio_max_inclusive': 0.80, 'bootstrap_upper_strict_max': 1.0,
                   'initialization_plus_feedback_cost_weighted_ratio_max_inclusive': 1.0,
                   'invalid_bootstrap_replicates_allowed': 0,
                   'clear_no_go_definition': 'Valid 90% interval lower bound >=1.0; otherwise failed G2 is INCONCLUSIVE_CURRENT_BANK_AND_QUOTA.',
                   'pass': 'All listed inequalities and valid denominators hold.'},
            'G3': {'mean_transport_minus_direct_min_inclusive': 0.01,
                   'mean_transport_minus_old_min_inclusive': 0.0,
                   'episodes_nonnegative_transport_minus_direct_min': 6,
                   'clear_no_go_definition': 'Negative mean transport-direct or transport-old; descriptive practical rejection, not a significance test.',
                   'otherwise_failure': 'INCONCLUSIVE_DECISION_VALUE'},
            'G4': {'owner': 'root automatic resource and throughput audit, no new user approval needed',
                   'pass': 'New complete pilot fits pre-reserved16GPUh or prefrozen<=32GPUh and remaining global160GPUh/5round budget; no outcome-based extension.',
                   'this_program': 'NOT_EVALUATED_RESOURCE_RECEIPT_REQUIRED; never launches GPU.'},
            'program_pass_label': 'ELIGIBLE_FOR_G4_RESOURCE_SCREEN',
            'final_gpu_go': 'Only root can record GO_NEW_INDEPENDENT_DEVELOPMENT_PILOT after G0-G4 pass.',
            'project_stop': False,
            'candidate_failure': 'Retain this route and diagnostics; continue a distinct authorized candidate within global limits, without automatically ending project.',
        },
        'fixed_secondary': {
            'bank_sizes': [12, 24, 48], 'high_feedback': {'m': 12, 'transport_n': 12, 'direct_n': 12},
            'm24_m48': 'Algebraic extrapolation using observed I variance; no new bank observed and no gate substitution.',
            'slices': ['dirty', 'non_dirty'], 'slices_are_validity_certificates': False,
            'no_change': 'Replace B-quality by the very same A-quality array, so drift exactly zero by construction. Verify beta1=old bank and separately evaluate pooled independent old+new3 mean on heldout A. Do not call copied A an independent new world.',
            'same_panel_identity': 'mean(A_gain)+mean(B_gain-A_gain)=mean(B_gain), verified numerically.',
            'risk0over0': 'null/INCONCLUSIVE, never zero benefit',
        },
        'formal_pilot': original['formal_pilot_outline'],
        'scientific_claim_boundary': 'No new estimator theory; no claim of system improvement without fresh service and full measured costs. Diagnostic success is only permission to spend a bounded research budget.',
    }
    json_path = ROOT / 'FROZEN_DIAGNOSTIC_CONTRACT.json'
    md_path = ROOT / 'FROZEN_DIAGNOSTIC_CONTRACT.md'
    if json_path.exists() or md_path.exists():
        raise SystemExit('Frozen files already exist; do not overwrite a prospective contract')
    raw = (json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with json_path.open('xb') as handle:
        handle.write(raw)
    md = '''# R2 已冻结的 I/C 低成本诊断协议\n\n状态：真实 I/C 读取前冻结；只用于研发投入筛选，不是统计确认门。原草案保留。\n\n机器可执行定义以同目录 FROZEN_DIAGNOSTIC_CONTRACT.json 为准，SHA256：{hash}。\n\n主档固定为独立 I12 银行、3 个新配对漂移问题、5 个 direct 新问题。主方差分别来自 I12 的 A 增益和 C12 配对的 B 增益/漂移；全部 48 个单元保留。二折行序、1pp/20%门、支持度阈值、10,000 次 seed72031 bootstrap 均在 JSON 中冻结。任何 bootstrap 分母不超过 1e-4 时，不删除复制，主区间记空且结论不确定。\n\n成本正式用语为“初始化+反馈的实测token成本”，不含未来服务。未来 S24 共同上限只作敏感性分析，不能代替完整系统的实测服务成本。直接法使用冷 B base 加增量的 token；不得把共享 A 缓存后的 B 成本当作冷启动成本。\n\n通用 CV 系数只在另外七个 episode 的 C 配对统计上拟合，按 episode/edge 内中心化后平均方差与协方差；不裁剪系数，退化分母按预设规则置零并标记。它和其他敏感性档均不能替代主 beta=1 候选的投入门。\n\n输入只允许绑定哈希的 I/C 专用数组、费用与哈希标识，不含 S、原题、gold 或模型文本。只有 R1 采购和服务动作已冻结，才可导出。该程序最多报告 ELIGIBLE_FOR_G4_RESOURCE_SCREEN；G4 由主任务核对真实资源、吞吐和剩余预算，程序不启动 GPU、不创建新许可要求。\n\n所有数值、异常、重采样顺序、系数口径与 go/no-go 定义已在看到真实数组前固定。单候选失败仅结束本银行/配额路线，项目继续其他获授权候选；总上限仍为160 GPU小时、最多5轮正式候选。\n'''.format(hash=digest_bytes(raw))
    with md_path.open('x', encoding='utf-8') as handle:
        handle.write(md)
    print(json.dumps({'status': contract['status'], 'sha256': digest_bytes(raw)}, ensure_ascii=False))


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    main()
