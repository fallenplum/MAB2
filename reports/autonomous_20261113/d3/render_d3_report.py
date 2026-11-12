"""Render the aggregate-only D3 development report.

python render_d3_report.py --results D3_RESULTS.json --review D3_REPLAY_SUMMARY.json --out .
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=['graph_only_order2','graph_only_order1','B0','fixed_old_C2_R3']
LABELS={'graph_only_order2':'Graph-only order 2 *','graph_only_order1':'Graph-only order 1 (control)',
        'B0':'B0 tied-index control','fixed_old_C2_R3':'Frozen D2 C2_R3 (control)'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fmt(v):return f'{v:.9f}'


def main():
    start=time.time();cpu=time.process_time()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('D3_RESULTS.json'))
    p.add_argument('--review',type=Path,default=Path('D3_REPLAY_SUMMARY.json'))
    p.add_argument('--out',type=Path,default=Path('.'))
    a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    d=json.loads(a.results.read_text(encoding='utf-8'));r=json.loads(a.review.read_text(encoding='utf-8'))
    assert d['spending_gate']['status']=='STOP_CURRENT_CANDIDATE_ROUTE' and not d['spending_gate']['pass']
    assert r['status']=='PASS_WITH_EXPLORATORY_LIMITATIONS' and d['models']['2']['alpha']==1000
    assert not d['independent_confirmation'] and d['new_GPU_calls']==0
    rows=[]
    for split in ['tune','exposed']:
        for name in ORDER:
            v=d['drift_results'][split][name];ci=v['enrichment']
            rows.append({'split':split,'selector':name,'candidate_eligible':name=='graph_only_order2','questions':v['question_count'],
                'k':v['k'],'selected_U_mean':v['selected_squared_drift_mean'],
                'uniform_top2_expected_U_mean':v['uniform_top2_expected_squared_drift_mean'],
                'enrichment':ci['estimate'],'ci_low':ci['interval95'][0],'ci_high':ci['interval95'][1],
                'negative_U_coordinate_count':v['negative_squared_drift_estimate_coordinates'],
                'exact_all_score_tie_questions':v['all_scores_exactly_tied_questions'],
                'bootstrap_resamples':ci['resamples'],'bootstrap_seed':ci['seed'],
                'bootstrap_unit':'question','evidence_role':'exposed_development' if split=='exposed' else 'tuning_descriptive',
                'negative_estimates_clipped':False,'independent_confirmation':False})
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    alpha_rows=[{'order':order,'alpha':trial['alpha'],'tune_credit_mse':trial['tune_credit_mse'],
                 'selected':trial['alpha']==d['models'][order]['alpha']} for order in ['1','2'] for trial in d['models'][order]['alpha_trials']]
    with (out/'alpha_paths.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(alpha_rows[0]));w.writeheader();w.writerows(alpha_rows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'svg.fonttype':'none','svg.hashsalt':'mab2-d3-diagnostic-v1','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8.6,4.5));fig.subplots_adjust(left=.32,right=.96,top=.79,bottom=.25)
    ax.axhspan(-.4,.4,color='#EDF0F2');ax.axvline(0,color='#444444',lw=.9)
    for i,name in enumerate(ORDER):
        v=d['drift_results']['exposed'][name]['enrichment'];point=v['estimate']*1000;lo,hi=[x*1000 for x in v['interval95']]
        color='#386B91' if i==0 else '#777777'
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o' if i==0 else 'D',
                    color=color,mfc=color if i==0 else 'white',mec=color,ms=6,capsize=3,lw=1)
    ax.set_yticks(range(4),[LABELS[n] for n in ORDER]);ax.set_ylim(3.5,-.6)
    ax.grid(axis='x',color='#E0E3E6',lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel('Top-2 minus uniform expected squared-drift estimate (× 10³)')
    fig.suptitle('D3: enrichment remains unresolved; the spending gate stops',y=.95,fontsize=12,fontweight='bold')
    fig.text(.32,.84,'Exposed development: n = 120 questions; top 2 of 12 edges',fontsize=9)
    fig.text(.03,.115,'95% whole-question bootstrap intervals; 10,000 resamples. Auxiliary controls are descriptive.',fontsize=8)
    fig.text(.03,.07,'* Only graph-only order 2 is candidate eligible. Negative U-statistic estimates are retained.',fontsize=8)
    fig.text(.03,.025,'No new GPU pilot is authorized by this failed gate; this does not establish universal impossibility.',fontsize=8)
    fig.savefig(out/'d3_enrichment.png',dpi=220,facecolor='white',metadata={'Software':'MAB2 D3 report generator'})
    fig.savefig(out/'d3_enrichment.svg',facecolor='white',metadata={'Date':None,'Creator':'MAB2 D3 report generator'})
    plt.close(fig)
    primary=d['drift_results']['exposed']['graph_only_order2'];en=primary['enrichment'];gate=d['spending_gate']
    table=['| 数据角色 | 选择器 | 题数 | 所选U均值 | uniform期望U均值 | 富集均值 [95% CI] | 负U坐标数 | 全分数并列题数 |',
           '|---|---|---:|---:|---:|---|---:|---:|']
    for row in rows:
        table.append(f"| {row['split']} | {row['selector']} | {row['questions']} | {fmt(row['selected_U_mean'])} | {fmt(row['uniform_top2_expected_U_mean'])} | {fmt(row['enrichment'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['negative_U_coordinate_count']} | {row['exact_all_score_tie_questions']} |")
    report=f"""# MAB2 D3：图条件信用漂移富集的最后一轮CPU诊断

作者侧内部报告草稿；机器独立复算完成，人工作者审阅待完成。使用已曝光G2数据，不是新独立确认，也不是完整论文。

**唯一预定候选未通过算力准入门：STOP_CURRENT_CANDIDATE_ROUTE；统计分类为INCONCLUSIVE，点估计方向为负。**

## 1. 预定问题与执行范围

本轮检验不使用问题文本的共享二阶图模型，能否在另一道题的两个图背景之间识别信用变化较大的边。训练仍为200题的24个预定存在信用方向，逐题总权重1；80题用于固定8点α网格的信用MSE选择，不重拟合。旧120题只承担开发评价。所选模型含79个图特征，信用差中有效参数为78维；最终α={d['models']['2']['alpha']:g}，位于既定网格上界。不得据此事后扩展网格或新增候选。

模型根据冻结系数与当前图掩码计算两锚点预测存在信用差的绝对值，固定选top-2；同题标签不进入选择。α由信用MSE确定，算力门只看漂移富集，不要求当前动作收益或全局MSE改善。两类指标的角色在协议中预先区分。

## 2. 主统计量及负值解释

令同一个种子下两锚点信用差为δ_s，使用U=(1/10)Σ_(s<t)δ_sδ_t估计平方漂移；每题对所选两边取均值，再减去全部12边均值。后者等于均匀不放回选择两边的期望，不是另抽一套随机策略造成的Monte Carlo波动。

在独立种子重复的假设下，U估计的是条件均值的平方；有限5种子估计可以为负。**所选U均值{fmt(primary['selected_squared_drift_mean'])}是去噪估计值，不能称为负的真实平方，也不能截断为零。**本轮保留所有负值；在120题×12边的评价坐标中，有{primary['negative_squared_drift_estimate_coordinates']}个U估计为负。

预定候选的uniform期望U均值为{fmt(primary['uniform_top2_expected_squared_drift_mean'])}；主富集为{fmt(en['estimate'])}，95%题级bootstrap区间[{fmt(en['interval95'][0])}, {fmt(en['interval95'][1])}]。区间跨零，因此是证据不确定，而不是已证明反向选择、无效或等效。

## 3. 全部候选与控制结果

{chr(10).join(table)}

唯一可触发下一阶段的候选是graph_only_order2。graph_only_order1、B0和冻结D2 C2_R3均为辅助控制，不能作为失败后的替补。前两者的图间漂移分数恒为零，按规则选择最小边序号；这种并列索引控制不具有漂移排序信息，不能与uniform期望混称。所有辅助区间均为描述性未校正区间。

![D3所有冻结选择器的富集区间](d3_enrichment.png)

图1：已曝光120题上top-2相对uniform期望的富集均值与95%题级bootstrap区间。整题重采样10,000次，种子59173；两锚点构成题内比较，不把240锚点当独立样本。灰背景仅标识唯一候选，不是显著性标记。完整8行数据见[metrics.csv](metrics.csv)，完整两模型α路径见[alpha_paths.csv](alpha_paths.csv)。

## 4. 准入判据与停止范围

| 条件 | 实际状态 |
|---|---|
| 来源、冻结与独立重放通过 | 通过 |
| 主富集点估计>0 | 不通过 |
| 主富集95%区间下界>0 | 不通过 |
| 所选U均值>0 | 不通过 |
| 独立GPU pilot额度 | {gate['independent_pilot_gpu_hour_cap']} GPU小时 |
| 允许控制替补 | 否 |

因此停止**当前固定候选路线**，不更换k、分数、α准则或控制组来争取通过。该停止不意味着所有信用维护与反馈采集方法都不可能有效。已曝光数据上的95%区间也不是“方法有95%概率成功”；这是有限资源分配的开发判据，不是新样本确认。

即使未来另一份合法协议显示漂移富集为正，也仍需验证这种信息能否改善未来题决策，并计入初始化、服务、复检、评分及失败成本。漂移幅度本身不是误删风险、信息收益或净服务收益。

## 5. 独立复核与成本

独立复核不导入生产评分模块，从已验收派生数组构造完整图势特征再作差；对两个阶数完整重拟合8个α，独立重算top-2、10个跨种子乘积、题级bootstrap与算力门。共{r['numerical_checks']}项数值检查，最大差{r['maximum_numerical_difference']:.3g}，拟合、冻结、统计和STOP结论一致。原题、gold与新测试数据未读取。

| 支出 | 数值 | 口径 |
|---|---:|---|
| D3主进程CPU | {d['process_cpu_seconds']:.6f} s | 从进程起点计，含线程 |
| D3墙钟 | {d['wall_seconds']:.6f} s | 独立记录 |
| Git子进程CPU | 未测；上界{d['git_cpu_upper_bound_seconds']:.6f} s | 计入保守资源账本 |
| 既往D3尝试CPU上界 | {d['previous_attempts_cpu_upper_bound_seconds']:.6f} s | 累计尝试 |
| 本次独立复核CPU | {r['costs']['review_main_process_cpu_seconds']:.6f} s | 含导入，无子进程 |
| 新GPU调用 / GPU小时 | 0 / 0 | 无新增推理 |
| 准备命令、代理推理、人工与能耗 | 未计量 | JSON中保留null，不假设免费 |

这不是整个历史项目的完整成本汇总。报告生成的可测开销另见[costs.json](costs.json)；未计量部分不据此推算金额或回本点。

## 6. 证据等级、优势与局限

算子与算术一致性置信度高，范围是已验收派生输入及固定协议；它不代表生成数据重新评分或独立科学复现。富集效应方向的证据不确定；跨任务、异质模型、恢复与在线完整预算收益未建立。

优势是有限单候选、标签不进入当前排序、完整α路径可查、控制不替补和负U保留。局限是评价块已曝光，主估计稀疏且区间跨零；当前强正则模型没有通过预定资源门。报告不将α上界或失败解释成可以无限追加搜索的理由。

## 7. 复现与归档

用`python render_d3_report.py --results D3_RESULTS.json --review D3_REPLAY_SUMMARY.json --out .`重建报告和图。两个JSON是脱敏汇总，只包含聚合数值、相对代码文件名和内容哈希。原D3聚合结果SHA256为`{d['original_results_sha256']}`；原独立审计SHA256为`{r['original_review_sha256']}`。

`review_d3_real.py`是已经运行的参数化独立复算脚本，按原字节保留，源SHA与复核记录一致；它要求本地受控派生输入，不随此包公开逐题数组。它生成的完整审计含输入定位，应留在私有记录中，公开时使用本包的脱敏摘要。没有修改已经验证的计算算子。

本报告使用AI辅助与Python绘图，作者人工核阅待完成；非期刊外审，非ACL-ready声明。审查边界见[INTERNAL_REVIEW.md](INTERNAL_REVIEW.md)。
"""
    (out/'D3_REPORT.md').write_text(report,encoding='utf-8')
    (out/'INTERNAL_REVIEW.md').write_text("""# D3作者侧内部审查摘要

机器独立计算审查完成；作者人工审阅待完成。本记录不是期刊外审或编辑决定。

1. **实现与预定估计量一致。** 独立正规方程重拟合完整α路径，完整图势切换构造信用设计，标量排序与跨种子U统计重放一致；没有生产评分器导入、同题标签选择或新数据访问。
2. **保留失败。** 唯一预定二阶图选择器未通过资源门；点估计为负但区间跨零，分类应为INCONCLUSIVE。不得使用控组替补、放宽阈值、改变k或扩α网格。
3. **负U不是负平方。** U是有限种子的去噪估计，可以为负；保持原始负值是正确处理，截断会改变目标。不能从负点估计推出真实漂移不存在。
4. **图漂移不等于决策价值。** 即便富集显著，仍需后续合法的新样本检验，证明未来题服务或误删决策改进并统一完整成本。当前结果没有完成这一链条。
5. **统计与信息范围明确。** 两锚点在同题内比较，重采样单位为题；已曝光120题不是封存测试；95%区间不是成功概率。辅助区间不增加准入门，控组不具有替补资格。
6. **资源与归档合规。** 本轮无新增GPU调用，算力门输出pilot额度0。CPU记录区分实测、保守上界和未计量；原输出不改写，完整审计留私有，公开包只发布汇总与可参数化代码。

优势是单候选和独立数值复核可追溯，风险是已曝光、稀疏信号和未完成系统实证。建议按合同停止当前候选路线，保留原科学问题及负结果；不将本轮结论升级成所有图条件信用维护不可能成立，也不生成完整成功论文。
""",encoding='utf-8')
    cost={'schema':'mab2_d3_report_cost_v1','successful_render_main_CPU_seconds':time.process_time()-cpu,
          'render_wall_seconds':time.time()-start,'scope':'excludes_imports_final_manifest_and_prior_QA_attempts',
          'preparation_commands_CPU_seconds':None,'agent_or_human_cost':None,'new_GPU_calls':0,'new_GPU_hours':0}
    dump(out/'costs.json',cost)
    assert len(rows)==8 and len(alpha_rows)==16
    original_script_hash=r['source_script_sha256']
    assert sha(out/'review_d3_real.py')==original_script_hash
    patterns=[r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]',r'\b(?:\d{1,3}\.){3}\d{1,3}\b',r'hf_[A-Za-z0-9]{15,}',r'ssh\s+-p']
    findings=[]
    for f in out.iterdir():
        if f.suffix in {'.py','.json','.md','.csv','.svg'}:
            text=f.read_text(encoding='utf-8')
            if any(re.search(pat,text) for pat in patterns):findings.append(f.name)
    assert not findings,findings
    dump(out/'validation.json',{'status':'PASS_MACHINE_CHECKS_HUMAN_REVIEW_PENDING','metric_rows':8,'alpha_path_rows':16,
        'privacy_findings':findings,'original_replay_script_unchanged':True,'submission_ready':False,
        'figure_visual_review':'pending','matplotlib_version':matplotlib.__version__,'data_sources':{a.results.name:sha(a.results),a.review.name:sha(a.review)}})
    dump(out/'artifact_manifest.json',{'schema':'mab2_d3_report_artifact_manifest_v1','files':{f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='artifact_manifest.json'}})
    print(json.dumps({'status':'PASS_MACHINE_CHECKS_HUMAN_REVIEW_PENDING','metric_rows':8,'alpha_path_rows':16,'privacy_findings':0}))


if __name__=='__main__':main()
