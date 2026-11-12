"""Generate an aggregate-only D2 diagnostic report from sanitized evidence.

Usage: python render_d2_report.py --evidence d2_evidence.json --out .
No original questions, individual records, credentials, or network access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ORDER = ["ZERO", "B0"] + [f"C{k}_{v}" for k in (1, 2) for v in ("R0", "R1", "R1b", "R2", "R3")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def display(value, scale=1, digits=4):
    if value is None:
        return "未计算"
    v = value*scale
    if abs(v) < .5*10**(-digits): v = 0.0
    return f"{v:.{digits}f}"


def gain_interval(row):
    ci = row.get("gain_vs_stay")
    if ci is None:
        return "未计算"
    return f"[{display(ci['interval95'][0],100,3)}, {display(ci['interval95'][1],100,3)}]"


def get_metrics(e, name, configuration, split):
    r = e["models"][name]
    baseline = name in ("ZERO", "B0")
    if configuration == "selected" or baseline:
        return r["tune" if split == "tune" else "exposed_evaluation"]
    return r["fixed_alpha_tune" if split == "tune" else "fixed_alpha_exposed_evaluation"]


def alpha(e, name, configuration):
    if name in ("ZERO", "B0"): return None
    return e["models"][name]["alpha" if configuration == "selected" else "fixed_historical_alpha"]


def build_metric_rows(e):
    rows = []
    for name in ORDER:
        for configuration in ("selected", "historical_fixed"):
            for split in ("tune", "exposed"):
                r = get_metrics(e, name, configuration, split)
                ci = r.get("gain_vs_stay")
                uci = r.get("gain_vs_uniform13")
                bci = r.get("gain_vs_B0")
                rows.append({
                    "method":name, "configuration":configuration, "alpha":alpha(e,name,configuration),
                    "baseline_repeated_for_layout":name in ("ZERO","B0") and configuration=="historical_fixed",
                    "split":split, "questions":r["question_count"], "anchors":2*r["question_count"],
                    "coordinates":24*r["question_count"], "seeds_per_coordinate":5,
                    "action_gain":r["mean_action_gain"],
                    "gain_ci_low":None if ci is None else ci["interval95"][0],
                    "gain_ci_high":None if ci is None else ci["interval95"][1],
                    "gain_ci_status":"not_computed_on_tuning_data" if ci is None else "exploratory_question_bootstrap_unadjusted",
                    "credit_risk":r["mean_credit_risk"], "credit_mse":r["mean_credit_mse"],
                    "risk_ci_status":"not_computed_in_D2",
                    "gain_vs_uniform":None if uci is None else uci["estimate"],
                    "uniform_ci_low":None if uci is None else uci["interval95"][0],
                    "uniform_ci_high":None if uci is None else uci["interval95"][1],
                    "gain_vs_B0":None if bci is None else bci["estimate"],
                    "B0_ci_low":None if bci is None else bci["interval95"][0],
                    "B0_ci_high":None if bci is None else bci["interval95"][1],
                    "additions":r["addition_count"], "deletions":r["deletion_count"], "stays":r["stay_count"],
                    "nonzero_observed_mean_support":r["nonzero_observed_mean"],
                    "nonzero_sign_agreements":r["nonzero_sign_agreements"],
                    "rank_defined_anchors":r["rank_defined"], "rank_undefined_anchors":r["rank_undefined"],
                    "mean_defined_spearman":r["mean_defined_spearman"],
                    "prediction_mean":r["mean_prediction"], "prediction_rms":r["prediction_rms"],
                    "positive_prediction_fraction":r["positive_prediction_fraction"],
                    "evidence_ids":"E01;E02", "status":"posthoc_development",
                })
    return rows


def method_table(e, configuration):
    lines = ["| 方法 | α | tune 收益 (pp) | exposed 收益 (pp) [95% CI] | tune 风险 ×10³ | exposed 风险 ×10³ | exposed 删/加/停 |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for name in ORDER:
        t=get_metrics(e,name,configuration,"tune"); x=get_metrics(e,name,configuration,"exposed")
        av=alpha(e,name,configuration)
        lines.append(f"| {name} | {'—' if av is None else f'{av:g}'} | {display(t['mean_action_gain'],100,3)} | {display(x['mean_action_gain'],100,3)} {gain_interval(x)} | {display(t['mean_credit_risk'],1000)} | {display(x['mean_credit_risk'],1000)} | {x['deletion_count']}/{x['addition_count']}/{x['stay_count']} |")
    return "\n".join(lines)


def draw_figure(e, out):
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9, "svg.fonttype":"none", "svg.hashsalt":"mab2-d2-diagnostic-v1", "pdf.fonttype":42,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.8})
    fig, axes = plt.subplots(1,2,figsize=(11.4,6.5),sharey=True,gridspec_kw={"width_ratios":[1,1.2]})
    fig.subplots_adjust(left=.155,right=.985,bottom=.18,top=.87,wspace=.20)
    nominee_index=ORDER.index(e["nomination"]["name"])
    for ax in axes:
        ax.axhspan(nominee_index-.45,nominee_index+.45,color="#EDF0F2",zorder=0)
        ax.grid(axis="x",color="#DDE1E4",linewidth=.5,zorder=0)
    for i,name in enumerate(ORDER):
        color="#777777" if name=="ZERO" else "#333333" if name=="B0" else "#386B91" if name.startswith("C1") else "#B77442"
        configurations=["selected"] if name in ("ZERO","B0") else ["selected","historical_fixed"]
        for cfg in configurations:
            row=get_metrics(e,name,cfg,"exposed")
            y=i if len(configurations)==1 else i+(-.13 if cfg=="selected" else .13)
            marker="o" if cfg=="selected" else "D"
            face=color if cfg=="selected" else "white"
            axes[0].plot(row["mean_credit_risk"]*1000,y,marker=marker,ms=5.1,mec=color,mfc=face,linestyle="none",zorder=3)
            ci=row["gain_vs_stay"]["interval95"]
            point=row["mean_action_gain"]*100
            axes[1].errorbar(point,y,xerr=[[point-ci[0]*100],[ci[1]*100-point]],fmt=marker,
                ms=5.1,mec=color,mfc=face,ecolor=color,elinewidth=.9,capsize=2,zorder=3)
    axes[0].axvline(e["models"]["ZERO"]["exposed_evaluation"]["mean_credit_risk"]*1000,color="#888888",ls="--",lw=.8)
    axes[1].axvline(0,color="#555555",lw=.9)
    axes[0].set_xlim(0,2.4)
    axes[0].set_yticks(range(len(ORDER)),[n+(" *" if n==e["nomination"]["name"] else "") for n in ORDER])
    axes[0].set_ylim(len(ORDER)-.45,-.6)
    axes[0].set_xlabel("Cross-seed credit risk (× 10³; lower is better)")
    axes[1].set_xlabel("One-edit quality gain vs stay (percentage points)")
    axes[0].set_title("a  Credit risk: point estimates",loc="left",fontweight="bold")
    axes[1].set_title("b  Action gain: exploratory 95% intervals",loc="left",fontweight="bold")
    fig.suptitle("D2: risk reduction does not establish benefit for the nominated policy",fontsize=12,fontweight="bold",y=.965)
    handles=[Line2D([],[],marker="o",mfc="#555555",mec="#555555",ls="none",label="Tune-selected α"),
             Line2D([],[],marker="D",mfc="white",mec="#555555",ls="none",label="Historical fixed α")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.58,.077),ncol=2,frameon=False)
    fig.text(.155,.045,"Exposed evaluation: 120 questions, 2 anchors, 5 seeds. Whole-question bootstrap; unadjusted intervals.",fontsize=8)
    fig.text(.155,.020,"* C1_R1 was nominated using tune data; its development filter failed. No risk intervals were computed.",fontsize=8)
    fig.savefig(out/"d2_risk_and_action.png",dpi=220,facecolor="white",metadata={"Software":"MAB2 diagnostic report generator"})
    fig.savefig(out/"d2_risk_and_action.svg",facecolor="white",metadata={"Date":None,"Creator":"MAB2 diagnostic report generator"})
    plt.close(fig)


def main():
    t0,cpu0=time.time(),time.process_time()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,default=Path("d2_evidence.json"))
    parser.add_argument("--out",type=Path,default=Path("."))
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    e=json.loads(args.evidence.read_text(encoding="utf-8"))
    assert e["schema"]=="mab2_d2_sanitized_report_evidence_v1"
    assert set(e["models"])==set(ORDER) and e["nomination"]=={"name":"C1_R1","passes_development_filter":False,"fallback_used":False}
    assert e["independent_replay"]["checks"]==914
    rows=build_metric_rows(e);write_csv(out/"method_metrics.csv",rows)
    selected=[r for r in rows if r["configuration"]=="selected" and r["split"]=="exposed"]
    fixed=[r for r in rows if r["configuration"]=="historical_fixed" and r["split"]=="exposed"]
    write_csv(out/"selected_exposed.csv",selected);write_csv(out/"fixed_exposed.csv",fixed)
    const_rows=[]
    for name,c in e["constant_response_control"].items():
        for split,components in c["by_partition"].items():
            for component in ["direct","constant","residual"]:
                const_rows.append({"method":name,"split":split,"component":component,
                    "historical_alpha":c["alpha"],"training_constant_response":c["weighted_target_mean"],
                    **components[component],"evidence_ids":"E01;E02"})
    write_csv(out/"constant_controls.csv",const_rows)
    noise_rows=[]
    for split,r in e["noise_by_partition"].items():
        for statistic in ["naive_observed_menu_oracle","leave_one_seed_out_selection_gain"]:
            v=r[statistic]
            noise_rows.append({"split":split,"statistic":statistic,"questions":v["question_count"],"estimate":v["estimate"],
                "ci_low":v["interval95"][0],"ci_high":v["interval95"][1],"samples":v["samples"],"seed":v["seed"],
                "folds_independent":False,"feedback_acquisition_cost_included":False,"future_question_service_benefit":False,"evidence_ids":"E01;E02"})
    write_csv(out/"noise_and_loso.csv",noise_rows)
    n=e["models"]["C1_R1"]["exposed_evaluation"]
    c1=e["constant_response_control"]["C1_R0"]["by_partition"]["test"]
    c2=e["constant_response_control"]["C2_R0"]["by_partition"]["test"]
    loso=e["noise_by_partition"]["test"]["leave_one_seed_out_selection_gain"]
    claims=[
        {"id":"C01","type":"result","confidence":"高（计算重放范围）","evidence":"E01;E02","locator":"nomination;models.C1_R1.exposed_evaluation","text":f"预定提名为C1_R1，开发过滤失败且无替补；其已曝光评价收益为{display(n['mean_action_gain'],100,3)} pp，95%区间为{gain_interval(n)} pp。"},
        {"id":"C02","type":"controlled_diagnostic","confidence":"高（固定ridge与数据几何范围）","evidence":"E01;E02;E03","locator":"constant_response_control","text":f"在图无关的常数响应控制中，原R0惩罚所有系数产生C1平均信用{c1['constant']['mean']:.8f}、C2平均信用{c2['constant']['mean']:.8f}，两者信用均为正。"},
        {"id":"C03","type":"attribution_boundary","confidence":"高（固定α对照）","evidence":"E01;E02;E03","locator":"models.C1_R1.fixed_alpha_exposed_evaluation;models.C2_R1.fixed_alpha_exposed_evaluation","text":"保持历史α不变，免罚全局截距仍降低本批数据上的信用风险；C2再从α=100改选1000的额外变化必须与截距因素分开。"},
        {"id":"C04","type":"nonclaim","confidence":"高（证据边界）","evidence":"E01;E02;E03","locator":"noise_by_partition.test;split_seed_diagnostic","text":f"LOSO同题诊断为{display(loso['estimate'],100,3)} pp，95%区间[{display(loso['interval95'][0],100,3)}, {display(loso['interval95'][1],100,3)}] pp；它未计全菜单反馈成本，不证明未来新题上的服务收益。"},
        {"id":"C05","type":"method_fidelity","confidence":"高（协议及历史重放范围）","evidence":"E02;E03","locator":"variants.R0;independent_refits","text":"R0全系数惩罚是原冻结目标；本次发现属于原目标的统计性质，不是旧实现违反原合同。"},
        {"id":"C06","type":"readiness","confidence":"高（缺失证据范围）","evidence":"E01;E02;E03","locator":"public_boundaries;nomination","text":"尚无新独立确认、完整成本在线机制收益或恢复证据；本报告不构成完整论文，也不支持ACL投稿就绪声明。"},
    ]
    claim_rows=[{"claim_id":c["id"],"text_sha256":hashlib.sha256(c["text"].encode()).hexdigest(),"type":c["type"],"evidence_ids":c["evidence"],"locator":c["locator"],"confidence":c["confidence"],"human_verification":"pending"} for c in claims]
    write_csv(out/"claims.csv",claim_rows)
    write_json(out/"claim_evidence.json",claims)
    write_json(out/"source_manifest.json",{"sources":e["source_manifest"],"verification_scope":"machine replay completed; author-side human review pending","source_code_sha256":e["source_code_sha256"]})
    const_table=["| 原模型 | 原响应信用均值 | 常数分量均值 | 常数分量为正 | 残差分量均值 |","|---|---:|---:|---:|---:|"]
    for name,c in [("C1_R0",c1),("C2_R0",c2)]:
        const_table.append(f"| {name} | {display(c['direct']['mean'],1,8)} | {display(c['constant']['mean'],1,8)} | {display(c['constant']['positive_fraction'],100,1)}% | {display(c['residual']['mean'],1,8)} |")
    attr=["| 对照 | α | exposed 风险 ×10³ | exposed 收益 (pp) [95% CI] |","|---|---:|---:|---|"]
    for name,cfg in [("C1_R0","selected"),("C1_R1","historical_fixed"),("C2_R0","selected"),("C2_R1","historical_fixed"),("C2_R1","selected")]:
        r=get_metrics(e,name,cfg,"exposed")
        attr.append(f"| {name} / {cfg} | {alpha(e,name,cfg):g} | {display(r['mean_credit_risk'],1000,6)} | {display(r['mean_action_gain'],100,3)} {gain_interval(r)} |")
    noise_table=["| 数据角色 | 题数 | 五种子同菜单最大观测收益 (pp) | LOSO收益 (pp) [95% CI] |","|---|---:|---:|---|"]
    for split in ["train","tune","test"]:
        r=e["noise_by_partition"][split];l=r["leave_one_seed_out_selection_gain"]
        noise_table.append(f"| {'exposed' if split=='test' else split} | {l['question_count']} | {display(r['naive_observed_menu_oracle']['estimate'],100,3)} | {display(l['estimate'],100,3)} [{display(l['interval95'][0],100,3)}, {display(l['interval95'][1],100,3)}] |")
    report=f"""# MAB2 D2：信用预测失败的有限根因诊断

作者侧内部科研报告草稿；机器复算完成，作者人工审阅待完成。证据等级为已曝光数据上的事后开发诊断；不是期刊外审、独立确认或投稿论文。

**结论：原R0的基准质量惩罚会产生正信用分量；去掉该偏置降低风险，却未得到通过预定开发过滤的预测器。** [evidence:E01,E02]

## 1. 范围、目标与来源

本轮复用G2的200题训练、80题调参及已曝光120题评价；每题两个锚点、每锚点12条单边比较、每图5个冻结种子。共10,394个唯一题—图行及51,970条历史输出；没有新生成调用。本轮不打开最终封存测试。原始题目、答案和逐题数据不包含在本报告包内。

主要开发指标是单次切边或保持的平均质量收益：使用预测存在信用乘以切换方向，选严格为正的最大增益，否则保持；质量成本权重为0。α仍仅由tune信用均值MSE选择，固定8点网格、不重拟合。tune动作收益用于8个改动变体中至多一个的提名，随后用已曝光120题做开发过滤，失败不改选第二名。[evidence:E03]

本包的E01为D2聚合结果，E02为独立真实复算，E03为冻结协议；来源摘要及SHA256见[source_manifest.json](source_manifest.json)。独立复算使用派生数组重新构造图特征、以正规方程重拟合选定及历史固定α，重算动作、跨种子风险、bootstrap与LOSO。共{e['independent_replay']['checks']}项检查，最大差{e['independent_replay']['maximum_numerical_difference']:.3g}。未重读原题、gold或原始服务输出；未重新拟合所有候选完整α路径。[evidence:E02]

## 2. 对照与归因

| 变体 | 训练目标中的改变 | 可以检验什么 | 解释限制 |
|---|---|---|---|
| R0 | 原始唯一图质量回归，所有系数L2惩罚 | 原目标重放与锚点 | 基准质量与掩码特征可能分摊常数响应 |
| R1 | 仅全局常数项免罚 | 全局截距惩罚的影响 | 不等于移除问题依赖基准 |
| R1b | 问题基准a0(x)的九列免罚 | 线性问题基准的影响 | 仍受表示能力限制 |
| R2 | 同题X与y同时按唯一图权重中心化 | 去除自由题内基准 | 与R3的观测零空间不同 |
| R3 | 使用同批图构造原定24条差分 | 决策方向重加权 | 共享图的差分相关；不增加独立样本数 |

所有变体分别用于C1一阶和C2二阶表示，保留ZERO与B0两个基线。两个锚点的比较图可能不连通；R2与R3不能只按“质量目标/信用目标”两个词归因。差分图秩与分量的汇总保留在证据JSON中。有效自由度针对各自准备后的观测算子，不能直接当作共同绝对图质量模型的复杂度。

## 3. 全部12方法：tune选定α

{method_table(e,'selected')}

pp为质量比例差×100，不是相对改善百分比。exposed共120题、240锚点；tune共80题、160锚点。误差区间为10,000次整题bootstrap，两个锚点共同重采样，随机种子59173，探索性且未作多重比较校正。tune收益CI和D2风险CI未计算，不能补造。风险是五种子误差的非重复跨种子乘积均值，不是朴素观测均值MSE；两者均保留在CSV中。

## 4. 全部12方法：历史固定α

{method_table(e,'historical_fixed')}

C1固定α=1000、C2固定α=100。ZERO与B0没有α，重复出现仅便于对照，不是额外实验。全部48个方法×配置×数据角色记录见[method_metrics.csv](method_metrics.csv)，含动作支持数、非零观测符号分母、排序可定义/不可定义锚点数。其浮点原值不被改写；表格仅按显示精度舍入。

## 5. 常数响应控制及固定α证据

{claims[1]['text']} [claim:C02] [evidence:E01,E02]

{chr(10).join(const_table)}

控制响应固定为训练质量的题均加权均值，真实图差由构造为零。在线性ridge下，原响应预测精确分解为常数分量与残差分量。它建立了这个目标与设计矩阵下的伪信用机制，不能把非线性动作选择后的收益也作同样加法归因。

{claims[4]['text']} [claim:C05] [evidence:E02,E03]

{chr(10).join(attr)}

{claims[2]['text']} [claim:C03] [evidence:E01,E02,E03]

## 6. 提名、失败与不确定性

{claims[0]['text']} [claim:C01] [evidence:E01,E02]

C1_R1仍在240个锚点中编辑{n['addition_count']+n['deletion_count']}次（删{n['deletion_count']}、加{n['addition_count']}、保持{n['stay_count']}）。风险缩小没有转化为可靠的编辑方向或弃权能力。其他方法在exposed数据上更高的点估计不能用于改选；例如C2_R2的正点估计不推翻事先规定的单候选规则。区间跨零表示精度不足以建立该项收益，不能等同于方法确实无效或与基线等效。

![全部方法的风险点估计与动作收益区间](d2_risk_and_action.png)

图1：左侧为已曝光120题的跨种子信用风险点估计；没有风险CI。右侧为相对保持的质量收益及探索性95%整题bootstrap区间。圆点为tune选定α，空心菱形为历史固定α；ZERO/B0只绘制一次。灰色背景行表示预定提名C1_R1，非显著性标记。图只用于阶段诊断，不是正式投稿图。[evidence:E01,E02,E03]

## 7. LOSO不是未来题在线收益

{chr(10).join(noise_table)}

{claims[3]['text']} [claim:C04] [evidence:E01,E02,E03]

LOSO先查看同一道题四个种子下全部候选动作的反馈，再选择动作并用第五种子评分；五个fold重叠，不是五次独立实验。它可提示同题重复执行中存在部分可复用信息，不能证明只依靠历史题反馈的路由器能够泛化到下一题。原方案要求本题计划在生成和评分前冻结，反馈更新未来决策；LOSO没有执行这一在线流程。获取同题菜单的重执行和评分成本也未纳入收益。五种子全零和均值零是含噪观测分层，不能当作真实信用为零。

## 8. 成本：已知、上界与未知

| 项目 | 记录 | 解释 |
|---|---:|---|
| D2主进程CPU | {e['cost']['process_cpu_seconds']:.6f} s | 含线程CPU，从进程起点计 |
| D2墙钟 | {e['cost']['wall_seconds']:.6f} s | 与CPU时间不同 |
| Git子进程CPU | 未测量；上界{e['cost']['git_cpu_upper_bound_seconds']:.6f} s | 由命令墙钟×逻辑核数保守计入 |
| 既往D2尝试CPU上界 | {e['cost']['previous_attempts_cpu_seconds_upper_bound']:.6f} s | 保留累计口径 |
| 独立复算主体CPU | {e['independent_replay']['review_cost']['main_process_cpu_seconds']:.6f} s | 不含进程初始化；与D2分别记录 |
| 新生成调用 / GPU时间 | 0 / 0 | 本轮复用历史输出 |
| 历史推理、评分与其他工程支出 | 本表未重汇总 | 不能把复用数据解释为零成本获取 |
| 人工时间、代理推理、能耗、货币费用 | 未计量 | 不提供虚构完整金额或回本点 |

报告生成开销另记录在[costs.csv](costs.csv)。以上是可追溯的本轮计算账目，不是整个项目的完整成本账本，也不构成任何在线全预算优势。

## 9. 置信等级与阶段决策

| 判断 | 置信等级 | 依据/范围 |
|---|---|---|
| 数值与提名规则实现一致 | 高 | 真实派生数组独立复算及来源哈希 |
| R0常数响应产生伪信用 | 高 | 固定目标控制与线性分解 |
| 固定α免罚截距降低本批风险 | 高 | 限本批已曝光数据的点估计比较 |
| 预测器具备未来图编辑能力 | 未建立 | 提名未通过，缺少新独立确认 |
| 图驱动复检优于随机/周期/最大不确定性 | 未知 | 本轮未执行在线机制比较 |
| 泛化至其他任务/异质模型/恢复场景 | 未知 | 本轮未评价 |

优势：有限且可分解的对照定位了原目标的一项偏置；全部负结果、支持数和无替补规则保留。风险：去偏置后信号仍弱；多次接触旧120题限制任何新肯定结论；差分模型未来接入G3必须重审观测算子、信息更新与付费反馈权限。

{claims[5]['text']} [claim:C06] [evidence:E01,E02,E03]

下一步只能在既定诊断次数与计算预算内，决定是否存在值得新独立确认的合法未来题决策信号；不能因LOSO的正区间直接宣称核心机制成功，也不能不断扩大模型或阈值搜索。任何第二轮诊断须保留本轮失败结论并单独编号。

## 10. 复现与审阅边界

运行`python render_d2_report.py --evidence d2_evidence.json --out .`可重建本报告、CSV和图。它只使用脱敏汇总包；不能用它替代派生数组的独立重拟合。完整源证据通过内容哈希关联，逐题材料及机器环境位置保留在项目私有记录中。

本包由AI协助生成，使用research-rigor、scientific-writing、peer-review的证据与表达检查，以及nature-figure的Python可视化约定。这里没有声称作者人工核验、期刊政策批准或编辑录用决定。对作者侧审查的具体事项见[INTERNAL_REVIEW.md](INTERNAL_REVIEW.md)；验证范围见[validation.json](validation.json)。
"""
    (out/"D2_REPORT.md").write_text(report,encoding="utf-8")
    internal="""# D2作者侧内部审查摘要

内部计算审查草稿；非期刊外审，不代表编辑决定。材料由项目所有者授权用于本项目，人工作者核阅与最终判断仍待完成。没有采用不适用的正式外审intake来推断期刊批准。

## 观察范围

审查对象为冻结D2协议、聚合结果、已完成的真实派生数组独立复算及本报告包。未审查新的语言模型推理、原题或gold，也未重拟合全部α路径。复算使用独立正规方程、标量动作循环、跨种子对乘积和计数权重bootstrap；机器一致性不是论文创新性或可发表性的证明。

## 主要判断

1. **阶段失败必须保留。** E01/E02的提名为C1_R1且过滤失败。任何候选替补、引用exposed最优点作为胜者，都会偏离E03。要求：维持无替补，后续另立协议。
2. **根因表述应限制在目标性质。** 常数响应控制支持R0全系数惩罚产生伪信用，固定α控制支持移除截距惩罚降低本批风险。旧代码实现了旧冻结合同。要求：不称旧结果因实现违规而无效，不把所有失败原因归结为同一个机制。
3. **区分重新选参。** C2_R1固定α=100与选定α=1000必须共同报告。要求：不能把选参后的全部变化都归因于免罚截距。
4. **LOSO不能替代在线实证。** 选择阶段已获得同题四种子完整菜单，成本未计，五fold相关。要求：保留正诊断，但不得把它写成未来题服务增益、主动采样收益或可达到的净收益下界。
5. **统计结论保持探索性。** 旧120题已曝光，区间未作多重校正。要求：不把区间跨零解释为等效，不将选择后的点估计称为独立确认。
6. **成本与数据范围完整披露。** 新GPU调用为零不代表历史数据免费；仅部分CPU开销可计量。要求：未来机制比较统一初始化、正常服务、反馈、恢复和隐藏诊断账本。

## 具体支持与不足

优势包括固定有限候选、同一底层唯一图证据、问题级统计、先提名后开发过滤、负结果和所有方法保留，以及对预测风险和行动质量的分开呈现。

不足包括缺少独立决策确认、校准不确定性、完整预算机制对照、跨任务与异质模型验证，以及暂停边恢复证据。当前这些是未完成证据，不能用报告格式或通过代码核验替代。

## 报告检查

全部12方法的选定/固定α、tune/exposed数据角色在method_metrics.csv中保留；缺失CI明确标为未计算。基线在固定α表重复仅用于排版。观测零与真实零区分；LOSO与五种子同菜单最大值区分；C1/C2两类颜色和选定/固定两类形状全图统一。

公开包只含汇总数据、程序及内容哈希；无机器路径、主机、凭据、问题标识或逐题值。自动扫描只验证指定模式，不能替代作者对外发布前的数据与权利复核。

## 建议的下一阶段门

若使用剩余有限诊断轮，应先固定假设、允许的反馈、训练/选择数据及停止条件；若仍无候选，应停止该预测器路线或降级相应主张。若找到候选，只能在新的未曝光数据上确认，再评价固定预测器下同完整预算的复检机制。此建议是科学工作顺序，不是ACL接受/拒绝意见。
"""
    (out/"INTERNAL_REVIEW.md").write_text(internal,encoding="utf-8")
    draw_figure(e,out)
    costs=[
        {"component":"D2_main_process","measure":"CPU_seconds","value":e['cost']['process_cpu_seconds'],"status":"measured","scope":"process_start_including_threads"},
        {"component":"D2_git_children","measure":"CPU_seconds","value":e['cost']['git_cpu_upper_bound_seconds'],"status":"upper_bound","scope":"wall_times_logical_cores; exact_CPU_unknown"},
        {"component":"D2_wall","measure":"wall_seconds","value":e['cost']['wall_seconds'],"status":"measured","scope":"D2_process"},
        {"component":"independent_replay","measure":"CPU_seconds","value":e['independent_replay']['review_cost']['main_process_cpu_seconds'],"status":"measured","scope":"replay_function_excludes_process_initialization"},
        {"component":"report_generation","measure":"CPU_seconds","value":time.process_time()-cpu0,"status":"measured","scope":"successful_render_main_excludes_imports_final_manifest_and_prior_QA_attempts"},
        {"component":"new_generation","measure":"GPU_hours","value":0,"status":"no_calls","scope":"D2_and_this_report"},
        {"component":"historical_inference","measure":"total_cost","value":None,"status":"not_reaggregated","scope":"existing_project_ledger_required"},
        {"component":"agent_human_energy_currency","measure":"total_cost","value":None,"status":"unmeasured","scope":"not_assumed_free"},
    ];write_csv(out/"costs.csv",costs)
    assert len(rows)==48 and len(selected)==len(fixed)==12
    for r in rows:
        assert r['additions']+r['deletions']+r['stays']==r['anchors']
        assert r['rank_defined_anchors']+r['rank_undefined_anchors']==r['anchors']
    for claim in claims:
        tagged=[line for line in report.splitlines() if '[claim:'+claim['id']+']' in line]
        assert len(tagged)==1
        clean=re.sub(r'\[(?:claim|evidence):[^]]+\]','',tagged[0])
        assert ' '.join(clean.split())==' '.join(claim['text'].split())
    svg=(out/"d2_risk_and_action.svg").read_text(encoding="utf-8")
    assert '<text' in svg
    prohibited=[r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]",r"\\\\[A-Za-z0-9_-]+\\",r"\b(?:\d{1,3}\.){3}\d{1,3}\b",r"hf_[A-Za-z0-9]{15,}",r"ssh\s+-p"]
    findings=[]
    for path in out.iterdir():
        if path.suffix not in {'.py','.json','.csv','.md','.svg'}:continue
        text=path.read_text(encoding='utf-8')
        for pattern in prohibited:
            if re.search(pattern,text):findings.append({'file':path.name,'pattern':pattern})
    assert not findings,findings
    validation={"schema":"mab2_d2_report_validation_v1","status":"PASS_MACHINE_CHECKS_HUMAN_REVIEW_PENDING",
        "method_count":12,"metric_rows":48,"selected_and_fixed_exposed_rows":24,"support_counts_consistent":True,"claim_text_bindings":6,
        "data_source_sha256":sha(args.evidence),"risk_intervals_invented":False,"tune_intervals_invented":False,
        "svg_editable_text":True,"privacy_pattern_findings":findings,"private_arrays_included":False,
        "backend":"Python/matplotlib","matplotlib_version":matplotlib.__version__,"numpy_version":np.__version__,
        "human_verified":False,"formal_peer_review":False,"submission_ready":False,
        "figure_visual_inspection":"pending_separate_review","generation_wall_seconds":time.time()-t0}
    write_json(out/"validation.json",validation)
    manifest={"schema":"mab2_d2_report_artifact_manifest_v1","scope":"aggregate_review_bundle_not_submission",
              "files":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}}
    write_json(out/"artifact_manifest.json",manifest)
    print(json.dumps({"status":validation['status'],"models":12,"metric_rows":48,"artifact_count":len(manifest['files']),"privacy_findings":len(findings),"figure":"d2_risk_and_action.png"}))


if __name__=="__main__":
    main()
