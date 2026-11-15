"""Aggregate-only reporting after frozen scoring and independent reconstruction.

No fitting, resampling, confidence intervals, or candidate selection. Scientific
decisions are copied from the frozen evaluator, never recalibrated here.
"""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def render(summary,private,independent,cost,out,synthetic=False):
    assert independent['status']=='PASS_INDEPENDENT_COST_POLICY_AND_POINT_RECONSTRUCTION'
    assert independent['gates']==summary['gates'] and summary['bootstrap_runs']==0 and not summary['confidence_intervals_computed']
    assert cost['actual_cost_closed'] and len(private['rows'])==1360
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'SUMMARY.json',summary);dump(out/'INDEPENDENT_AUDIT.json',independent);dump(out/'COST_CLOSURE.json',cost)
    flat=[]
    for r in private['rows']:
        item={k:r[k] for k in ['episode','seed_index','generation_seed','condition','method','selected_condition','correct','service_questions','accuracy','F1','utility','pairs','unused_token_budget','corrections_vs_keep','breakages_vs_keep','corrections_vs_stale','breakages_vs_stale','restoration_to_keep','initial_score_observations','feedback_score_observations','initial_gold_questions','feedback_gold_questions']}
        for stage in ['initial','feedback','service','complete']:
            item.update({stage+'_'+k:v for k,v in r[stage+'_cost'].items()})
        flat.append(item)
    write_csv(out/'EPISODE_SEED_AGGREGATES.csv',flat)
    points=[]
    for cond,methods in summary['results'].items():
        for method,r in methods.items():
            peers=[x for x in private['rows'] if x['condition']==cond and x['method']==method]
            points.append({'condition':cond,'method':method,'service_question_seed_observations':2560,'unique_service_questions':512,
                'accuracy':r['accuracy'],'F1':r['F1'],'utility':r['utility'],'tokens_per_service_question_including_I_C':r['total_tokens']/2560,
                'mean_feedback_pairs':r['pairs'],'mean_initial_gold_questions':sum(x['initial_gold_questions'] for x in peers)/40,
                'mean_feedback_gold_questions':sum(x['feedback_gold_questions'] for x in peers)/40,
                'corrections_vs_keep':r['corrections_vs_keep'],'breakages_vs_keep':r['breakages_vs_keep'],
                'corrections_vs_stale':sum(x['corrections_vs_stale'] for x in peers),'breakages_vs_stale':sum(x['breakages_vs_stale'] for x in peers),
                'terminal_keep_after_stale_delete_episode_seed_count':sum(x['restoration_to_keep'] for x in peers),
                'mean_unused_token_cap':r['unused_token_budget']})
    write_csv(out/'METHOD_POINTS.csv',points)
    # Descriptive prefixes only: policy was frozen for H=64. Do not reselect
    # feedback, weight, method, or horizon from these plotted service prefixes.
    curves=[]
    for cond,methods in summary['results'].items():
        for method in methods:
            peers=[x for x in private['rows'] if x['condition']==cond and x['method']==method]
            cumulative_tokens=[x['initial_cost']['tokens']+x['feedback_cost']['tokens'] for x in peers]
            cumulative_correct=[0]*40
            for n in range(1,65):
                for j,x in enumerate(peers):
                    i=x['episode']*80+15+n;s=x['seed_index'];g=x['selected_condition']
                    cumulative_correct[j]+=private['em'][i][s][g]
                    cumulative_tokens[j]+=sum(v['generation']['input_tokens']+v['generation']['output_tokens'] for v in private['records'][i][s][g]['nodes'].values())
                curves.append({'condition':cond,'method':method,'service_prefix_questions_per_episode':n,
                    'mean_cumulative_tokens_per_episode':sum(cumulative_tokens)/40,
                    'mean_cumulative_correct_per_episode':sum(cumulative_correct)/40,
                    'mean_cumulative_net_correct_equivalent':sum(cumulative_correct[j]-summary['utility_token_weight']*cumulative_tokens[j] for j in range(40))/40,
                    'descriptive_prefix_of_fixed_H64_policy':True})
            r=summary['results'][cond][method]
            assert abs(curves[-1]['mean_cumulative_correct_per_episode']/64-r['accuracy'])<1e-12
            assert abs(curves[-1]['mean_cumulative_tokens_per_episode']*40-r['total_tokens'])<1e-6
            assert abs(curves[-1]['mean_cumulative_net_correct_equivalent']/64-r['utility'])<1e-12
    write_csv(out/'CUMULATIVE_SERVICE_POINTS.csv',curves)
    lines=['# 第五轮：完整成本下的图条件信用复检','',
        ('**人工数据报告器测试，禁止作为科研结果。**' if synthetic else '**冻结开发实验结果；不是最终测试或独立确认。**'),'',
        '640 个新 HotpotQA train 问题，八个互斥 I8/C8/S64 episode，五个生成种子。每题九个条件，共 28,800 个终态。512 个不同服务问题；每种方法、每种图变化条件的 2,560 个问题—种子观测不是 2,560 个独立问题。', '',
        '固定提名 `closure_EVSI`；冻结筛选结论：`'+summary['screen_action']+'`。对照中表现更好的方法不能替补提名。', '',
        '| 冻结判据 | 结果 |','|---|---|']
    lines += ['| '+k+' | '+('PASS' if v else 'FAIL')+' |' for k,v in summary['gates'].items()]
    for cond in ['change','no_change']:
        lines+=['','## '+('图改变' if cond=='change' else '无图改变对照'),'',
            '| 方法 | EM % | F1 % | 完整 utility | 完整 tokens/服务题 | 平均复检对数 | 初始/复检/服务 tokens 每 episode |',
            '|---|---:|---:|---:|---:|---:|---|']
        for m,r in summary['results'][cond].items():
            lines.append(f"| {m} | {100*r['accuracy']:.3f} | {100*r['F1']:.3f} | {r['utility']:.6f} | {r['total_tokens']/2560:.2f} | {r['pairs']:.3f} | {r['mean_initial_tokens']:.1f} / {r['mean_feedback_tokens']:.1f} / {r['mean_service_tokens']:.1f} |")
        lines+=['','| 对照 | 提名减对照 EM (pp) | utility 点差 | 完整 token 比 | 五种子 utility 点差 |','|---|---:|---:|---:|---|']
        for m,c in summary['comparisons'][cond].items():
            lines.append(f"| {m} | {100*c['accuracy_difference']:.3f} | {c['utility_difference']:.6f} | {c['token_ratio']:.4f} | "+', '.join(f'{v:.6f}' for v in c['utility_difference_by_seed'])+' |')
    lines+=['','## 成本、反馈与公平性','',
        f"正式采集实际 GPU 驻留 {cost['actual_GPU_hours']:.8f} 小时；资格验证另计 0.0661232537121719 小时，不能重复加账。驻留包含模型加载和等待，不等同 GPU kernel 时间。",
        '主效用为服务 EM − (0.005/1000) × (初始化+复检+服务 tokens)/64。该权重是预先声明的研究折衷，不是货币价格。各方法共享每 episode 3,000,000 token 上限，但实际花费不同；不把同一上限写成相同实际花费。剩余额度、分模型 kernel、调用、token 与 Gold 观测数保留在 CSV/JSON。',
        '完整测量网格使用物理缓存以减少实验开销；策略只在购买后读取相应 C 反馈，只减免已经付费的精确请求。服务成本重新全额计入选定图的全部活跃节点。所有方法共享精确死节点省略，不能将其收益归为复检策略优势。',
        '静态免初始化对照不收取初始化反馈费用；维护方法包括 keep 均收取初始化。反馈来自 benchmark gold 理想 oracle；人工标注、金融价格、能耗、代理推理等未测量，不能声称现实总经济成本优势。',
        '', '## 决策错误与恢复的解释', '',
        '`METHOD_POINTS.csv` 提供相对 keep/stale 的纠错、破坏，以及旧信用建议删除但最终选择 keep 的 episode—seed 数。后者只是终态决策反转代理，不是已经部署删除后成功恢复边的在线轨迹。报告没有证明恢复机制。',
        '删除造成错误的计数使用相同问题与种子的 keep 结果作反事实对照；它不等于已知真实期望信用符号。跨种子共享题目，episode 是一次构造的有限随机划分，不能用种子数量冒充跨任务重复。',
        '', '## 证据与局限', '',
        '优势：同预测器、同 I/C/S、同模型、固定方法与门槛、完整实际逻辑成本，以及无标签收集后 CPU 请求回放和第二实现算术审计。风险：一个任务、两个模型、两条可编辑边、一次受控图变化、小样本初始化、简单可分离先验、benchmark 标签 oracle，以及缓存回放而非实时部署。',
        '主判据是开发筛选，不是显著性检验、非劣保证或外部泛化结论。没有计算置信区间或执行 bootstrap。审计为研究代理的第二实现，不是人类独立审稿或真实新种子模型再生成。',
        '所有图的额外分组、固定 H=64 策略的服务前缀曲线和四个 token 权重敏感性均为描述结果，不用于改变主权重、服务期限或重选候选。前缀曲线不能解释为每个较短期限下重新最优化的策略。官方 validation/test 保持未打开；本轮数据评分后属于已曝光开发数据。',
        '', '![固定方法和基线的点估计](service_points.png)', '',
        '每个小点是一个生成种子；没有误差条。完整成本包含 I/C/S。图表和报告从冻结汇总与已审计私有重放生成，不含题目、答案、逐题矩阵或凭据。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    methods=list(summary['results']['change']);y=np.arange(len(methods));fig,axes=plt.subplots(1,3,figsize=(13,8),sharey=True,layout='constrained')
    for i,m in enumerate(methods):
        r=summary['results']['change'][m];color='#0072B2' if m=='closure_EVSI' else '#777777'
        axes[0].scatter([r['by_generation_seed'][str(s)]['accuracy']*100 for s in range(5)],[i]*5,s=12,color=color,alpha=.55)
        axes[0].scatter(r['accuracy']*100,i,s=35,marker='D',color=color)
        delta=summary['comparisons']['change'][m]['utility_difference'];axes[1].scatter(delta,i,s=28,color=color)
        left=0
        for stage,col in [('initial','#56B4E9'),('feedback','#E69F00'),('service','#009E73')]:
            v=r['stage_cost_totals'][stage]['tokens']/2560;axes[2].barh(i,v,left=left,color=col,height=.65,label=stage if i==0 else None);left+=v
    axes[0].set_yticks(y,methods);axes[0].invert_yaxis();axes[0].set_xlabel('Service EM (%)');axes[0].set_title('A  Means and five generation seeds')
    axes[1].axvline(0,color='black',lw=.7);axes[1].set_xlabel('Nominee utility minus comparator');axes[1].set_title('B  Fixed token-price utility difference')
    axes[2].set_xlabel('Complete tokens / service question');axes[2].set_title('C  Initialization + feedback + service',pad=27);axes[2].legend(loc='lower center',bbox_to_anchor=(.5,1),ncol=3,frameon=False)
    if synthetic:fig.suptitle('ARTIFICIAL TEST DATA — NOT EXPERIMENT RESULTS',fontsize=12,color='#9A3412')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    for ext in ['png','pdf','svg']:fig.savefig(out/('service_points.'+ext),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
    for m in ['closure_EVSI','EVSI_fixed_price','random','stale','DELETE_CHANGED_NO_INIT','SINGLE_Q','INDEPENDENT3_Q']:
        rows=[r for r in curves if r['condition']=='change' and r['method']==m]
        ax.plot([r['mean_cumulative_tokens_per_episode']/1000 for r in rows],[r['mean_cumulative_correct_per_episode'] for r in rows],label=m,lw=1.3)
    ax.set_xlabel('Mean cumulative tokens per episode (thousands; includes I/C)');ax.set_ylabel('Mean cumulative correct service answers')
    ax.set_title('Descriptive prefixes of fixed H=64 policies');ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
    if synthetic:fig.suptitle('ARTIFICIAL TEST DATA — NOT EXPERIMENT RESULTS',fontsize=10,color='#9A3412')
    for ext in ['png','pdf','svg']:fig.savefig(out/('cumulative_service.'+ext),dpi=180)
    plt.close(fig)
    dump(out/'REPORTING_RECEIPT.json',{'status':'SYNTHETIC_REPORT_TEST' if synthetic else 'AGGREGATE_REPORT_GENERATED','frozen_screen_action':summary['screen_action'],'unique_service_questions':512,'question_seed_observations_per_method_condition':2560,'episode_seed_rows':len(flat),'new_model_calls':0,'bootstrap_runs':0,'private_question_records_exported':False,'source_sha256':sha(__file__)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--private',required=True);p.add_argument('--audit',required=True);p.add_argument('--cost',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=load(a.audit);assert audit['input_sha256']['summary']==sha(a.summary) and audit['input_sha256']['private']==sha(a.private)
    render(load(a.summary),load(a.private),audit,load(a.cost),a.output)
if __name__=='__main__':main()
