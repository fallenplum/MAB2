"""Render frozen descriptive factorial summaries; never reads question-level material."""
from pathlib import Path
import argparse,json,hashlib
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=json.loads(a.summary.read_bytes());out=a.output;out.mkdir(parents=True,exist_ok=False)
    def avg(n):return sum(r['accuracy_by_condition_seed'][n])/2
    lines=['# 第四轮：接收模型与原始证据访问的固定对照','','96 个未暴露开发问题 × 2 个生成种子 × 29 配置；答案仅在完整收集、实际成本闭合与逐请求 CPU 回放通过后开放。官方验证/测试集未读取。以下为冻结的经验筛选，不是显著性或总体非劣保证。','','## 固定提名及判据','',f"固定提名：`{r['fixed_nominee']}`。结果：**{r['screen_action']}**。其他单元不能替补。",'','| 判据 | 结果 |','|---|---|']
    for k,v in r['gates'].items():lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += ['','## 全部准确率与完整逻辑成本','','| 配置 | 种子一 EM | 种子二 EM | 均值 | 平均 tokens/题 | 平均 GPU kernel 秒/题 |','|---|---:|---:|---:|---:|---:|']
    for n,v in r['accuracy_by_condition_seed'].items():lines.append(f"| {n} | {100*v[0]:.3f}% | {100*v[1]:.3f}% | {100*avg(n):.3f}% | {sum(r['tokens_by_condition_seed'][n])/192:.2f} | {sum(r['kernel_seconds_by_condition_seed'][n])/192:.5f} |")
    lines += ['','成本包含该配置的全部模型节点；物理测量缓存重用不减免逻辑服务费用。GPU 驻留小时另记，包含加载和等待。两个生成种子不是两个任务。','','## 条件信用变化','','| 接收单元 | D | 种子一 D | 种子二 D | 非零题均值支持数 |','|---|---:|---:|---:|---:|']
    for n,v in r['credit_signals_by_cell'].items():lines.append(f"| {n} | {v['D']:.6f} | {v['D_by_seed'][0]:.6f} | {v['D_by_seed'][1]:.6f} | {v['nonzero_question_support']}/96 |")
    lines += ['','D 为受影响边与另一条边的信用变化之差；图基准项代数抵消，不能解释成直接服务质量提升。完整四单元效应、交互、纠错/破坏与 F1 保留在 SUMMARY.json。','','## 解释边界','','优势：固定上游请求，分别控制接收模型与原始证据访问；包含单模型、完整上下文聚合和廉价三样本投票；正式结果不能触发改选胜者。风险：仅一个任务、两个生成种子、有限受控图，且暴露后只能作为开发数据。没有验证在线信用维护、独立 I/C/S 或完整反馈预算收益。无置信区间、无 bootstrap。工程和算术审计通过不能替代独立人工审稿或科学复现。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(out/'SUMMARY.json').write_bytes(a.summary.read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    cells=['M_PARTIAL','Q_PARTIAL','M_FULL','Q_FULL'];names=['G1','G1__Q_PARTIAL','G1__M_FULL','G1__Q_FULL']
    colors=['#777777','#5279af','#b88144','#368672']
    for i,(c,n,color) in enumerate(zip(cells,names,colors)):
        axes[0].bar(i,100*avg(n),color=color,alpha=.75);axes[0].scatter([i-.09,i+.09],[100*v for v in r['accuracy_by_condition_seed'][n]],s=14,color='black',zorder=3)
        s=r['credit_signals_by_cell'][c];axes[1].bar(i,s['D'],color=color,alpha=.75);axes[1].scatter([i-.09,i+.09],s['D_by_seed'],s=14,color='black',zorder=3)
    for ax in axes[:2]:ax.set_xticks(range(4),['M / notes','Q / notes','M / raw','Q / raw'],rotation=25,ha='right');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exact match (%)');axes[0].set_ylim(0,100);axes[0].set_title('A. Post-shock receiver factorial')
    axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel('Differential credit shift D');axes[1].set_title('B. Credit shift (seed points)')
    for n,color in [('G1__Q_FULL','#368672'),('SINGLE_Q','#5279af'),('VOTE_Q3','#b88144'),('INDEPENDENT3_Q','#a65376'),('INDEPENDENT3_M','#777777')]:
        cost=sum(r['tokens_by_condition_seed'][n])/192;acc=100*avg(n);axes[2].scatter(cost,acc,color=color,s=35,label=n)
    axes[2].set_xlabel('Full logical tokens / service question');axes[2].set_ylabel('Exact match (%)');axes[2].set_title('C. Quality and complete service cost');axes[2].legend(fontsize=6.5,loc='best');axes[2].spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:fig.savefig(out/('factorial_points.'+ext),dpi=180)
    plt.close(fig)
    (out/'FIGURE_PROVENANCE.json').write_text(json.dumps({'summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed_markers_are_not_intervals':True,'aggregation':'96 questions x 2 seeds; quality mean and complete logical service cost','image_visual_review':'PENDING'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'REPORT_RENDERED_PENDING_VISUAL_REVIEW','screen_action':r['screen_action']}))
if __name__=='__main__':main()
