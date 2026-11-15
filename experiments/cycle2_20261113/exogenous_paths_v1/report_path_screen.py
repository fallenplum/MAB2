"""Render audited aggregate evidence only; private answers never copied to report."""
import argparse,hashlib,json,math
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--audit',required=True);p.add_argument('--archive',required=True);p.add_argument('--cost',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    s=read(a.summary);audit=read(a.audit);cost=read(a.cost);root=Path(a.archive);out=Path(a.output)
    assert audit['status']=='PASS_INDEPENDENT_STDLIB_POINT_AUDIT' and audit['inputs_sha256']['summary']==sha(a.summary)
    assert cost['actual_cost_closed'] and cost['archive_status']=='COMPLETE_UNSCORED'
    matrices=[read(f) for f in sorted(root.glob('worker_*/unit_*/MATRIX.json'))];assert len(matrices)==96
    physical={};logical={};failures=unknown=terminals=0
    for m in matrices:
        assert m['status']=='COMPLETE_UNSCORED';terminals+=m['complete_terminal_conditions']
        for model,values in m['physical_cost_by_model'].items():
            failures+=values['native_failures'];unknown+=values['unknown_backend_requests']
            if model not in physical:physical[model]={k:0 for k in values['totals']}
            for k,v in values['totals'].items():assert v is not None;physical[model][k]+=v
        for model,values in m['logical_completed_node_cost_by_model'].items():
            if model not in logical:logical[model]={k:0 for k in values}
            for k,v in values.items():logical[model][k]=max(logical[model][k],v) if k=='peak_allocated_bytes' else logical[model][k]+v
    assert terminals==1728 and sum(v['calls'] for v in logical.values())==5760
    assert failures==unknown==0
    s['cost_accounting']={'formal_actual_resident_GPU_hours':cost['actual_GPU_hours'],'qualification_actual_GPU_hours':0.11696458899163796,
       'candidate_GPU_hours':cost['actual_GPU_hours']+0.11696458899163796,'physical_by_model':physical,'full_logical_by_model':logical,
       'formal_native_failures':failures,'unknown_backend_requests':unknown,'terminal_conditions':terminals,
       'failed_qualification_attempts':1,'prelaunch_placement_rejections':1,'monetary_cost':None,'human_label_cost':None,
       'note':'Shared exact-request caching lowers physical collection cost, not the charged full logical service cost. Wall/residence and kernel measurements are distinct.'}
    s['provenance']={'evaluation_summary_sha256':sha(a.summary),'independent_audit_sha256':sha(a.audit),'actual_cost_closure_sha256':sha(a.cost),'archive_manifest_sha256':sha(root/'ARCHIVE_MANIFEST.json'),'report_source_sha256':sha(__file__)}
    out.mkdir(parents=True,exist_ok=False)
    (out/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    lines=['# 第三轮：外生替代路径的信用变化机制筛选','',
      '**开发机制筛选，不是有效方法确认。** 固定 96 个新 HotpotQA train 题、两个生成种子、9 个条件，全部 1,728 个终态纳入。官方 validation/test 未打开。',
      '',f"预定主对比 D={s['primary_D']:.8f}；两个种子分别为 {s['D_by_seed'][0]:.8f}、{s['D_by_seed'][1]:.8f}；非零题均对比支持 {s['nonzero_question_support']}/96。",'',
      f"冻结筛选结论：`{s['screen_action']}`。判据为 D≥0.05、两种子 D 都为正、非零题均支持至少 20；不以 F1 或分组结果替代。它是经验开发过滤，不表示显著性或总体保证，也不自动批准新 GPU 实验。",'',
      '## 实际准确率','', '| 条件 | Seed 26111311 | Seed 26111312 | 两种子均值 |','|---|---:|---:|---:|']
    for name,v in s['accuracy_by_condition_seed'].items():lines.append(f'| {name} | {v[0]*100:.3f}% | {v[1]*100:.3f}% | {sum(v)*50:.3f}% |')
    lines+=['','## 配对点差、纠错与破坏','','| 比较 | 种子 | 点差 pp | 纠错 | 破坏 | 题数 |','|---|---:|---:|---:|---:|---:|']
    for name,rows in s['comparisons_by_seed'].items():
        for seed,row in zip([26111311,26111312],rows):lines.append(f"| {name} | {seed} | {row['paired_difference']*100:+.3f} | {row['corrections']} | {row['breakages']} | {row['paired_observations']} |")
    lines+=['','## 完整成本与失败','',
      f"实际正式 GPU 驻留 {cost['actual_GPU_hours']:.8f} 小时；含全部资格验证 {s['cost_accounting']['candidate_GPU_hours']:.8f} 小时。物理生成 {sum(v['calls'] for v in physical.values())} 次，完整逻辑调用 5,760 次。正式原生失败 {failures} 次、未知后台请求 {unknown} 次。",'',
      '先前资格验证的过长人工样例失败已原样保留并计入费用；另有一次显存放置检查在启动子进程前拒绝，GPU 费用为零。实际模型内核时间、完整逻辑 token/调用与进程驻留分别记录，不能互相替代。金钱费用和人工标签费用未测，记 null。', '',
      '## 证据边界','',
      '文档分片与单条中继路径删除是人为控制机制；没有声称真实部署必须分散信息。D 衡量两条边信用变化之差，不等于受影响边绝对信用一定增加。共同随机数用于图间配对，两种子不是两个独立数据集。', '',
      '当前 96 题在评分后属于已曝光开发数据，不得重新作为独立确认。没有运行在线信用维护策略、预算化主动采集或未来服务实验，因此不宣称它们优于随机复检、免费结构保护或中央条件路由。', '',
      '本轮不计算置信区间、不执行 bootstrap。独立核对是独立标准库算术实现，不是独立人类审稿或独立模型再生成。候选/实例结论不等于项目停止。', '',
      '![实际准确率与信用变化](screen_points.png)', '',
      '图为实际点估计；每个点对应一个生成种子，没有误差条。汇总 JSON 同时保留 F1、两条边的旧/新信用、分模型完整逻辑成本与物理成本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(s['accuracy_by_condition_seed']);fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for k,color in [(0,'#2864a0'),(1,'#be682e')]:
        axes[0].plot([s['accuracy_by_condition_seed'][n][k]*100 for n in names],list(range(len(names))),'o',color=color,label=f'Seed {26111311+k}',markersize=5)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[0].set_xlabel('Exact match (%)');axes[0].set_title('All frozen conditions');axes[0].legend(fontsize=8)
    axes[1].bar(['Overall','Seed 1','Seed 2'],[s['primary_D'],*s['D_by_seed']],color=['#555555','#2864a0','#be682e'])
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].axhline(.05,color='gray',linestyle='--',label='Development D threshold')
    axes[1].set_ylabel('Differential credit change (EM units)');axes[1].set_title(f"Nonzero question support: {s['nonzero_question_support']}/96");axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'screen_points.png',dpi=200);fig.savefig(out/'screen_points.svg');plt.close(fig)
    print(json.dumps({'status':'AGGREGATE_REPORT_RENDERED','output':str(out),'primary_D':s['primary_D'],'action':s['screen_action']}))
if __name__=='__main__':main()
