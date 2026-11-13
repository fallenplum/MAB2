"""Presentation-only view of the original nine-row aggregate CSV."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

METHODS=['candidate','uniform_pair_expectation','cyclic_pair','max_uncertainty',
         'no_graph_ablation','dirty_only','source_only','current_stay','full_revalidation_high_cost']
LABELS=['Full method (candidate)','Random (exact expectation)','Cyclic','Uncertainty',
        'Priority w/o graph','Dirty only','Reuse source','Stay (B)','All edges (higher cost)']


def sha(raw): return hashlib.sha256(raw).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)


def main():
    parser=argparse.ArgumentParser()
    for name in ('source-csv','result','complete','independent-review','original-manifest','original-plot','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();cpu=time.process_time()
    files={k:getattr(args,k) for k in ('source_csv','result','complete','independent_review','original_manifest','original_plot')}
    raw={k:p.read_bytes() for k,p in files.items()};pins={k:sha(v) for k,v in raw.items()}
    result=json.loads(raw['result']);complete=json.loads(raw['complete'])
    review=json.loads(raw['independent_review']);old_manifest=json.loads(raw['original_manifest'])
    require(complete['status']=='COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION','Incomplete analysis')
    require(complete['public_result_sha256']==pins['result']==review['official_result_sha256'],'Result/replay mismatch')
    require(review['status']=='PASS' and review['complete_receipt_sha256']==pins['complete'],'Replay completion binding mismatch')
    require(old_manifest['result_sha256']==pins['result'] and old_manifest['code_sha256']==pins['original_plot'],'Original figure provenance mismatch')
    original_csv=[x for x in old_manifest['files'] if x['name']=='r1_main_cost_quality.csv']
    require(len(original_csv)==1 and original_csv[0]['sha256']==pins['source_csv'],'Original CSV hash mismatch')
    rows=list(csv.DictReader(io.StringIO(raw['source_csv'].decode('utf-8-sig'))))
    require([r['method'] for r in rows]==METHODS,'Fixed nine-method order differs')
    method={r['method']:r for r in result['methods'] if r['condition']=='change' and r['k']==2}
    cost={r['method']:r for r in result['logical_policy_costs'] if r['condition']=='change' and r['k']==2}
    numeric_checks=0
    for row in rows:
        name=row['method'];c=cost[name]['all_eight_episode_actual_cost']
        expected={'accuracy':method[name]['accuracy'],'logical_input_plus_output_tokens':c[0]+c[1],
                  'logical_model_calls':c[2],'logical_score_count':c[3],'scorer_cpu_seconds':c[4]}
        require(set(row)=={'method',*expected},'CSV schema differs')
        for key,value in expected.items():
            require(float(row[key])==float(value),'Source CSV differs from result: '+name+'/'+key)
            numeric_checks+=1
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'r1_main_cost_quality.csv').write_bytes(raw['source_csv'])
    tokens=np.array([float(r['logical_input_plus_output_tokens']) for r in rows])
    accuracy=100*np.array([float(r['accuracy']) for r in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
    fig,(left,right)=plt.subplots(1,2,figsize=(9.5,5.6),sharey=True,
                                  gridspec_kw={'width_ratios':[1.45,1], 'wspace':.23})
    fig.subplots_adjust(left=.245,right=.97,top=.82,bottom=.20)
    y=np.arange(9);colors=['#C45D32','#397FA3']+['#A4ADB5']*7
    for i in y:
        if i%2==0:
            for ax in (left,right): ax.axhspan(i-.48,i+.48,color='#F3F5F7',zorder=0)
    left.barh(y,tokens/1e6,height=.48,color=colors,edgecolor='none',zorder=2)
    left.set_xlim(0,3.65);left.set_yticks(y,LABELS);left.set_ylim(8.55,-.55)
    left.set_xlabel('Complete logical tokens (millions)')
    left.set_title('a   Token charge across eight episodes',loc='left',pad=12)
    left.xaxis.set_major_locator(MultipleLocator(1));left.grid(axis='x',color='#D9DFE3',linewidth=.6,zorder=1)
    left.tick_params(axis='y',length=0,pad=9)
    value_artists=[]
    for i,value in enumerate(tokens):
        label=f'{value:,.0f}' if value.is_integer() else f'{value:,.1f}'
        value_artists.append(left.text(3.56,i,label,ha='right',va='center',fontsize=8,color='#28343F'))
    right.scatter(accuracy,y,s=39,c=colors,edgecolors='#FFFFFF',linewidths=.5,zorder=3)
    right.set_xlim(32,40);right.set_xticks([32,34,36,38,40]);right.set_xlabel('Future-service accuracy (%)')
    right.set_title('b   Accuracy on 192 service questions',loc='left',pad=12)
    right.grid(axis='x',color='#D9DFE3',linewidth=.6,zorder=1);right.tick_params(axis='y',left=False,labelleft=False)
    right.spines['left'].set_visible(False)
    for i,value in enumerate(accuracy):
        value_artists.append(right.text(value+.22,i,f'{value:.2f}%',ha='left',va='center',fontsize=8,color='#28343F'))
    fig.text(.245,.94,'R1 development: graph change, k = 2',ha='left',va='top',fontsize=13,weight='bold',color='#26313C')
    fig.text(.245,.892,'Same nine predefined methods; point estimates only',ha='left',va='top',fontsize=9,color='#53616C')
    fig.text(.245,.115,'Logical charge includes I initialization + permitted C revalidation + S deployment.\n'
             'Random is an exact policy expectation. Actual spending is not equalized.\n'
             '8 learning episodes; development evidence. Research GPU occupancy is separate.',
             ha='left',va='top',fontsize=8,color='#53616C',linespacing=1.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in (left,right):
        require(all(a.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)) for a in value_artists if a.axes is ax),'Value annotation outside panel')
    caption=('R1 development analysis, graph change and k=2. The two panels retain the same nine predefined methods and '
             'the original CSV values. Bars show complete logical input+output token charges across eight episodes: initial '
             'credit, permitted revalidation and final service. Reuse source has no C purchase; Stay (B) has neither I nor C '
             'charges, and all-edges buys the full menu. Mechanism methods share the frozen conservative cap and paired-feedback '
             'quota, not equal actual expenditure. Random cost and quality are the exact expectation over all 15 equally weighted '
             'purchased edge subsets, not summed evaluator enumeration cost. Points show accuracy over 192 service questions in '
             'eight learning episodes. No new interval, error bar, hypothesis test or gate is introduced. Priority w/o graph is '
             'the priority-score ablation, not a graph with no edges. Physical research GPU occupancy is accounted separately. '
             'Displayed labels are rounded; the copied source CSV is byte-identical to the original. This figure does not add '
             'a throughput, cost-optimality or independent-confirmation claim.')
    exports=[]
    for ext in ('svg','png','pdf'):
        path=args.output/('r1_cost_quality_rows_v2.'+ext)
        fig.savefig(path,dpi=300,facecolor='white')
        exports.append({'name':path.name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size})
    plt.close(fig)
    (args.output/'r1_cost_quality_rows_v2.caption.txt').write_text(caption+'\n',encoding='utf-8')
    require(all(sha(p.read_bytes())==pins[k] for k,p in files.items()),'Input changed during rendering')
    manifest={'status':'RENDERED_PENDING_VISUAL_REVIEW','presentation_only':True,'new_inference':False,
              'new_statistics_or_gates':False,'numeric_fields_exactly_checked':numeric_checks,
              'source_csv_byte_identical':sha((args.output/'r1_main_cost_quality.csv').read_bytes())==pins['source_csv'],
              'input_sha256':pins,'new_plot_source_sha256':sha(Path(__file__).read_bytes()),
              'original_plot_unchanged':True,'fixed_method_order':METHODS,'display_labels':LABELS,
              'exports':exports,'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,
              'CPU_seconds':time.process_time()-cpu,'new_model_calls':0,'new_GPU_hours':0}
    (args.output/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':manifest['status'],'numeric_checks':numeric_checks,'source_csv_byte_identical':manifest['source_csv_byte_identical']}))


if __name__=='__main__': main()
