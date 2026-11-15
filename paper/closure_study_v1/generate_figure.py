"""Paper-size descriptive figure from frozen aggregate evidence."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);r=p.parse_args().repo
s=json.loads((r/'reports/cycle2_20261113/closure_formal5_results_v1/SUMMARY.json').read_text(encoding='utf-8'))
keys=['closure_EVSI','EVSI_fixed_price','EVSI_no_cost_ranking','EVSI_old_graph_cost','random','periodic','uncertainty','graph_changed_first','stale','protect_changed','keep','KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT','SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q']
labels=['Closure EVSI','Fixed price','No cost ranking','Old graph cost','Random','Periodic','Uncertainty','Changed first','Stale','Protect changed','Keep + init.','Keep, no init.','Delete changed, no init.','Delete other, no init.','Single Qwen','Vote Qwen (3)','Independent 3 + Qwen']
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(6.8,4.9),sharey=True,gridspec_kw={'width_ratios':[1,1.05]})
fig.subplots_adjust(left=.29,right=.99,bottom=.13,top=.91,wspace=.17)
for i,key in enumerate(keys):
 v=s['results']['change'][key];color='#0072B2' if key=='closure_EVSI' else '#777777'
 axs[0].scatter([100*x['accuracy'] for x in v['by_generation_seed'].values()],[i]*5,s=10,color=color,alpha=.4,zorder=2)
 axs[0].scatter([100*v['accuracy']],[i],s=24,marker='D',color=color,zorder=3)
 axs[1].scatter([s['comparisons']['change'][key]['utility_difference']],[i],s=21,color=color,zorder=3)
axs[0].set_yticks(range(len(keys)),labels);axs[0].invert_yaxis()
axs[0].set_xlabel('Service exact match (%)');axs[1].set_xlabel('Nominee utility minus comparator')
axs[0].set_title('A  Mean and five seeds',fontsize=9);axs[1].set_title('B  Full-cost utility difference',fontsize=9)
axs[1].axvline(0,color='black',lw=.7);axs[1].axvline(.005,color='#D55E00',ls='--',lw=.8)
axs[1].set_xticks([-.015,-.005,.005]);axs[0].set_xticks([57,59,61,63])
for ax in axs: ax.grid(axis='x',alpha=.15);ax.tick_params(labelsize=8)
for ext in ['pdf','png','svg']: fig.savefig(r/f'paper/closure_study_v1/figures/paper_points.{ext}',dpi=240)
plt.close(fig)
