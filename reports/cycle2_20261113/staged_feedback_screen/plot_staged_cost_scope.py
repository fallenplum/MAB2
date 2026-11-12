"""Post-hoc descriptive figure of a frozen historical oracle screen."""
from pathlib import Path
import argparse,csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists() or not any(a.output.iterdir());a.output.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(a.csv.open(encoding='utf-8-sig',newline='')))
 selected=[r for r in rows if r['scenario']=='observed_token_score_equivalent_ratio_0' and r['method']=='staged_Poisson_HT_oracle' and float(r['fraction'])<1]
 fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
 for ax,mode,title in zip(axes,['suffix_only','lazy_baseline'],['A  Suffix costs only (baseline excluded)','B  Baseline construction charged on demand']):
  for split,color,marker in [('train','#0072B2','o'),('tune','#D55E00','s')]:
   sub=sorted([r for r in selected if r['split']==split and r['base_mode']==mode],key=lambda r:float(r['fraction']))
   assert len(sub)==12 and all(r['feasible']=='True' for r in sub)
   if mode=='lazy_baseline':
    assert all(abs(float(r['p1'])-1)<1e-10 and abs(float(r['p2'])-1)<1e-10 for r in sub)
   ax.plot([float(r['fraction']) for r in sub],[float(r['variance_ratio_to_direct_Poisson']) for r in sub],label=split,marker=marker,color=color,markersize=4,linewidth=1.5)
  ax.axhline(1,color='#555555',linestyle='--',linewidth=1,zorder=0)
  ax.set_title(title,loc='left',fontsize=10,pad=11);ax.set_xlabel('Fraction of the corresponding census cost')
  ax.set_xlim(0,.94);ax.set_ylim(.68,1.045);ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Oracle variance / direct Poisson variance')
 axes[0].legend(frameon=False,loc='lower left',fontsize=9)
 axes[1].text(.06,.79,'Completion probability = 1 in both strata\nThe staged policy reduces to direct sampling',fontsize=9)
 fig.subplots_adjust(left=.08,right=.985,bottom=.26,top=.85,wspace=.11)
 fig.suptitle('Paid response screening: apparent headroom depends on cost scope',fontsize=12,y=.98)
 fig.text(.08,.08,'Historical train/tune finite-bank design variance; shared exact-wire cache; oracle allocation.\nInput + output tokens are a logical cost proxy; scoring equivalent set to zero, not measured free.\nNo new-question guarantee. High-scoring-cost sensitivity and small counterexamples remain in the report.',fontsize=8,linespacing=1.5)
 for suffix in ['png','pdf','svg']:fig.savefig(a.output/('cost_scope.'+suffix),dpi=180)
 plt.close(fig)
 receipt={'status':'DESCRIPTIVE_FIGURE_RENDERED_REQUIRES_VISUAL_REVIEW','source_csv_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'panels':2,'points':48,'filter':{'scenario':'observed_token_score_equivalent_ratio_0','method':'staged_Poisson_HT_oracle','fractions':'all <1 including threshold breakpoints; census excluded because both variances zero'},'reference':'same direct Poisson with exact-wire cache and same cost scope','inferential_CI':False,'raw_question_data':False}
 (a.output/'FIGURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':receipt['status'],'points':receipt['points']}))
if __name__=='__main__':main()
