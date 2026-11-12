"""Predetermined R1 figures from verified aggregate results; never opens raw labels."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROTOCOL='5336eacc6713d4e444ddb631032409cd2718ad3b4aaee1387aff5baac40f5ecf'
METHODS=['candidate','uniform_pair_expectation','cyclic_pair','max_uncertainty','no_graph_ablation','dirty_only','source_only','current_stay','full_revalidation_high_cost']
NAMES=['Full method','Random (exact expectation)','Cyclic','Uncertainty','No graph','Dirty only','Reuse source','Stay','All edges (higher cost)']
COLORS=['#D55E00','#0072B2','#009E73','#CC79A7','#56B4E9','#E69F00','#666666','#999999','#000000']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(ok,message):
 if not ok: raise ValueError(message)

def finite_vector(value,shape,label,lower=None,upper=None):
 x=np.asarray(value,dtype=float)
 require(x.shape==shape and np.isfinite(x).all(),label+' shape/nonfinite values')
 if lower is not None: require(np.all(x>=lower),label+' below range')
 if upper is not None: require(np.all(x<=upper),label+' above range')
 return x

def validate_aggregates(d):
 require(d['protocol_sha256']==PROTOCOL and d['evidence_tier']=='development_pilot_not_independent_confirmation','Wrong evidence protocol/tier')
 expected={(c,k,m) for c in ('change','no_change') for k in (1,2) for m in METHODS}
 methods={};costs={}
 for field,target in (('methods',methods),('logical_policy_costs',costs)):
  require(len(d[field])==len(expected),'Incomplete '+field)
  for row in d[field]:
   key=(row['condition'],row['k'],row['method'])
   require(key in expected and key not in target,'Unknown/duplicate '+field+' key')
   target[key]=row
  require(set(target)==expected,'Missing '+field+' keys')
 for row in methods.values():
  values=finite_vector(row['episode_accuracy'],(8,),'Episode accuracy',0,1)
  accuracy=float(row['accuracy'])
  require(np.isfinite(accuracy) and 0<=accuracy<=1 and np.isclose(accuracy,values.mean(),rtol=0,atol=1e-12),'Accuracy/episode mean mismatch')
  require(row['service_questions']==192 and row['learning_episodes']==8,'Service/episode support mismatch')
 require(d['cost_vector_fields']==['input_tokens','output_tokens','calls','score_count','score_cpu_seconds'],'Unknown cost units/order')
 for row in costs.values():
  episodes=finite_vector(row['episode_actual_costs'],(8,5),'Episode cost',0)
  total=finite_vector(row['all_eight_episode_actual_cost'],(5,),'Total cost',0)
  mean=finite_vector(row['mean_episode_actual_cost'],(5,),'Mean cost',0)
  require(np.allclose(total,episodes.sum(axis=0),rtol=1e-12,atol=1e-9) and np.allclose(mean,episodes.mean(axis=0),rtol=1e-12,atol=1e-9),'Cost episode aggregation mismatch')
 expected_contrasts={}
 for condition in ('change','no_change'):
  for k in (1,2):
   candidate=np.asarray(methods[(condition,k,'candidate')]['episode_accuracy'])
   for method in METHODS[1:]:
    expected_contrasts[f'{condition}/k{k}/candidate-minus-{method}']=candidate-np.asarray(methods[(condition,k,method)]['episode_accuracy'])
 for k in (1,2):
  expected_contrasts[f'change-minus-no_change/k{k}/candidate-minus-uniform']=expected_contrasts[f'change/k{k}/candidate-minus-uniform_pair_expectation']-expected_contrasts[f'no_change/k{k}/candidate-minus-uniform_pair_expectation']
 require(set(d['contrasts'])==set(expected_contrasts),'Missing/extra predefined contrasts')
 for name,expected_values in expected_contrasts.items():
  row=d['contrasts'][name];values=finite_vector(row['episode_values'],(8,),'Contrast episode values')
  require(np.allclose(values,expected_values,rtol=0,atol=1e-12) and np.isclose(row['mean'],values.mean(),rtol=0,atol=1e-12),'Contrast/accuracy arithmetic mismatch')
  interval=finite_vector(row['ci95'],(2,),'Contrast interval')
  require(interval[0]<=interval[1],'Reversed contrast interval')
  require(row['unit']=='episode' and row['independent_episode_count']==8 and row['resamples']==10000 and row['seed']==69173,'Bootstrap metadata differs from protocol')
 return methods,costs

def load_verified_inputs(analysis_dir,independent_review):
 paths={'result':analysis_dir/'public/R1_RESULTS.json','complete':analysis_dir/'private/COMPLETE.json','review':Path(independent_review)}
 snapshots={k:path.read_bytes() for k,path in paths.items()}
 pins={k:hashlib.sha256(raw).hexdigest() for k,raw in snapshots.items()}
 d=json.loads(snapshots['result']);done=json.loads(snapshots['complete']);review=json.loads(snapshots['review'])
 require(done['status']=='COMPLETE_DEVELOPMENT_ANALYSIS_NOT_CONFIRMATION' and done['public_result_sha256']==pins['result'],'Analysis completion/result binding failed')
 require(review.get('status')=='PASS','Independent replay must pass before plotting')
 require(review['official_result_sha256']==pins['result'] and review['complete_receipt_sha256']==pins['complete'] and review['replay_inputs_sha256']==done['replay_inputs_sha256'],'Independent audit belongs to different data')
 require(d['provenance']['replay_inputs_sha256']==done['replay_inputs_sha256'],'Result/replay input binding failed')
 validate_aggregates(d)
 return d,paths,pins

def verify_snapshots_unchanged(paths,pins):
 require(all(sha(path)==pins[key] for key,path in paths.items()),'Verified aggregate inputs changed during plotting')

def main():
 started=time.monotonic();started_cpu=time.process_time()
 p=argparse.ArgumentParser();p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--independent-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d,input_paths,input_pins=load_verified_inputs(a.analysis_dir,a.independent_review);code_pin=sha(__file__)
 a.output.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 files=[]
 def save(fig,name,caption):
  for ext in ['pdf','svg','png']:
   q=a.output/(name+'.'+ext);fig.savefig(q,dpi=300,bbox_inches='tight');files.append({'name':q.name,'sha256':sha(q),'bytes':q.stat().st_size})
  q=a.output/(name+'.caption.txt');q.write_text(caption+'\n',encoding='utf-8');files.append({'name':q.name,'sha256':sha(q),'bytes':q.stat().st_size});plt.close(fig)
 # All predeclared comparisons are shown; no selection by significance.
 fig,ax=plt.subplots(figsize=(6.6,4.3));comparators=METHODS[1:]
 for y,m in enumerate(comparators):
  x=d['contrasts']['change/k2/candidate-minus-'+m];mean=100*x['mean'];lo,hi=np.asarray(x['ci95'])*100
  ax.plot([lo,hi],[y,y],color=COLORS[0] if m=='uniform_pair_expectation' else '#777777',lw=1.6)
  ax.plot(mean,y,'o',color=COLORS[0] if m=='uniform_pair_expectation' else '#333333',ms=5)
 ax.set_yticks(range(len(comparators)),NAMES[1:]);ax.invert_yaxis();ax.axvline(0,color='#555555',lw=.8,ls='--');ax.set_xlabel('Full method minus comparator (percentage points)');ax.set_title('Graph change, k = 2: exploratory service-quality contrasts');ax.grid(axis='x',alpha=.2)
 save(fig,'r1_contrasts','Eight disjoint learning episodes and 192 service questions. Bars: frozen 10,000-resample episode percentile intervals. Random is the single primary comparator; other comparisons are descriptive. Reuse/stay cost less and all-edges costs more. These are development results, not independent confirmation.')
 methodrows={r['method']:r for r in d['methods'] if r['condition']=='change' and r['k']==2};costrows={r['method']:r for r in d['logical_policy_costs'] if r['condition']=='change' and r['k']==2}
 fig,ax=plt.subplots(figsize=(6.6,4.2))
 table=[]
 for i,m in enumerate(METHODS):
  costs=costrows[m]['all_eight_episode_actual_cost'];tokens=costs[0]+costs[1];accuracy=100*methodrows[m]['accuracy'];marker='D' if m=='full_revalidation_high_cost' else ('s' if m in ['source_only','current_stay'] else 'o')
  ax.scatter(tokens/1e6,accuracy,s=42,c=COLORS[i],marker=marker,label=NAMES[i],edgecolors='white',linewidths=.4,zorder=3)
  table.append({'method':m,'accuracy':methodrows[m]['accuracy'],'logical_input_plus_output_tokens':tokens,'logical_model_calls':costs[2],'logical_score_count':costs[3],'scorer_cpu_seconds':costs[4]})
 ax.set_xlabel('Logical input + output tokens across eight episodes (millions)');ax.set_ylabel('Service accuracy (%)');ax.set_title('Graph change, k = 2: logical token cost and service quality');ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=7,frameon=False)
 save(fig,'r1_cost_quality','Logical cost includes initial credit, permitted revalidation, and final service. Random costs and quality are exact policy expectations, not the sum of evaluator replicas. Actual expenditure is not forced equal; the matched mechanism methods share a conservative cap and feedback quota. Research precomputation GPU occupancy is reported separately. No throughput or cost-optimality claim follows from this scatter.')
 fig,ax=plt.subplots(figsize=(6.6,3.6))
 for condition,color,label,offset in [('change',COLORS[0],'Graph changed',-.06),('no_change',COLORS[1],'Graph unchanged',.06)]:
  rows={r['method']:r for r in d['methods'] if r['condition']==condition and r['k']==2};delta=100*(np.array(rows['candidate']['episode_accuracy'])-np.array(rows['uniform_pair_expectation']['episode_accuracy']))
  ax.plot(np.arange(1,9)+offset,delta,'o',linestyle='None',color=color,label=label,ms=5)
 ax.axhline(0,color='#555555',lw=.8,ls='--');ax.set_xticks(range(1,9));ax.set_xlabel('Predefined learning episode');ax.set_ylabel('Full method minus random (pp)');ax.set_title('Episode-level paired effects, k = 2');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
 save(fig,'r1_episode_effects','Each point summarizes 24 disjoint service questions for a fixed episode. The two conditions share the episode source bank and service-question identities; they are paired, not independent replications. Episodes are predefined categorical conditions, not an online learning timeline; no lines connect different episodes.')
 with (a.output/'r1_main_cost_quality.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 q=a.output/'r1_main_cost_quality.csv';files.append({'name':q.name,'sha256':sha(q),'bytes':q.stat().st_size})
 verify_snapshots_unchanged(input_paths,input_pins);require(sha(__file__)==code_pin,'Plotting code changed during execution')
 receipt={'status':'FIGURES_GENERATED_VISUAL_REVIEW_REQUIRED','result_sha256':input_pins['result'],'complete_receipt_sha256':input_pins['complete'],'independent_review_sha256':input_pins['review'],'code_sha256':code_pin,'source_protocol_sha256':PROTOCOL,'input_scope':'same verified byte snapshots; paths rechecked before manifest','files':files,'episode_graph_has_time_connecting_lines':False,'publication_claim':'development_pilot_not_confirmation','cpu_seconds':time.process_time()-started_cpu,'wall_seconds':time.monotonic()-started,'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__}
 (a.output/'FIGURE_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':receipt['status'],'figures':3,'exports':9,'bound_artifacts':len(files)}))
if __name__=='__main__':main()
