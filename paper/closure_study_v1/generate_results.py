"""Generate paper tables from frozen public aggregates; no fitting or model calls."""
import argparse,csv,json,hashlib
from pathlib import Path

def main():
 a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,required=True);p=a.parse_args()
 r=p.repo; base=r/'reports/cycle2_20261113/closure_formal5_results_v1'
 paper=r/'paper/closure_study_v1'; out=paper/'generated';out.mkdir(exist_ok=True)
 s=json.loads((base/'SUMMARY.json').read_text(encoding='utf-8'))
 assert s['screen_action']=='STOP_FORMAL5_CANDIDATE'
 order=['closure_EVSI','EVSI_fixed_price','EVSI_no_cost_ranking','EVSI_old_graph_cost','random','periodic','uncertainty','graph_changed_first','stale','protect_changed','keep','KEEP_NO_INIT','DELETE_CHANGED_NO_INIT','DELETE_OTHER_NO_INIT','SINGLE_Q','VOTE_Q3','INDEPENDENT3_Q']
 names=['Closure EVSI (nominee)','EVSI fixed price','EVSI no cost ranking','EVSI old graph cost','Random','Periodic','Uncertainty','Changed first','Stale','Protect changed','Keep (with init.)','Keep (no init.)','Delete changed (no init.)','Delete other (no init.)','Single Qwen','Vote Qwen (3)','Independent 3 + Qwen']
 def put(f,lines): (out/f).write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
 for condition in ['change','no_change']:
  lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Method & EM (\%) & F1 (\%) & Utility & Tokens / $S$ item & Pairs & Correct / break \\',r'\midrule']
  for key,name in zip(order,names):
   v=s['results'][condition][key]
   assert abs(v['utility']-(v['accuracy']-0.000005*v['total_tokens']/2560))<1e-12
   lines.append(f"{name} & {100*v['accuracy']:.3f} & {100*v['F1']:.3f} & {v['utility']:.6f} & {v['total_tokens']/2560:.2f} & {v['pairs']:.3f} & {v['corrections_vs_keep']} / {v['breakages_vs_keep']} "+r'\\')
  caption=('Graph-change' if condition=='change' else 'No-change')+r' service results. Each method has 512 distinct service questions and 2,560 question--seed observations. Tokens include initialization and feedback; pairs are per episode--seed. Corrections/breakages use the matched keep output. No uncertainty intervals are shown.'
  lines += [r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{tab:'+condition+'}',r'\end{table*}']
  put(condition+'.tex',lines)
 lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Seed & EM (\%) & $J$ & $J-J_{\rm random}$ \\',r'\midrule']
 for i in range(5):
  v=s['results']['change']['closure_EVSI']['by_generation_seed'][str(i)]
  d=s['comparisons']['change']['random']['utility_difference_by_seed'][i]
  lines.append(f"{26111571+i} & {v['accuracy']*100:.3f} & {v['utility']:.6f} & {d:+.6f} "+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\caption{All five nominee seeds under graph change. Seeds share the same question groups.}',r'\label{tab:seeds}',r'\end{table}'];put('seeds.tex',lines)
 lines=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & $\lambda=0$ & $0.005/1000$ & $0.02/1000$ & $0.05/1000$ \\',r'\midrule']
 for key,name in zip(order,names):
  v=s['results']['change'][key]['utility_weight_sensitivity']
  lines.append(name+' & '+' & '.join(f"{v[k]:.6f}" for k in ['0','0.005','0.02','0.05'])+r' \\')
 lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Predeclared descriptive utility sensitivity on the fixed graph-change policies. Policies are not retuned at other weights. The primary weight remains $0.005/1000$.}',r'\label{tab:weights}',r'\end{table*}'];put('weights.tex',lines)
 # Validate all published episode counts and sum reconciliation.
 with (base/'EPISODE_SEED_AGGREGATES.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
 assert len(rows)==1360
 for condition in ['change','no_change']:
  for key in order:
   group=[x for x in rows if x['condition']==condition and x['method']==key]
   assert len(group)==40 and len({(x['episode'],x['generation_seed']) for x in group})==40
   v=s['results'][condition][key]
   assert sum(int(x['correct']) for x in group)==round(v['accuracy']*2560)
   assert sum(int(x['complete_tokens']) for x in group)==v['total_tokens']
 prov={'status':'PASS_AGGREGATE_PAPER_TABLE_RECONCILIATION','summary_sha256':hashlib.sha256((base/'SUMMARY.json').read_bytes()).hexdigest(),'episode_rows':len(rows),'methods':17,'conditions':2,'new_model_calls':0,'bootstrap_runs':0,'human_verification':False}
 (paper/'RESULTS_PROVENANCE.json').write_text(json.dumps(prov,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(prov))
if __name__=='__main__': main()
