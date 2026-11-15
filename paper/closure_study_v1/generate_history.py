"""Extract only historical aggregate points; never read Formal5 outcomes."""
from pathlib import Path
import argparse, csv, hashlib, json

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
root = args.repo
out = root / 'paper/closure_study_v1'
pins = {}
def read(rel):
    p = root / rel
    pins[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_bytes())
g2 = read('results/G2_STAGE_SUMMARY_20261105.json')
d3 = read('reports/autonomous_20261113/d3/D3_RESULTS.json')
c3 = read('reports/cycle2_20261113/path_formal3_results_v1/SUMMARY.json')
c4 = read('reports/cycle2_20261113/receiver_formal4_results_v1/SUMMARY.json')
read('reports/cycle2_20261113/path_posthoc_policy_v1/SUMMARY.json')
read('reports/cycle2_20261113/dead_node_cost_diagnostic_v1/SUMMARY.json')
rel = 'reports/autonomous_20261113/d2/method_metrics.csv'
pins[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
with (root / rel).open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
d2 = next(x for x in rows if x['method']=='C1_R1' and x['configuration']=='selected' and x['split']=='exposed')
mean = lambda x: sum(x)/len(x)
facts = {
    'status': 'HISTORICAL_POINTS_ONLY_FORMAL5_NOT_READ',
    'source_sha256': pins,
    'G2_cross_seed_risk': g2['statistics']['mean_cross_seed_risk'],
    'D2_fixed_nominee_exposed_action_gain': float(d2['action_gain']),
    'D2_fixed_nominee_exposed_actions': {k:int(d2[k]) for k in ['additions','deletions','stays']},
    'D3_fixed_nominee_exposed_enrichment': d3['drift_results']['exposed']['graph_only_order2']['enrichment']['estimate'],
    'C3': {'questions':96,'generation_seeds':2,'D':c3['primary_D'],'support':c3['nonzero_question_support'],
           'accuracy':{k:mean(c3['accuracy_by_condition_seed'][k]) for k in ['G1','SINGLE_Q','INDEPENDENT3_M']}},
    'C4': {'questions':96,'generation_seeds':2,'D':c4['credit_signals_by_cell']['Q_FULL']['D'],
           'support':c4['credit_signals_by_cell']['Q_FULL']['nonzero_question_support'],
           'accuracy':{k:mean(c4['accuracy_by_condition_seed'][k]) for k in ['G1__Q_FULL','SINGLE_Q','INDEPENDENT3_Q']}},
    'new_CI_computations':0,'bootstrap_runs':0,
    'cross_cohort_comparison_valid':False,
}
tex = r'''\section{Historical diagnostics and unsuccessful routes}
\label{app:history}
These results explain the sequence of research decisions; they are not
additional independent replications of the final candidate. Earlier exposed
evaluation questions remain development data. We retain each original
candidate decision and do not replace a failed nominee with a better-looking
control. The following tables reproduce aggregate point estimates only.

\paragraph{Prediction error did not establish editing value.}
The original G2 experiment used 200 training, 80 tuning and 120 held-out
GSM8K questions, five generation seeds, two anchor graphs and 12 edges.
Its cross-seed squared-risk estimates were %.9f for the edgewise constant
baseline, %.9f for the first-order predictor, and %.9f for the second-order
predictor. Thus the second-order model failed its frozen prediction gate.
These are cross-seed risk estimates, not ordinary empirical MSE values.
Subsequent D2 controls showed that penalizing the intercept can induce
spurious positive edge credit even for a graph-independent constant response.
This is a property of the original objective, not evidence that the original
implementation violated its contract. Removing the intercept penalty reduced
risk, but the fixed D2 nominee achieved %.3f percentage points of editing
gain on the already exposed evaluation block. It edited %d of 240 anchors
and therefore did not acquire useful abstention merely by reducing risk.
The D3 graph-only second-order nominee likewise failed its separate
drift-enrichment gate (point estimate %.9f).

\paragraph{Mechanism evidence is distinct from system quality.}
The controlled-path experiments C3 and C4 each used a new 96-question
development cohort and two generation seeds. Their topology-sensitive
credit filters passed, yet their frozen graph nominees were worse than
their single-model and independent-aggregation controls (Table~\ref{tab:prior}).
C4 varied receiver identity and access to raw context; the table reports
the preselected full-context Qwen receiver, without selecting a different
factorial cell after outcomes. C3 used a Mistral terminal receiver, whereas
C4's nominated terminal receiver was Qwen. Cohorts and operators differ;
the two rows do not estimate the effect of the receiver change.

\begin{table}[t]
\centering\small
\begin{tabular}{lrrrr}
\hline
Stage & Graph EM & Single Q & Indep. & $D$ \\
\hline
C3 & %.2f & %.2f & %.2f & %.4f \\
C4 & %.2f & %.2f & %.2f & %.4f \\
\hline
\end{tabular}
\caption{Historical within-cohort comparisons. EM is a percentage;
$D$ is the frozen topology-sensitive credit contrast in its original
proportion units. The independent aggregator is Mistral in C3 and Qwen
in C4. Nonzero question support is %d/96 and %d/96, respectively.
Each row averages two generation seeds on the same question cohort;
seeds are not additional independent question samples.}
\label{tab:prior}
\end{table}

\paragraph{Feedback must beat inexpensive structural controls.}
Post-hoc replay on exposed C3 outputs found that affected-edge refresh
could match a zero-new-feedback protection rule without improving it.
This diagnostic motivated including protection and no-initialization
static actions in the later contract; it does not confirm that contract.
An execution audit also found disconnected source calls in the old operator.
Eliminating those calls is ordinary dead-code elimination and is shared
by every applicable method in the new operator. Accounting savings from
old traces are not measured runtime speedups of the new implementation.
'''
v=facts
tex = tex % (v['G2_cross_seed_risk']['B0'],v['G2_cross_seed_risk']['C1'],v['G2_cross_seed_risk']['C2'],
             100*v['D2_fixed_nominee_exposed_action_gain'],int(d2['additions'])+int(d2['deletions']),
             v['D3_fixed_nominee_exposed_enrichment'],
             100*v['C3']['accuracy']['G1'],100*v['C3']['accuracy']['SINGLE_Q'],100*v['C3']['accuracy']['INDEPENDENT3_M'],v['C3']['D'],
             100*v['C4']['accuracy']['G1__Q_FULL'],100*v['C4']['accuracy']['SINGLE_Q'],100*v['C4']['accuracy']['INDEPENDENT3_Q'],v['C4']['D'],
             v['C3']['support'],v['C4']['support'])
for name,data in [('HISTORICAL_POINT_PROVENANCE.json',json.dumps(facts,indent=2)+'\n'),('sections/history.tex',tex)]:
    p=out/name
    with p.open('w',encoding='utf-8',newline='\n') as f:f.write(data)
print(json.dumps({'status':facts['status'],'source_count':len(pins),'new_model_calls':0,'new_CI_computations':0}))
