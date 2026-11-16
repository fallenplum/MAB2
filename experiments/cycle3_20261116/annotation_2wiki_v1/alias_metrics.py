"""Official answer-only max-over-alias scoring; no supporting-fact evaluation."""
from hotpot_support import answer_metrics as single_answer_metrics

def answer_metrics(prediction,golds):
    assert isinstance(golds,list) and golds and all(isinstance(v,str) for v in golds)
    values=[single_answer_metrics(prediction,gold) for gold in golds]
    return {key:max(v[key] for v in values) for key in ['em','f1']}
