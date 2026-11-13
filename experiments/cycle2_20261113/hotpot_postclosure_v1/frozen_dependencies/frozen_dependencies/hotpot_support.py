"""Answer-only HotpotQA support; no dataset selection or inference.

Answer metric rules derive from the Apache-2.0 HotpotQA evaluation script by
Zhilin Yang, Peng Qi and Saizheng Zhang (2018). This module is a rewritten,
dependency-free adapter; upstream source, license and immutable pins are kept in
upstream/. JSON transport and strict no-gold input validation are MAB2 additions.
Supporting-fact and joint metrics are deliberately not represented as answer F1.
"""
from collections import Counter
import json
import re
import string


def normalize_answer(value: str) -> str:
    if not isinstance(value,str):raise TypeError('Answer must be a string')
    value=''.join(c for c in value.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',value).split())


def answer_metrics(prediction: str,gold: str) -> dict:
    pred,target=normalize_answer(prediction),normalize_answer(gold)
    em=float(pred==target)
    if ((pred in ('yes','no','noanswer') or target in ('yes','no','noanswer')) and pred!=target):
        return {'em':em,'f1':0.,'precision':0.,'recall':0.}
    p,g=pred.split(),target.split();overlap=sum((Counter(p)&Counter(g)).values())
    if overlap==0:return {'em':em,'f1':0.,'precision':0.,'recall':0.}
    precision=overlap/len(p);recall=overlap/len(g)
    return {'em':em,'f1':2*precision*recall/(precision+recall),'precision':precision,'recall':recall}


def score_aggregate_node(node: dict,gold: str) -> dict:
    """Evaluator-only: never infer an answer from malformed JSON or gold text."""
    meta=node['message_metadata'];parsed=node['parsed']
    if type(meta['format_ok']) is not bool:raise TypeError('Explicit format flag required')
    if set(parsed)!= {'answer','evidence','uncertainty'} or not all(isinstance(v,str) for v in parsed.values()):
        raise ValueError('Unvalidated normalized message')
    metrics=(answer_metrics(parsed['answer'],gold) if meta['format_ok'] else
             {'em':0.,'f1':0.,'precision':0.,'recall':0.})
    return {**metrics,'format_ok':meta['format_ok'],
            'metric_scope':'answer_only_not_supporting_fact_or_joint'}


def collector_payload(row: dict) -> dict:
    allowed={'id','question','context','type','level'}
    if set(row)-allowed:raise ValueError('Collector input contains nonallowlisted fields, potentially labels')
    for name in ('id','question'):
        if not isinstance(row.get(name),str) or not row[name].strip():raise ValueError('Missing input identity or question')
    context=row.get('context')
    if not isinstance(context,dict) or set(context)!= {'title','sentences'}:raise ValueError('Invalid context schema')
    titles,sentences=context['title'],context['sentences']
    if not isinstance(titles,list) or not isinstance(sentences,list) or not titles or len(titles)!=len(sentences):
        raise ValueError('Mismatched document arrays')
    if any(not isinstance(t,str) or not t.strip() for t in titles) or len(set(titles))!=len(titles):
        raise ValueError('Invalid or duplicate titles')
    if any(not isinstance(s,list) or not all(isinstance(x,str) for x in s) for s in sentences):
        raise ValueError('Sentences must be explicit string lists')
    # Preserve every sentence and document in original order. No support-label
    # filtering, retrieval or truncation is performed by this adapter.
    documents=[{'title':t,'sentences':s} for t,s in zip(titles,sentences)]
    return {'id':'hotpotqa_'+row['id'],'question':row['question'],
            'context':json.dumps(documents,ensure_ascii=False,separators=(',',':'),allow_nan=False)}


def normalized_question_group(question: str) -> str:
    import hashlib
    if not isinstance(question,str):raise TypeError('Question must be a string')
    return hashlib.sha256(' '.join(question.lower().split()).encode()).hexdigest()
