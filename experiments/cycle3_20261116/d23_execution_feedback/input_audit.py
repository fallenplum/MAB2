"""D23 frozen input-only eligibility/exposure audit. Never execute solutions."""
import argparse,collections,hashlib,json,re,time,unicodedata
from pathlib import Path

def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def grams(s):
 t=s.split();return set(tuple(t[i:i+5]) for i in range(max(0,len(t)-4)))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--train',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--tokenizers',nargs=2,required=True);a.add_argument('--out',type=Path,required=True);a=a.parse_args();start=time.monotonic();a.out.mkdir(parents=True,exist_ok=True)
 assert not (a.out/'INPUT_AUDIT.json').exists()
 assert hashlib.sha256(a.train.read_bytes()).hexdigest()=='45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae'
 rows=[json.loads(l) for l in a.train.open(encoding='utf-8') if l.strip()];assert len(rows)==5000
 # Inspect every cached benchmark's prompt fields, not predictions, scores or labels.
 qs=[norm(x['question']) for x in rows];qgrams=[grams(q) for q in qs];inverted=collections.defaultdict(set)
 for i,g in enumerate(qgrams):
  for v in g:inverted[v].add(i)
 exact_questions=collections.defaultdict(list)
 for i,q in enumerate(qs):exact_questions[q].append(i)
 short_questions=[i for i,g in enumerate(qgrams) if not g]
 exposed=set();seen=set();cachefiles=[];cachedrows=0;malformed=[]
 for f in sorted(a.cache.rglob('*.json')):
  d=json.loads(f.read_bytes());records=d.get('records') if isinstance(d,dict) else None
  if not isinstance(records,list):malformed.append(str(f.relative_to(a.cache)));continue
  cachedrows+=len(records);cachefiles.append(dict(relative=str(f.relative_to(a.cache)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),records=len(records)))
  for row in records:
   for key in ['origin_query','prompt']:
    value=row.get(key)
    if isinstance(value,list):value=' '.join(x.get('content','') for x in value if isinstance(x,dict) and isinstance(x.get('content'),str))
    if not isinstance(value,str) or not value.strip():continue
    n=norm(value);h=digest(n)
    if h in seen:continue
    seen.add(h);g=grams(n);counts=collections.Counter(i for v in g for i in inverted.get(v,()))
    for i,c in counts.items():
     # Symmetric containment conservatively catches prompt wrappers and near copies.
     if c/min(len(g),len(qgrams[i]))>=.8:exposed.add(i)
    exposed.update(exact_questions.get(n,[]))
    for i in (set(counts)|set(short_questions)) if g else range(len(qs)):
     q=qs[i]
     if q==n or (len(q)>=40 and q in n) or (len(n)>=40 and n in q):exposed.add(i)
 # Documentation exposed APPS id0 and an interview example; exclude id0 explicitly.
 exposed.update(i for i,r in enumerate(rows) if r['id']==0)
 parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):parent[find(i)]=find(j)
 exact={};urls={}
 for i,r in enumerate(rows):
  for mapping,key in [(exact,qs[i]),(urls,r.get('url','').strip())]:
   if not key:continue
   if key in mapping:union(i,mapping[key])
   else:mapping[key]=i
  counts=collections.Counter(j for v in qgrams[i] for j in inverted[v] if j<i)
  for j,c in counts.items():
   if c/min(len(qgrams[i]),len(qgrams[j]))>=.8:union(i,j)
 exposuregroups={find(i) for i in exposed};eligible=[];reasons=collections.Counter();private=[]
 from transformers import AutoTokenizer
 toks=[AutoTokenizer.from_pretrained(v,local_files_only=True) for v in a.tokenizers]
 for i,r in enumerate(rows):
  reason=None;pairs=[];hidden=[]
  if r['difficulty']!='introductory':reason='not_introductory'
  elif find(i) in exposuregroups:reason='cached_or_documentation_overlap_group'
  else:
   try:
    io=json.loads(r['input_output']);assert isinstance(io,dict) and not io.get('fn_name');assert isinstance(io['inputs'],list) and isinstance(io['outputs'],list) and len(io['inputs'])==len(io['outputs'])
    for x,y in zip(io['inputs'],io['outputs']):
     if isinstance(x,str) and isinstance(y,str) and x.strip() and y.strip():
      pair=(x.replace('\r\n','\n').strip(),y.replace('\r\n','\n').strip())
      if pair not in pairs:pairs.append(pair)
    # Ambiguous expected output for the same input disqualifies the task.
    assert len({x for x,y in pairs})==len(pairs)
   except (ValueError,TypeError,KeyError,AssertionError):reason='invalid_or_function_io'
   if reason is None:
    hidden=[j for j,(x,y) in enumerate(pairs) if ' '.join(x.split()) not in ' '.join(r['question'].split())]
    if len(pairs)<4:reason='fewer_than_four_distinct_pairs'
    elif len(hidden)<2:reason='fewer_than_two_nonliteral_hidden_inputs'
  lengths=[len(t.encode(r['question'],add_special_tokens=False)) for t in toks] if reason is None else []
  if lengths and max(lengths)>4096:reason='question_over_4096_tokens'
  group=digest(str(min(rows[j]['id'] for j in range(len(rows)) if find(j)==find(i))))
  private.append(dict(id=r['id'],question_hash=digest(qs[i]),group=group,reason=reason,pair_count=len(pairs),hidden_candidate_count=len(hidden),token_lengths=lengths))
  if reason:reasons[reason]+=1
  else:eligible.append(dict(id=r['id'],group=group,question=r['question'],pairs=pairs,hidden_indices=hidden,token_lengths=lengths,url=r.get('url','')))
 groups=len({r['group'] for r in eligible});status='PASS_INPUT_FEASIBILITY_ONLY' if groups>=128 and not malformed else 'FAIL_FEASIBILITY_OR_COVERAGE'
 for name,obj in [('PRIVATE_ELIGIBLE.json',eligible),('PRIVATE_AUDIT.json',private),('PRIVATE_CACHE_PROVENANCE.json',cachefiles)]:
  with (a.out/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False)
 result=dict(status=status,original_train_rows=len(rows),eligible_tasks=len(eligible),eligible_independent_groups=groups,required_groups=128,exclusion_counts=dict(reasons),cache_files=len(cachefiles),cache_rows=cachedrows,unique_prompt_fields=len(seen),unparsed_cache_files=len(malformed),all_local_cache_coverage='all JSON files under supplied bench-release root',overlap_rule='normalized equality/substring or 5-word-gram containment>=0.8; union same canonical URL and near-duplicate prompts',pretraining_contamination_audited=False,official_test_downloaded=False,solutions_executed=False,model_calls=0,GPU_hours=0,elapsed_seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),train_sha256=hashlib.sha256(a.train.read_bytes()).hexdigest())
 with (a.out/'INPUT_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
