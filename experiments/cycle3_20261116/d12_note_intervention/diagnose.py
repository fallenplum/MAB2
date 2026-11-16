"""Exploratory request-matched annotation diagnosis, after audited scoring."""
from pathlib import Path
import argparse,copy,hashlib,json,re
import numpy as np

def without_notes(request):
    value=copy.deepcopy(request)
    assert len(value['messages'])==2
    user=json.loads(value['messages'][1]['content'])
    assert set(user)=={'question','documents','evidence_notes'}
    user['evidence_notes']=[]
    value['messages'][1]['content']=json.dumps(user,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return value

def phrase_in(answer,note):
    a=re.findall(r'\w+',answer.casefold());n=re.findall(r'\w+',note.casefold())
    return bool(a) and any(n[i:i+len(a)]==a for i in range(len(n)-len(a)+1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--formal',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();f=Path(a.formal);out=Path(a.output);assert not out.exists()
    read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert read(f/'INDEPENDENT_SCORING_AUDIT.json')['status']=='PASS_INDEPENDENT_RAW_SCORING_AND_GATES'
    s=read(f/'scoring_v1/SUMMARY.json');arrays=np.load(f/'scoring_v1/PRIVATE_ARRAYS.npz');names=list(s['conditions']);ix={n:i for i,n in enumerate(names)};seeds=s['seeds']
    units=sorted((f/'monitor/archive').glob('worker_*/unit_*'),key=lambda p:read(p/'INPUT.json')['index']);assert len(units)==96
    stats={n:dict(matched=0,total=192,EM_sum=0.,single_EM_sum=0.,corrections=0,destructions=0,paid_prefix_ignore_tokens_sum=0.,paid_prefix_ignore_EM_sum=0.) for n in names if n!='SINGLE_Q'}
    copying={k:dict(observations=0,answer_copied_from_any_note=0,answer_copied_from_R_note=0,multiword_answer_copied_from_any_note=0) for k in ['correction','destruction','both_correct','both_wrong']}
    private=[]
    for i,u in enumerate(units):
        rows={(r['execution_seed'],r['condition']):r for r in read(u/'MATRIX.json')['conditions']}
        for j,seed in enumerate(seeds):
            single=rows[seed,'SINGLE_Q']['nodes']['M'];base=float(arrays['em'][i,j,ix['SINGLE_Q']])
            for name in stats:
                row=rows[seed,name];m=row['nodes']['M'];matched=without_notes(m['request'])==single['request']
                if not matched:continue
                st=stats[name];st['matched']+=1;v=float(arrays['em'][i,j,ix[name]]);st['EM_sum']+=v;st['single_EM_sum']+=base
                st['corrections']+=int(v==1 and base==0);st['destructions']+=int(v==0 and base==1)
                prefix=sum(n['generation']['input_tokens']+n['generation']['output_tokens'] for node,n in row['nodes'].items() if node!='M')
                cost=prefix+single['generation']['input_tokens']+single['generation']['output_tokens'];st['paid_prefix_ignore_tokens_sum']+=cost;st['paid_prefix_ignore_EM_sum']+=base
                private.append(dict(index=i,seed=seed,condition=name,exact_terminal_without_notes_matches_single=True,paid_prefix_ignore_tokens=cost))
                if name=='G1':
                    group='correction' if v>base else 'destruction' if v<base else 'both_correct' if v else 'both_wrong'
                    notes=json.loads(m['request']['messages'][1]['content'])['evidence_notes'];answer=m['generation']['text'];c=copying[group];c['observations']+=1
                    any_copy=any(phrase_in(answer,n['text']) for n in notes)
                    c['answer_copied_from_any_note']+=any_copy;c['answer_copied_from_R_note']+=any(n['source']=='R' and phrase_in(answer,n['text']) for n in notes)
                    c['multiword_answer_copied_from_any_note']+=any_copy and len(re.findall(r'\w+',answer.casefold()))>=2
    for name,st in stats.items():
        count=st['matched']
        if count:
            st.update(matched_EM=st['EM_sum']/count,matched_single_EM=st['single_EM_sum']/count,paired_EM_delta=(st['EM_sum']-st['single_EM_sum'])/count,paid_prefix_ignore_service_tokens=st['paid_prefix_ignore_tokens_sum']/count,paid_prefix_ignore_service_utility=(st['paid_prefix_ignore_EM_sum']-5e-6*st['paid_prefix_ignore_tokens_sum'])/count)
    for name in ['G0','G1','FLAT_ANNOTATION']:assert stats[name]['matched']==192
    result=dict(status='EXPLORATORY_EXACT_TERMINAL_NOTE_INTERVENTION',questions=96,question_seed_observations=192,matched_controls=stats,G1_literal_note_copy_associations=copying,source_sha256=sha(__file__),scoring_summary_sha256=sha(f/'scoring_v1/SUMMARY.json'),formal_gates_unchanged=s['gates'],new_GPU_calls=0,confidence_intervals=False,bootstrap_runs=0,limits=['Matching changes only terminal evidence_notes while preserving exact remaining request fields, model and RNG. It does not identify which note or semantic mechanism caused a change.','Substring association is not semantic contradiction verification or causal mediation; single-word matches can be incidental.','Ignoring notes after generating their prefix retains every prefix token. It cannot be charged as a free single-model service.','All data are exposed development; no formal gate is rescued and no policy efficacy is established.'])
    out.mkdir();(out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');(out/'PRIVATE_PROVENANCE.json').write_text(json.dumps(private,indent=2)+'\n');print(json.dumps(dict(status=result['status'],matched_G1=stats['G1']['matched'],new_GPU_calls=0)))

if __name__=='__main__':main()
