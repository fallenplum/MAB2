"""Author deterministic synthetic long context; no benchmark or answer-file input."""
import argparse
import hashlib
import json
from pathlib import Path

TOKENIZER_PIN='e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49'


def make_question(filler_records):
    paragraphs=['The following is a fictional archive registry. Each record describes a distinct venue and archive location.']
    for i in range(1,filler_records+1):
        if i==8:
            paragraphs.append('Venue file: Alder Pavilion\nAlder Pavilion stores its visitor records in Cedar Hall. Its office also maintains a separate attendance book at the pavilion.')
        if i==filler_records-7:
            paragraphs.append('Archive file: Cedar Hall\nCedar Hall is located in Marlow District. The archive contains shelves for visitor records and a small reading room.')
        paragraphs.append(f'Registry entry {i:03d}: Grove Annex {i:03d} sends its records to Slate Room {i:03d}. '
            f'Slate Room {i:03d} is located in Harbor Sector {i:03d}. '
            'The venue has an information desk, a notice board, and a separate public entrance. '
            'This numbered record concerns only the named annex and room.')
    return {'id':'synthetic_long_two_hop_registry_v1',
            'question':"In which district is the archive hall that holds Alder Pavilion's visitor records located?",
            'context':'\n\n'.join(paragraphs)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-json',type=Path,required=True)
    p.add_argument('--filler-records',type=int,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args()
    assert a.filler_records>=20
    raw=a.tokenizer_json.read_bytes();assert hashlib.sha256(raw).hexdigest()==TOKENIZER_PIN
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(a.tokenizer_json));question=make_question(a.filler_records)
    complete=question['question']+'\n\nSupplied context:\n'+question['context']
    count=len(tokenizer.encode(complete,add_special_tokens=False).ids)
    result={'synthetic_only':True,'filler_records':a.filler_records,'low_level_Mistral_complete_user_text_tokens':count,
            'target':[3500,4000],'within_target':3500<=count<=4000,'tokenizer_json_sha256':TOKENIZER_PIN,
            'server_HF_runtime_equivalence':False,'server_measurement_required':True,'context_truncated':False}
    if a.output:
        assert result['within_target'],'Revise the authored input before freezing; do not crop at runtime'
        a.output.write_bytes((json.dumps(question,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
        result['synthetic_input_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
        a.output.with_name('SYNTHETIC_INPUT_PREPARATION.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(result))


if __name__=='__main__':main()
