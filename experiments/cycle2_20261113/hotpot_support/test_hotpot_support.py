import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import string
import unittest
from hotpot_support import answer_metrics,collector_payload,normalized_question_group,score_aggregate_node


class HotpotSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).parent/'upstream'
        pins=json.loads((cls.root/'UPSTREAM.json').read_bytes())
        cls.pins=pins
        source=(cls.root/'hotpot_evaluate_v1.py').read_text(encoding='utf-8')
        module=ast.parse(source)
        selected=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in ('normalize_answer','f1_score','exact_match_score')]
        assert len(selected)==3
        cls.reference={'Counter':Counter,'re':re,'string':string}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'<pinned official metric functions>','exec'),cls.reference)

    def test_upstream_bytes_match_git_blobs(self):
        for filename,key in [('hotpot_evaluate_v1.py','source_blob'),('LICENSE.txt','license_blob')]:
            raw=(self.root/filename).read_bytes()
            self.assertEqual(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),self.pins[key])

    def test_differential_official_metrics(self):
        rng=random.Random(96173)
        vocabulary=['yes','no','noanswer','the','a','an','blue','red','card','Mira','Leon','one','two','?',',','.','!','  ','\n','é','甲']
        cases=[('',''),('the','a'),('yes','yes please'),('no answer','noanswer'),('red red blue','red blue blue'),('The Blue Card!','blue card')]
        cases += [(' '.join(rng.choices(vocabulary,k=rng.randrange(12))),' '.join(rng.choices(vocabulary,k=rng.randrange(12)))) for _ in range(2000)]
        for p,g in cases:
            actual=answer_metrics(p,g);f,pr,rc=self.reference['f1_score'](p,g)
            self.assertEqual(actual,{'em':float(self.reference['exact_match_score'](p,g)),'f1':f,'precision':pr,'recall':rc})

    def test_malformed_transport_is_not_repaired_with_gold(self):
        node={'parsed':{'answer':'blue','evidence':'','uncertainty':''},'message_metadata':{'format_ok':False},'generation':{'text':'blue'}}
        self.assertEqual(score_aggregate_node(node,'blue')['em'],0)
        self.assertEqual(score_aggregate_node(node,'the')['em'],0)
        node['message_metadata']['format_ok']=True
        self.assertEqual(score_aggregate_node(node,'blue')['em'],1)

    def test_context_preservation(self):
        row={'id':'synthetic_only','question':'What color?','context':{'title':['First','Second'],'sentences':[['Mira has blue.','Leon has red.'],['Another sentence.']]},'type':'bridge','level':'easy'}
        p=collector_payload(row);self.assertEqual(set(p),{'id','question','context'})
        self.assertEqual(json.loads(p['context']),[{'title':t,'sentences':s} for t,s in zip(row['context']['title'],row['context']['sentences'])])
        for field in ['answer','supporting_facts','label','quality','gold','target']:
            with self.assertRaises(ValueError):collector_payload({**row,field:'forbidden'})

    def test_invalid_context_rejected(self):
        row={'id':'synthetic','question':'What?'}
        for context in [{'title':['A'],'sentences':[]},{'title':['A','A'],'sentences':[[],[]]},{'title':['A'],'sentences':['one string']},{'title':['A'],'sentences':[[]],'supporting_facts':[]}]:
            with self.assertRaises(ValueError):collector_payload({**row,'context':context})

    def test_question_groups(self):
        self.assertEqual(normalized_question_group(' What  Color? '),normalized_question_group('what color?'))
        self.assertNotEqual(normalized_question_group('What color?'),normalized_question_group('What colour?'))


if __name__=='__main__':unittest.main()
