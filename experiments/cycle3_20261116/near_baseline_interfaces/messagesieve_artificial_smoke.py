"""Artificial-only interface qualification of pinned upstream MessageSieve."""
from pathlib import Path
import hashlib,json,sys,types
root=Path('E:/\u56fe\u7ed3\u6784\u6f14\u5316');up=root/'external/MessageSieve-20261116';sys.path.insert(0,str(up));sys.dont_write_bytecode=True
# Upstream package initializers import a Linux-only code execution sandbox.
# Load only the unchanged CPU routing module and its dataclasses on Windows.
for name,rel in [('masbench','masbench'),('masbench.core','masbench/core'),('masbench.methods','masbench/methods')]:
 package=types.ModuleType(name);package.__path__=[str(up/rel)];sys.modules[name]=package
from masbench.methods.messagesieve import MessageSieveMethod
from masbench.core.message import Message
from masbench.core.agent import AgentProfile
receiver=AgentProfile('worker_3','Verifier','Verifier','Check the supplied evidence and compare proposed answers.')
question='Which fictional station opened first?'
fixtures={
 'answer_evidence':['Final Answer: Alpha\nKey Evidence: Alpha station opened in 1901.','Final Answer: Beta\nKey Evidence: Beta station opened in 1912.'],
 'document_id_lists':['["D00", "D01"]','["D04", "D05"]'],
}
results={}
for name,texts in fixtures.items():
 m=MessageSieveMethod();m.configure({'task_family':'qa'});m.reset_for_run();messages=[]
 for i,text in enumerate(texts):
  sender=AgentProfile(f'worker_{i}',f'Source{i}','Evidence reader','Read the given evidence.')
  msg=Message(source=sender.agent_id,target=receiver.agent_id,round_index=0,channel='spatial',content=text,message_id=f'fixture_{name}_{i}')
  messages.append(m.prepare_message(preset=None,sender=sender,receiver=receiver,round_index=0,task_input=question,message=msg))
 before=[x.content for x in messages]
 filtered=m.filter_messages(preset=None,receiver=receiver,round_index=0,task_input=question,candidates=messages)
 judge=m.filter_judge_messages(preset=None,judge=receiver,task_input=question,candidates=messages)
 assert [x.content for x in messages]==before and [x.content for x in judge]==before
 results[name]=dict(input_messages=2,kept=len(filtered),parsed_answers=[bool(m.cards[x.message_id].answer) for x in messages],low_content=[m.cards[x.message_id].cues['low_content'] for x in messages],judge_preserved=True)
 m.reset_for_run();assert not m.cards and not m.previous_pair_vectors
out=root/'artifacts/private/cycle3_20261116/messagesieve_interface_v1';out.mkdir(parents=True,exist_ok=False)
pins={str(x.relative_to(up)).replace('\\','/'):hashlib.sha256(x.read_bytes()).hexdigest() for x in (up/'masbench/methods/messagesieve').glob('*.py')}
result=dict(status='PASS_ARTIFICIAL_INTERFACE_INVARIANTS',upstream_commit='580728b9251467009c702ccbfc7c1f65f887b967',source_pins=pins,fixtures=results,new_model_calls=0,new_GPU_hours=0,benchmark_labels_read=False,efficacy_inference=False,import_limitation='Normal upstream package import failed on Windows resource module. Namespace-only loader bypasses package initializers; unchanged routing modules run, not the full MAS runtime.')
(out/'RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
