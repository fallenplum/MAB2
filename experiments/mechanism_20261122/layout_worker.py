"""Score a four-layout orbit, keeping every public answer in semantic coordinates."""
import itertools
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_path, input_path, output_path):
    rows = json.loads(Path(input_path).read_bytes())
    assert all(set(row) == {'key', 'task', 'stem', 'options'} for row in rows)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to('cuda:0').eval()
    letters = 'ABCD'
    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in letters]
    assert all(len(x) == 1 for x in tokens) and len(set(x[0] for x in tokens)) == 4
    advice_pairs = list(itertools.product(range(-1, 4), repeat=2))
    with Path(output_path).open('x', encoding='utf8') as out, torch.inference_mode():
        for row in rows:
            layouts = []
            lengths = []
            for shift in range(4):
                order = [(j + shift) % 4 for j in range(4)]
                inverse = [order.index(j) for j in range(4)]
                prompt = row['stem'] + '\n' + '\n'.join(
                    c + '. ' + row['options'][i] for c, i in zip(letters, order)
                )
                texts = [prompt + '\nChoose your own answer. Return only A, B, C or D.']
                for pair in advice_pairs:
                    hints = [f'Adviser {j+1}: {letters[inverse[a]]}'
                             for j, a in enumerate(pair) if a >= 0]
                    texts.append('\n'.join([prompt, *hints, 'Choose your final answer. Return only A, B, C or D.']))
                enc = [tokenizer.apply_chat_template([{'role':'user','content':s}],
                       tokenize=True, add_generation_prompt=True) for s in texts]
                batch = tokenizer.pad({'input_ids':enc}, return_tensors='pt').to('cuda:0')
                positions = (batch['attention_mask'].long().cumsum(1)-1).clamp_min(0)
                logits = model(**batch, position_ids=positions, use_cache=False).logits[:, -1]
                prob = logits[:, [tokens[i][0] for i in inverse]].float().softmax(-1)
                layouts.append(prob.cpu().tolist())
                lengths.append(list(map(len,enc)))
            out.write(json.dumps(dict(key=row['key'], probs=layouts, input_tokens=lengths))+'\n')
            out.flush()
    print(json.dumps(dict(rows=len(rows), sequences=len(rows)*104, batch_forwards=len(rows)*4,
                          torch=torch.__version__, transformers=transformers.__version__)),flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
