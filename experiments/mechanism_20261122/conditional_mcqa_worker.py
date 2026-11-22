import itertools
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_path, input_path, output_path):
    rows = json.loads(Path(input_path).read_bytes())
    assert len(rows) == 128
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                  local_files_only=True).to("cuda:0").eval()
    n = len(rows[0]["options"])
    assert n in (4, 8)
    source_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tok.encode(chr(65 + i), add_special_tokens=False) for i in range(n)]
    assert all(len(v) == 1 for v in label_ids)

    def code(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                       add_generation_prompt=True)

    with torch.inference_mode(), Path(output_path).open("x", encoding="utf8") as out:
        for row in rows:
            menu = "\n".join(chr(65 + i) + ". " + v for i, v in enumerate(row["options"]))
            problem = row["stem"] + "\n" + menu
            group_menu = "\n".join(chr(65 + i) + ". " +
                                      (row["options"][2 * i] + " OR " + row["options"][2 * i + 1]
                                       if n == 8 else row["options"][i])
                                      for i in range(4))
            source_problem = row["stem"] + "\nHypotheses:\n" + group_menu
            source = [source_problem + ("\nEvidence: " + fact if fact else "") +
                      "\nChoose the best hypothesis. Return A, B, C or D."
                      for fact in (row.get("first_fact", ""), "", row.get("control_fact", ""))]
            old_relay = [source_problem + "\nEvidence: " + row.get("second_fact", "") +
                         ("\nFirst researcher proposes hypothesis " + chr(65 + i) + "." if i >= 0 else "") +
                         "\nChoose the best hypothesis. Return A, B, C or D."
                         for i in (-1, 0, 1, 2)]
            new_fact = row.get("second_fact_new", row.get("second_fact", ""))
            new_relay = [source_problem + "\nEvidence: " + (new_fact if i == 0 else row.get("second_fact", "")) +
                         ("\nFirst researcher proposes hypothesis " + chr(65 + i) + "." if i >= 0 else "") +
                         "\nChoose the best hypothesis. Return A, B, C or D."
                         for i in (-1, 0, 1, 2)]
            terminals = [problem + "\nFirst researcher proposes hypothesis " + chr(65 + i) +
                         ".\nSecond researcher proposes hypothesis " + chr(65 + j) +
                         ".\nReturn only the best answer letter."
                         for i, j in itertools.product(range(4), repeat=2)]
            seq = [code(v) for v in source + old_relay + new_relay + terminals]
            x = tok.pad({"input_ids": seq}, return_tensors="pt").to("cuda:0")
            pos = (x.attention_mask.long().cumsum(1) - 1).clamp_min(0)
            logits = model(**x, position_ids=pos, use_cache=False).logits[:, -1]
            probs = logits[:11, source_ids].float().softmax(1).cpu().tolist()
            labels = logits[11:, [v[0] for v in label_ids]].argmax(1).cpu().tolist()
            full = {str(i): probs[i] for i in range(11)}
            full.update({str(11 + i): labels[i] for i in range(len(labels))})
            out.write(json.dumps({"key": row["key"], "full": full,
                                  "tokens": list(map(len, seq))}) + "\n")
            out.flush()
    print(json.dumps({"questions": len(rows), "n_options": n, "requests": len(rows) * len(seq),
                      "torch": torch.__version__, "transformers": transformers.__version__}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
