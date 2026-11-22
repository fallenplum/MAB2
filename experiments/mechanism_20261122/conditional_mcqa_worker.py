import itertools
import json
import os
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
        encoded = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                           add_generation_prompt=True)
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        if isinstance(encoded, dict):
            return encoded["input_ids"]
        return encoded

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
            changed_rows = [index for index, advice in enumerate((-1, 0, 1, 2))
                            if (new_fact if advice == 0 else row.get("second_fact", ""))
                            != row.get("second_fact", "")]
            new_relay = [source_problem + "\nEvidence: " + (new_fact if i == 0 else row.get("second_fact", "")) +
                         ("\nFirst researcher proposes hypothesis " + chr(65 + i) + "." if i >= 0 else "") +
                         "\nChoose the best hypothesis. Return A, B, C or D."
                         for i in (-1, 0, 1, 2)]
            terminals = [problem + "\nFirst researcher proposes hypothesis " + chr(65 + i) +
                         ".\nSecond researcher proposes hypothesis " + chr(65 + j) +
                         ".\nReturn only the best answer letter."
                         for i, j in itertools.product(range(4), repeat=2)]
            seq = [code(v) for v in source + old_relay + new_relay + terminals]
            microbatch = max(1, int(os.environ.get("MAB2_MICROBATCH", "2")))
            logits_parts = []
            for start in range(0, len(seq), microbatch):
                batch = seq[start:start + microbatch]
                max_len = max(len(ids) for ids in batch)
                input_ids = torch.full((len(batch), max_len), int(tok.pad_token_id), dtype=torch.long)
                attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
                for row_index, ids in enumerate(batch):
                    input_ids[row_index, -len(ids):] = torch.as_tensor(ids, dtype=torch.long)
                    attention_mask[row_index, -len(ids):] = 1
                x = {"input_ids": input_ids.to("cuda:0"), "attention_mask": attention_mask.to("cuda:0")}
                pos = (x["attention_mask"].long().cumsum(1) - 1).clamp_min(0)
                logits_parts.append(model(**x, position_ids=pos, use_cache=False).logits[:, -1].float().cpu())
                del x, pos, input_ids, attention_mask
            logits = torch.cat(logits_parts, dim=0)
            probs = logits[:11, source_ids].softmax(1).tolist()
            labels = logits[11:, [v[0] for v in label_ids]].argmax(1).tolist()
            full = {str(i): probs[i] for i in range(11)}
            full.update({str(11 + i): labels[i] for i in range(len(labels))})
            out.write(json.dumps({"key": row["key"], "full": full,
                                  "tokens": list(map(len, seq)),
                                  "changed_rows": changed_rows}) + "\n")
            out.flush()
    print(json.dumps({"questions": len(rows), "n_options": n, "requests": len(rows) * len(seq),
                      "torch": torch.__version__, "transformers": transformers.__version__}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
