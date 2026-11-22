import hashlib
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
    tok = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tok.padding_side = "left"
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else tok.unk_token_id
    if pad_id is None:
        pad_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    n = len(rows[0]["options"])
    assert n in (4, 8)
    source_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(4)]
    label_ids = [tok.encode(chr(65 + i), add_special_tokens=False) for i in range(n)]
    assert all(len(value) == 1 for value in label_ids)

    def code(text):
        if getattr(tok, "chat_template", None):
            encoded = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            encoded = tok.encode(text, add_special_tokens=True)
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids
        if isinstance(encoded, dict):
            return encoded["input_ids"]
        return encoded

    with torch.inference_mode(), Path(output_path).open("x", encoding="utf8") as out:
        for row in rows:
            menu = "\n".join(
                chr(65 + i) + ". " + value for i, value in enumerate(row["options"])
            )
            problem = row["stem"] + "\n" + menu
            n_groups = 4
            group_menu = "\n".join(
                chr(65 + i)
                + ". "
                + (
                    row["options"][2 * i]
                    + " OR "
                    + row["options"][2 * i + 1]
                    if n == 8
                    else row["options"][i]
                )
                for i in range(n_groups)
            )
            source_problem = row["stem"] + "\nHypotheses:\n" + group_menu
            source = [
                source_problem
                + ("\nEvidence: " + fact if fact else "")
                + "\nChoose the best hypothesis. Return A, B, C or D."
                for fact in (
                    row.get("first_fact", ""),
                    "",
                    row.get("control_fact", ""),
                )
            ]
            old_fact = row.get("second_fact", "")
            old_relay = [
                source_problem
                + "\nEvidence: "
                + old_fact
                + (
                    "\nFirst researcher proposes hypothesis " + chr(65 + i) + "."
                    if i >= 0
                    else ""
                )
                + "\nChoose the best hypothesis. Return A, B, C or D."
                for i in (-1, 0, 1, 2)
            ]
            suffixes = row.get("new_relay_interface", [""] * 4)
            if not isinstance(suffixes, list) or len(suffixes) != 4:
                suffixes = [""] * 4
            new_relay = [
                source_problem
                + "\nEvidence: "
                + old_fact
                + (
                    "\nFirst researcher proposes hypothesis " + chr(65 + i) + "."
                    if i >= 0
                    else ""
                )
                + (("\n" + suffixes[index]) if suffixes[index] else "")
                + "\nChoose the best hypothesis. Return A, B, C or D."
                for index, i in enumerate((-1, 0, 1, 2))
            ]
            terminals = [
                problem
                + "\nFirst researcher proposes hypothesis "
                + chr(65 + i)
                + ".\nSecond researcher proposes hypothesis "
                + chr(65 + j)
                + ".\nReturn only the best answer letter."
                for i, j in itertools.product(range(4), repeat=2)
            ]
            seq = [code(value) for value in source + old_relay + new_relay + terminals]
            microbatch = max(1, int(os.environ.get("MAB2_MICROBATCH", "2")))
            unique_keys = []
            unique_index = {}
            inverse = []
            for ids in seq:
                key = tuple(ids)
                index = unique_index.get(key)
                if index is None:
                    index = len(unique_keys)
                    unique_index[key] = index
                    unique_keys.append(ids)
                inverse.append(index)
            logits_parts = []
            for start in range(0, len(unique_keys), microbatch):
                batch = unique_keys[start : start + microbatch]
                max_len = max(len(ids) for ids in batch)
                input_ids = torch.full(
                    (len(batch), max_len), int(pad_id), dtype=torch.long
                )
                attention_mask = torch.zeros(
                    (len(batch), max_len), dtype=torch.long
                )
                for batch_index, ids in enumerate(batch):
                    input_ids[batch_index, -len(ids) :] = torch.as_tensor(
                        ids, dtype=torch.long
                    )
                    attention_mask[batch_index, -len(ids) :] = 1
                values = {
                    "input_ids": input_ids.to("cuda:0"),
                    "attention_mask": attention_mask.to("cuda:0"),
                }
                positions = (values["attention_mask"].long().cumsum(1) - 1).clamp_min(0)
                logits_parts.append(
                    model(**values, position_ids=positions, use_cache=False)
                    .logits[:, -1]
                    .float()
                    .cpu()
                )
                del values, positions, input_ids, attention_mask
            unique_logits = torch.cat(logits_parts, dim=0)
            logits = unique_logits[torch.as_tensor(inverse, dtype=torch.long)]
            probs = logits[:11, source_ids].softmax(1).tolist()
            labels = logits[11:, [value[0] for value in label_ids]].argmax(1).tolist()
            full = {str(i): probs[i] for i in range(11)}
            full.update({str(11 + i): labels[i] for i in range(len(labels))})
            old_token_sequences = [code(value) for value in old_relay]
            new_token_sequences = [code(value) for value in new_relay]
            old_fingerprints = [
                hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()
                for value in old_token_sequences
            ]
            new_fingerprints = [
                hashlib.sha256(json.dumps(value).encode("utf-8")).hexdigest()
                for value in new_token_sequences
            ]
            changed_rows = [
                i
                for i, (before, after) in enumerate(
                    zip(old_fingerprints, new_fingerprints)
                )
                if before != after
            ]
            out.write(
                json.dumps(
                    {
                        "key": row["key"],
                        "full": full,
                        "tokens": list(map(len, seq)),
                        "changed_rows": changed_rows,
                        "old_relay_fingerprints": old_fingerprints,
                        "new_relay_fingerprints": new_fingerprints,
                    }
                )
                + "\n"
            )
            out.flush()
    print(
        json.dumps(
            {
                "questions": len(rows),
                "n_options": n,
                "requests": len(rows) * len(seq),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])

