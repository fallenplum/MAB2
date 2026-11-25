"""Small auditable retrieval and verifier runtime for the E0 smoke test."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEPARATOR = "####SPECIAL####SEPARATOR####"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    title: str
    text: str
    rank: int
    score: float


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(token) > 2}


class SQLiteTopicRetriever:
    def __init__(self, db_path: Path):
        uri = f"file:{db_path.as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def get_passages(self, topic: str, query: str, k: int = 5) -> list[Evidence]:
        row = self.connection.execute("SELECT text FROM documents WHERE title = ?", (topic,)).fetchone()
        if row is None:
            raise KeyError(f"topic not found in knowledge source: {topic}")
        query_tokens = _tokens(topic + " " + query)
        passages = [part for part in str(row[0]).split(SEPARATOR) if part.strip()]
        scored = []
        for index, passage in enumerate(passages):
            overlap = len(query_tokens & _tokens(passage))
            score = overlap / max(len(query_tokens), 1)
            evidence_id = hashlib.sha256(f"{topic}|{index}|{passage}".encode()).hexdigest()[:20]
            scored.append(Evidence(evidence_id, topic, passage, index, score))
        scored.sort(key=lambda item: (-item.score, item.rank))
        return [Evidence(item.evidence_id, item.title, item.text, rank, item.score) for rank, item in enumerate(scored[:k])]

    def close(self) -> None:
        self.connection.close()


class QwenFactVerifier:
    def __init__(self, model_path: Path, max_input_tokens: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.max_input_tokens = max_input_tokens

    def verify(self, topic: str, fact: str, evidence: Iterable[Evidence]) -> dict[str, object]:
        context = "\n\n".join(f"[{item.rank}] {item.text}" for item in evidence)
        prompt = (
            "You are a factuality verifier. Decide whether the atomic fact is directly supported by the evidence. "
            "Reply with exactly SUPPORT or UNSUPPORTED.\n\n"
            f"Topic: {topic}\nAtomic fact: {fact}\nEvidence:\n{context}\nDecision:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        elapsed = time.perf_counter() - start
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
        if "SUPPORT" in text and "UNSUPPORTED" not in text:
            label = 1
        elif "UNSUPPORTED" in text:
            label = 0
        else:
            label = None
        return {
            "label": label,
            "raw_output": text,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[0]),
            "latency_seconds": elapsed,
        }

