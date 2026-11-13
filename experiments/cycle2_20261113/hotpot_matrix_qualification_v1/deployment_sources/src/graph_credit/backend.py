"""One-process, one-GPU frozen Transformers backend; no inference API costs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from .executor import Generation, GraphConfig, canonical


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_manifest(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve(strict=True)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
                   and ".cache" not in p.relative_to(root).parts)
    if not any(p.suffix in {".safetensors", ".bin"} for p in paths):
        raise FileNotFoundError("No local model weights found; point --model at a complete snapshot directory")
    return {str(p.relative_to(root)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths}


class HFBackend:
    """Weights and tokenizer are hashed before loading; every call gets its own RNG.

    ``physical_calls`` is append-only, including failures. A failed CUDA call is
    never retried silently and its resource use is not replaced with a zero.
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda:0",
                 physical_log: str | Path | None = None) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise RuntimeError("Formal frozen-model inference requires an explicitly selected CUDA GPU")
        self.device = torch.device(device)
        self.gpu_index = self.device.index
        if self.gpu_index is None or self.gpu_index >= torch.cuda.device_count():
            raise ValueError("Invalid CUDA device index")
        self.model_dir = str(Path(model_dir).resolve(strict=True))
        manifest = model_manifest(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        with torch.cuda.device(self.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is required by this frozen protocol; device does not support it")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, torch_dtype=torch.bfloat16, local_files_only=True,
                trust_remote_code=False, attn_implementation="eager",
            ).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.identity = {
            "backend": "hf_transformers_local_bf16_eager_v1", "model_files": manifest,
            "backend_code_sha256": sha256_file(Path(__file__)),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_chat_template_sha256": hashlib.sha256((self.tokenizer.chat_template or "").encode()).hexdigest(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
            "dtype": "bfloat16", "attention": "eager", "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "generation_override": "empty_generation_config_all_defaults_explicit_v1",
        }
        self.runtime = {"gpu_name": torch.cuda.get_device_name(self.device), "device": str(self.device),
                        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": platform.node()}
        self.physical_calls: list[dict[str, Any]] = []
        self.physical_log = Path(physical_log) if physical_log else None
        if self.physical_log:
            self.physical_log.parent.mkdir(parents=True, exist_ok=True)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _record(self, value: dict[str, Any]) -> None:
        self.physical_calls.append(value)
        if self.physical_log:
            with self.physical_log.open("a", encoding="utf-8") as handle:
                handle.write(canonical(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def generate(self, messages: list[dict[str, str]], seed: int,
                 max_new_tokens: int, config: GraphConfig) -> Generation:
        import torch
        from transformers import GenerationConfig

        # Truncation is rejected. Silently dropping evidence would change the
        # experimental information budget and invalidate cache equivalence.
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt")
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if encoded.shape[-1] > config.max_input_tokens:
            raise ValueError(f"Prompt has {encoded.shape[-1]} tokens; frozen cap is {config.max_input_tokens}")
        model_cap = getattr(self.model.config, "max_position_embeddings", config.max_input_tokens + max_new_tokens)
        if encoded.shape[-1] + max_new_tokens > model_cap:
            raise ValueError("Prompt plus output reservation exceeds model position limit")
        encoded = encoded.to(self.device)
        attention_mask = torch.ones_like(encoded)
        settings = GenerationConfig(
            max_new_tokens=max_new_tokens, min_new_tokens=0,
            do_sample=config.do_sample, temperature=config.temperature if config.do_sample else 1.0,
            top_p=config.top_p if config.do_sample else 1.0, top_k=0,
            num_beams=1, num_return_sequences=1, repetition_penalty=1.0,
            length_penalty=1.0, use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=False, output_scores=False,
        )
        call_id = uuid.uuid4().hex
        call_started_unix = time.time()
        started = time.perf_counter()
        try:
            with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.gpu_index]):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.reset_peak_memory_stats(self.device)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                start_event.record()
                with torch.inference_mode():
                    output = self.model.generate(input_ids=encoded, attention_mask=attention_mask,
                                                 generation_config=settings)
                end_event.record()
                torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - started
                gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
                peak = torch.cuda.max_memory_allocated(self.device)
            generated = output[0, encoded.shape[-1]:].tolist()
            text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            eos_ids = self.tokenizer.eos_token_id
            eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
            finish_reason = "eos" if generated and generated[-1] in eos_set else "length"
            result = Generation(text, int(encoded.shape[-1]), len(generated), elapsed, gpu_seconds,
                                int(peak), finish_reason, call_id)
            self._record({"call_id": call_id, "status": "SUCCESS", "seed": seed,
                          "started_unix": call_started_unix,
                          "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                          "elapsed_seconds": elapsed, "gpu_seconds": gpu_seconds,
                          "gpu_peak_allocated_bytes": int(peak), "finish_reason": finish_reason,
                          "prompt_sha256": hashlib.sha256(canonical(messages).encode()).hexdigest(),
                          "generation_config": settings.to_dict(), "runtime": self.runtime})
            return result
        except Exception as exc:
            self._record({"call_id": call_id, "status": "FAILED", "seed": seed,
                          "started_unix": call_started_unix, "input_tokens": int(encoded.shape[-1]),
                          "output_tokens": None, "elapsed_seconds": time.perf_counter() - started,
                          "error_type": type(exc).__name__, "error": str(exc), "runtime": self.runtime})
            raise
