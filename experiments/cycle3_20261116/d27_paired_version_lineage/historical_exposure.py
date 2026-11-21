"""Prove historical ancestor overlap with exact, contained original prompt text."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalize(s):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",s).casefold()))


def count_exposure(data, cache, registry):
    records=json.loads(registry.read_bytes())
    files=[r for r in records if r["relative"].replace("\\","/").casefold().startswith("humaneval/test/")]
    assert files
    with gzip.open(data/"EvoEval_subtle.jsonl.gz","rt",encoding="utf-8") as f:
        linked={json.loads(line)["parent"] for line in f}
    with gzip.open(data/"HumanEval.jsonl.gz","rt",encoding="utf-8") as f:
        sources={row["task_id"]:normalize(row["prompt"]) for row in map(json.loads,f)}
    assert len(linked)==100 and linked<=sources.keys()
    assert all(len(sources[k])>=40 for k in linked)
    seen=set();hits=set();rows=0
    for info in files:
        path=cache.joinpath(*info["relative"].replace("\\","/").split("/"))
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==info["sha256"]
        cached=json.loads(raw)["records"]
        assert len(cached)==info["records"]
        rows+=len(cached)
        for record in cached:
            for key in ("origin_query","prompt"):
                text=record.get(key)
                if isinstance(text,list):
                    text=" ".join(x["content"] for x in text if isinstance(x,dict) and isinstance(x.get("content"),str))
                if not isinstance(text,str) or not text.strip():continue
                text=normalize(text)
                if text in seen:continue
                seen.add(text)
                hits.update(i for i,original in sources.items() if original in text)
    return dict(project_cache_files=len(files),historical_records=rows,unique_normalized_prompt_fields=len(seen),
                original_parents_covered=len(hits),paired_families=100,paired_families_with_observed_ancestor=len(linked&hits),
                paired_families_not_matched_in_this_subset=len(linked-hits),cached_predictions_or_gold_used=False,
                model_calls=0,GPU_hours=0,source_registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
                scope="Only pre-existing HumanEval/test cache entries. Exact full normalized ancestor prompt containment. An unmatched family is not a clean certificate.")
