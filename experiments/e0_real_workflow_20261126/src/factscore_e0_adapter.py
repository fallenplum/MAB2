"""Minimal E0 adapter for the official FActScore labeled-data projection.

The online view exposes only the frozen topic, generation and model-provided
atomic-fact texts. Human labels remain outside this module and are reserved
for the post-hoc scorer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

from .e0_certificate import Certificate, fixed_threshold_certificate


@dataclass(frozen=True)
class OnlineFactGroup:
    group_id: str
    topic: str
    generation: str
    atomic_facts: tuple[str, ...]
    source_row_index: int
    decomposition_source: str = "official_model_atomic_facts"


def _group_id(topic: str, salt: str) -> str:
    return "e0g_" + hashlib.sha256((salt + "|" + topic).encode()).hexdigest()[:16]


def load_online_groups(
    labeled_jsonl: Path,
    group_ids: set[str],
    salt: str,
    fact_source: str = "model",
    allow_missing: bool = False,
) -> list[OnlineFactGroup]:
    """Load only online-safe fields for the selected groups.

    ``fact_source='human'`` means human atomic-fact *text* is used as a
    frozen decomposition input. Labels are deliberately discarded and remain
    available only to the post-hoc scorer.
    """

    if fact_source not in {"model", "human"}:
        raise ValueError("fact_source must be 'model' or 'human'")

    groups: list[OnlineFactGroup] = []
    with labeled_jsonl.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            topic = row["topic"]
            group_id = _group_id(topic, salt)
            if group_id not in group_ids:
                continue
            facts = []
            for annotation in row.get("annotations") or []:
                field = "model-atomic-facts" if fact_source == "model" else "human-atomic-facts"
                for fact in (annotation or {}).get(field) or []:
                    text = fact.get("text")
                    if isinstance(text, str) and text.strip():
                        facts.append(text.strip())
            if not facts and not allow_missing:
                raise ValueError(f"no frozen atomic facts for {group_id}")
            groups.append(
                OnlineFactGroup(
                    group_id,
                    topic,
                    row["output"],
                    tuple(facts),
                    row_index,
                    decomposition_source=(
                        "official_model_atomic_facts"
                        if fact_source == "model" and facts
                        else "official_human_atomic_fact_text_only"
                        if fact_source == "human" and facts
                        else "INVALID_NO_FROZEN_FACTS"
                    ),
                )
            )
    found = {group.group_id for group in groups}
    missing = group_ids - found
    if missing:
        raise ValueError(f"missing selected groups: {sorted(missing)}")
    return sorted(groups, key=lambda group: group.group_id)


def fixed_contract(group: OnlineFactGroup, threshold: float = 0.80) -> dict[str, object]:
    """Create the D contract without exposing any target labels."""

    n = len(group.atomic_facts)
    if n == 0:
        raise ValueError("empty fact list is not a valid D task")
    return {
        "group_id": group.group_id,
        "n": n,
        "required_support": ceil(threshold * n),
        "threshold": threshold,
        "weights": [1] * n,
        "fact_ids": [f"{group.group_id}:f{i:03d}" for i in range(n)],
    }


def certificate_from_verifier_labels(labels: Sequence[int | None], threshold: float = 0.80) -> Certificate:
    return fixed_threshold_certificate(labels, threshold=threshold)


def online_payload(group: OnlineFactGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "topic": group.topic,
        "generation": group.generation,
        "atomic_facts": list(group.atomic_facts),
        "source_row_index": group.source_row_index,
        "decomposition_source": group.decomposition_source,
    }


def assert_no_human_labels(payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=False)
    forbidden = ("human-atomic-facts", '"label"', "annotations")
    if any(token in encoded for token in forbidden):
        raise AssertionError("human labels or raw annotations leaked into online payload")

