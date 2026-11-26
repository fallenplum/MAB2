"""Strict fact/document verdict parsers for R1."""

from __future__ import annotations

import re


def parse_fact_verdict(text: str) -> int | None:
    normalized = re.sub(r"\s+", "", text.upper())
    if normalized in {"UNSUPPORTED", "NOTSUPPORTED"}:
        return 0
    if normalized in {"SUPPORT", "SUPPORTED"}:
        return 1
    return None


def parse_doc_verdict(text: str) -> str | None:
    normalized = re.sub(r"\s+", "", text.upper())
    if normalized == "PASS":
        return "PASS"
    if normalized == "FAIL":
        return "FAIL"
    return None

