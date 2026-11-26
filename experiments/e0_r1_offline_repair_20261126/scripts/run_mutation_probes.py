"""Small fail-closed mutation probes for the R1 certificate and parsers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r1_certificate import certificate
from src.r1_parsers import parse_doc_verdict, parse_fact_verdict


def main() -> None:
    base = [1] * 27 + [None] * 6
    cases = {
        "certificate_boundary_pass": certificate(base, 80).action == "PASS",
        "future_label_cannot_change_certified_pass": certificate(base[:27] + [0] + base[28:], 80).action == "PASS",
        "observed_witness_mutation_invalidates_pass": certificate([0] + base[1:], 80).status == "UNRESOLVED",
        "fact_parser_rejects_document_fragment": parse_fact_verdict("SUPPORT\nTO") is None,
        "doc_parser_rejects_fact_fragment": parse_doc_verdict("SUPPORT") is None,
        "doc_parser_accepts_exact_pass": parse_doc_verdict("PASS") == "PASS",
    }
    out = {"schema": "mab2_r1_mutation_probe_v1", "all_pass": all(cases.values()), "cases": cases, "status": "PASS" if all(cases.values()) else "FAIL"}
    (ROOT / "results/MUTATION_TESTS.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

