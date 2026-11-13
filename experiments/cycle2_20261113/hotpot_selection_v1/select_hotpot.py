"""Private Hotpot selection preparation; default CLI is metadata-only preflight.

No dataset download, model execution, scoring, gold extraction, or scheduling.
Source byte hashing includes opaque parquet bytes, but label columns are never
projected, decoded, or returned. Revision provenance is receipt-bound, not
inferred from parquet bytes or independently re-fetched from Hugging Face.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

ALLOWLIST = ("id", "question", "context", "type", "level")
SOURCE_COLUMNS = set(ALLOWLIST) | {"answer", "supporting_facts"}
NORMALIZER = "nfkc_unicode_whitespace_casefold_v1"
CONTENT_HASH = "canonical_question_context_json_utf8_sha256_v1"
N = 384
EDGES = (1, 5, 7, 9)
SEEDS = {"generation_seed1": 26111301, "generation_seed2": 26111302,
         "sampling_pcg64": 261113101, "edge_assignment_pcg64": 261113102}


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _pinned_json(path, expected_sha: str) -> dict:
    if not _sha(expected_sha):
        raise ValueError("explicit lowercase SHA256 pin required")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("descriptor/receipt hash mismatch")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON descriptor keys forbidden")
            result[key] = value
        return result
    obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(obj, dict):
        raise ValueError("descriptor must be a JSON object")
    return obj


def question_key(question: str) -> str:
    """Group by text only: NFKC, Unicode whitespace collapse, then casefold."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return " ".join(unicodedata.normalize("NFKC", question).split()).casefold()


def question_group_sha256(question: str) -> str:
    return hashlib.sha256(question_key(question).encode("utf-8")).hexdigest()


def validate_row(row: dict) -> dict:
    if not isinstance(row, dict) or set(row) != set(ALLOWLIST):
        raise ValueError("row has missing/nonallowlisted columns; labels forbidden")
    for key in ("id", "question", "type", "level"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"invalid {key}")
    if row["id"] != row["id"].strip() or row["id"].startswith("hotpotqa_"):
        raise ValueError("raw official ID required, without collector prefix")
    context = row["context"]
    if not isinstance(context, dict) or set(context) != {"title", "sentences"}:
        raise ValueError("context must contain exactly title and sentences")
    titles, sentences = context["title"], context["sentences"]
    if not isinstance(titles, list) or not isinstance(sentences, list) or not titles:
        raise ValueError("context document arrays required")
    if len(titles) != len(sentences):
        raise ValueError("context document arrays differ in length")
    if any(not isinstance(t, str) or not t.strip() for t in titles) or len(set(titles)) != len(titles):
        raise ValueError("invalid or duplicate context titles")
    if any(not isinstance(s, list) or not all(isinstance(v, str) for v in s) for s in sentences):
        raise ValueError("context sentences must be nested string lists")
    # JSON roundtrip makes a private copy without normalizing/truncating text.
    return json.loads(canonical_bytes(row))


def content_sha256(row: dict) -> str:
    return _digest({"question": row["question"], "context": row["context"]})


def validate_exposures(obj: dict) -> dict:
    keys = {"schema_version", "id_namespace", "ids", "question_group_hashes",
            "question_normalizer", "content_hashes", "content_hash_algorithm",
            "known_records_scope", "unknown_cross_project_exposure"}
    if set(obj) != keys or obj["schema_version"] != "1.0":
        raise ValueError("exposure manifest schema mismatch")
    if obj["id_namespace"] != "hotpot_raw":
        raise ValueError("exposure IDs must be official raw IDs")
    if obj["question_normalizer"] != NORMALIZER or obj["content_hash_algorithm"] != CONTENT_HASH:
        raise ValueError("legacy/incompatible hash normalization cannot be silently reused")
    if obj["unknown_cross_project_exposure"] != "UNKNOWN":
        raise ValueError("unknown cross-project/pretraining exposure must remain UNKNOWN")
    if not isinstance(obj["known_records_scope"], str) or not obj["known_records_scope"].strip():
        raise ValueError("known-record coverage description required")
    for key in ("ids", "question_group_hashes", "content_hashes"):
        seq = obj[key]
        if not isinstance(seq, list) or len(seq) != len(set(seq)):
            raise ValueError("exposure entries must be unique lists")
        if key == "ids":
            if any(not isinstance(v, str) or not v or v != v.strip() or v.startswith("hotpotqa_") for v in seq):
                raise ValueError("invalid/prefixed exposure ID")
        elif any(not _sha(v) for v in seq):
            raise ValueError("invalid exposure hash")
    return obj


def select_records(rows, exposure_manifest: dict) -> tuple[list, list, dict]:
    """Pure input-to-selection core; production entry requires receipts first.

    This helper has no file IO. Tests pass fabricated allowlisted records only.
    Do not call it on real data outside run_selection's frozen receipt checks.
    """
    exp = validate_exposures(exposure_manifest)
    ids, qhashes, chashes = set(exp["ids"]), set(exp["question_group_hashes"]), set(exp["content_hashes"])
    seen = set()
    groups = {}
    exposed_groups = set()
    for raw in rows:
        row = validate_row(raw)
        if row["id"] in seen:
            raise ValueError("duplicate/conflicting ID in source; no silent deduplication")
        seen.add(row["id"])
        key = question_group_sha256(row["question"])
        groups.setdefault(key, []).append(row)
        if row["id"] in ids or key in qhashes or content_sha256(row) in chashes:
            exposed_groups.add(key)
    eligible = []
    for key, members in groups.items():
        if key not in exposed_groups:
            eligible.append(min(members, key=lambda r: r["id"].encode("utf-8")))
    eligible.sort(key=lambda r: r["id"].encode("utf-8"))
    if len(eligible) < N:
        raise ValueError("fewer than 384 eligible question groups; no replacement dataset")
    order = np.random.Generator(np.random.PCG64(SEEDS["sampling_pcg64"])).permutation(len(eligible))[:N]
    chosen = [eligible[int(i)] for i in order]
    edges = np.random.Generator(np.random.PCG64(SEEDS["edge_assignment_pcg64"])).permutation(np.repeat(EDGES, 96))
    manifest = [{"id": r["id"], "question_group_sha256": question_group_sha256(r["question"]),
                 "input_sha256": _digest(r), "content_sha256": content_sha256(r),
                 "edge_index": int(e), "generation_seed1": SEEDS["generation_seed1"],
                 "generation_seed2": SEEDS["generation_seed2"]} for r, e in zip(chosen, edges)]
    stats = {"source_rows": len(seen), "question_groups": len(groups),
             "duplicate_question_groups": sum(len(g) > 1 for g in groups.values()),
             "exposed_groups_removed": len(exposed_groups), "eligible_representatives": len(eligible),
             "selected_questions": N, "questions_per_edge": {str(e): int(np.count_nonzero(edges == e)) for e in EDGES},
             "exposure_ids_matched_source": len(ids & seen),
             "exposure_ids_not_in_this_train_source": len(ids - seen),
             "exposure_question_hashes_matched_source": len(qhashes & set(groups)),
             "unknown_cross_project_exposure": "UNKNOWN",
             "pretraining_contamination_guarantee": False}
    return chosen, manifest, stats


def _validate_protocol(protocol: dict, select: bool) -> None:
    if select and protocol.get("status") != "FROZEN":
        raise ValueError("real selection requires FROZEN scientific protocol")
    pop, seeds = protocol.get("population", {}), protocol.get("seeds", {})
    if pop.get("n_questions") != N or pop.get("edge_indices") != list(EDGES) or pop.get("n_questions_per_edge") != 96:
        raise ValueError("protocol differs from fixed 384/equal-four-edge contract")
    if any(seeds.get(k) != v for k, v in SEEDS.items()):
        raise ValueError("protocol random seeds differ from frozen selector")
    if protocol.get("scientific_scope", {}).get("task") != "HotpotQA distractor train":
        raise ValueError("wrong task/source scope")


def _source_preflight(source: dict, descriptor_path: Path) -> list[tuple[Path, dict]]:
    import pyarrow.parquet as pq
    required = {"schema_version", "status", "dataset_id", "revision", "config", "split",
                "source_audit_sha256", "read_columns", "parquet_files"}
    if set(source) != required or source["schema_version"] != "1.0":
        raise ValueError("source descriptor schema mismatch")
    if not isinstance(source["dataset_id"], str) or not source["dataset_id"]:
        raise ValueError("dataset ID required")
    if not isinstance(source["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None:
        raise ValueError("immutable 40-hex dataset revision required")
    if not _sha(source["source_audit_sha256"]):
        raise ValueError("frozen upstream source-audit pin required")
    if source["config"] != "distractor" or source["split"] != "train" or source["read_columns"] != list(ALLOWLIST):
        raise ValueError("only distractor/train allowlisted projection is permitted")
    files = source["parquet_files"]
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("exactly two pinned official train parquet files required")
    paths = []
    for item in files:
        if set(item) != {"path", "sha256", "num_rows"} or not _sha(item["sha256"]):
            raise ValueError("parquet descriptor schema/hash invalid")
        path = Path(item["path"])
        if not path.is_absolute():
            path = descriptor_path.parent / path
        path = path.resolve(strict=True)
        if path.suffix.lower() != ".parquet" or path in [x[0] for x in paths]:
            raise ValueError("duplicate/non-parquet source")
        if sha256_file(path) != item["sha256"]:
            raise ValueError("parquet byte SHA mismatch")
        pf = pq.ParquetFile(path)
        if set(pf.schema_arrow.names) != SOURCE_COLUMNS:
            raise ValueError("unexpected official source columns")
        if type(item["num_rows"]) is not int or item["num_rows"] <= 0 or pf.metadata.num_rows != item["num_rows"]:
            raise ValueError("parquet row count mismatch")
        paths.append((path, item))
    return paths


def iter_allowed_rows(paths):
    import pyarrow.parquet as pq
    for path, _ in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=list(ALLOWLIST)):
            if set(batch.schema.names) != set(ALLOWLIST):
                raise ValueError("projected batch contains forbidden/missing columns")
            yield from batch.to_pylist()


def _admission(obj: dict, pins: dict, output: Path) -> None:
    required = {"schema_version", "action", "decision", "source_descriptor_sha256",
                "protocol_sha256", "exposure_manifest_sha256", "selector_sha256",
                "output_directory", "data_classification", "gpu_authorized"}
    if set(obj) != required or obj["schema_version"] != "1.0":
        raise ValueError("admission schema mismatch")
    if obj["action"] != "SELECT_HOTPOT_APPLICABILITY_384" or obj["decision"] != "APPROVED":
        raise ValueError("private data selection admission not granted")
    if obj["gpu_authorized"] is not False or obj["data_classification"] != "PRIVATE_NO_LABELS":
        raise ValueError("wrong admission scope")
    for name in ("source_descriptor_sha256", "protocol_sha256", "exposure_manifest_sha256", "selector_sha256"):
        if obj[name] != pins[name]:
            raise ValueError("admission binding mismatch")
    if not Path(obj["output_directory"]).is_absolute() or Path(obj["output_directory"]).resolve() != output:
        raise ValueError("output directory differs from pinned private admission")


def run_selection(*, source_path, source_sha, expected_revision, protocol_path, protocol_sha,
                  exposures_path, exposures_sha, mode="preflight",
                  admission_path=None, admission_sha=None, output_directory=None) -> dict:
    if mode not in {"preflight", "select"}:
        raise ValueError("unsupported mode")
    select = mode == "select"
    source_path = Path(source_path)
    source = _pinned_json(source_path, source_sha)
    if not isinstance(expected_revision, str) or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise ValueError("explicit audited immutable revision pin required")
    if source.get("revision") != expected_revision:
        raise ValueError("dataset revision differs from independent expected pin")
    protocol = _pinned_json(protocol_path, protocol_sha)
    exposure = validate_exposures(_pinned_json(exposures_path, exposures_sha))
    _validate_protocol(protocol, select=select)
    pins = {"source_descriptor_sha256": source_sha, "protocol_sha256": protocol_sha,
            "exposure_manifest_sha256": exposures_sha, "selector_sha256": sha256_file(Path(__file__))}
    output = None
    if select:
        if source.get("status") != "FROZEN" or admission_path is None or admission_sha is None or output_directory is None:
            raise ValueError("frozen source and explicit pinned selection admission/output required")
        output = Path(output_directory).resolve()
        if output.exists():
            raise ValueError("new output directory required; no overwrite or silent resume")
        admission = _pinned_json(admission_path, admission_sha)
        _admission(admission, pins, output)
        pins["admission_sha256"] = admission_sha
    paths = _source_preflight(source, source_path)
    preflight = {"status": "PREFLIGHT_ONLY_NO_QUESTION_ROWS_READ", "pins": pins,
                 "revision_receipt_bound": source["revision"],
                 "revision_independently_verified_online": False,
                 "projected_columns": list(ALLOWLIST), "parquet_files_verified": len(paths),
                 "source_rows_metadata": sum(item["num_rows"] for _, item in paths),
                 "question_rows_decoded": False, "labels_decoded": False,
                 "GPU_admission": False}
    if not select:
        return preflight
    chosen, manifest_rows, stats = select_records(iter_allowed_rows(paths), exposure)
    for path, item in paths:
        if sha256_file(path) != item["sha256"]:
            raise ValueError("source changed during selection; no output written")
    # Recheck every input pin after source scanning, before publishing any output.
    for path, digest in ((source_path, source_sha), (protocol_path, protocol_sha),
                         (exposures_path, exposures_sha), (admission_path, admission_sha),
                         (Path(__file__), pins["selector_sha256"])):
        if sha256_file(Path(path)) != digest:
            raise ValueError("frozen input changed during selection")
    payload = b"".join(canonical_bytes(row) for row in chosen)
    manifest = {"schema_version": "1.0", "classification": "PRIVATE_NO_LABELS",
                "pins": pins, "normalizer": NORMALIZER, "content_hash_algorithm": CONTENT_HASH,
                "seeds": SEEDS, "questions": manifest_rows}
    manifest_bytes = canonical_bytes(manifest)
    receipt = {**preflight, "status": "COMPLETE_PRIVATE_SELECTION_NOT_GPU_ADMISSION",
               "question_rows_decoded": True, "stats": stats, "numpy_version": np.__version__,
               "question_normalizer": NORMALIZER, "unicode_database_version": unicodedata.unidata_version,
               "questions_jsonl_sha256": hashlib.sha256(payload).hexdigest(),
               "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "label_vault_created": False, "preserved_context_order_and_text": True}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (("questions.jsonl", payload), ("SELECTION_MANIFEST.json", manifest_bytes),
                          ("SELECTION_RECEIPT.json", canonical_bytes(receipt))):
        with (output / name).open("xb") as stream:
            stream.write(content)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "select"), default="preflight")
    for key in ("source", "protocol", "exposures"):
        parser.add_argument(f"--{key}", required=True)
        parser.add_argument(f"--{key}-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    result = run_selection(source_path=args.source, source_sha=args.source_sha256,
                           expected_revision=args.expected_revision,
                           protocol_path=args.protocol, protocol_sha=args.protocol_sha256,
                           exposures_path=args.exposures, exposures_sha=args.exposures_sha256,
                           mode=args.mode, admission_path=args.admission,
                           admission_sha=args.admission_sha256, output_directory=args.output_directory)
    # Report aggregate metadata only; never print selected IDs or question text.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
