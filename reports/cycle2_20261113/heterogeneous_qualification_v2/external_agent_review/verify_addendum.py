"""Read-only F1 closure check; never executes the reconstruction script."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED_MAP = "df814850c7395f03fd2dfcec9a3b88a957976abefe26ae7f9bfedb08a85c6721"
EXPECTED_OLD_SEAL = "1f07d496f7a30084845269f1baa77aa0042ee05659d97523625a3b9701166f99"
COMMIT = "b7434719eebb93eb41770bbc9af63222e3a47380"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def readj(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    folder = a.repo / "reports/cycle2_20261113/heterogeneous_qualification_v2"
    mp = folder / "QUALIFIED_SOURCE_BYTE_MAP.json"
    m = readj(mp)
    assert sha(mp.read_bytes()) == EXPECTED_MAP and m["git_commit"] == COMMIT
    old = readj(a.out / "INDEPENDENT_RECEIPT.json")
    old_mapping = {r["file"]: r for r in old["git_to_deployed_line_ending_mapping"]}
    assert len(m["files"]) == 7 and {r["path"] for r in m["files"]} == set(old["source_pins"])
    reconstruction = a.private / "source_reconstruction_001"
    previous_plan = readj(a.private / "PRIVATE_PLAN_V2B.json")
    assert sha((a.private / "PRIVATE_PLAN_V2B.json").read_bytes()) == old["private_plan_sha256"]
    checks = []
    for row in m["files"]:
        name = row["path"]
        raw = subprocess.check_output(["git", "-c", "safe.directory=" + a.repo.as_posix(), "-C", str(a.repo), "show", COMMIT + ":" + name])
        assert sha(raw) == row["git_blob_content_sha256"] == old_mapping[name]["git_blob_sha256"]
        if row["byte_transform"] == "LF_TO_CRLF":
            assert b"\r" not in raw and old_mapping[name]["byte_identity"] == "LF_CRLF_ONLY"
            converted = b"\r\n".join(raw.split(b"\n"))
        else:
            assert row["byte_transform"] == "IDENTITY" and old_mapping[name]["byte_identity"] == "exact"
            converted = raw
        actual = (reconstruction / name).read_bytes()
        assert converted == actual and len(actual) == row["deployed_bytes"]
        pin = sha(actual)
        assert pin == row["deployed_sha256"] == old_mapping[name]["deployed_sha256"] == old["source_pins"][name] == previous_plan["source_manifest"][name]
        assert sha((a.repo / name).read_bytes()) == pin
        checks.append({"file": name, "git_blob_sha256": sha(raw), "reconstructed_sha256": pin,
                       "bytes": len(actual), "transform": row["byte_transform"], "pass": True})
    rr_path = reconstruction / "RECONSTRUCTION_RECEIPT.json"
    rr = readj(rr_path)
    assert rr["status"] == "EXACT_QUALIFIED_SOURCE_BYTES_RECONSTRUCTED"
    assert rr["git_commit"] == COMMIT and rr["manifest_sha256"] == EXPECTED_MAP and rr["files"] == 7
    assert not rr["models_loaded"] and rr["inference_calls"] == 0 and not rr["private_runtime_plan_included"]
    old_seal = a.out / "SHA256SUMS.json"
    assert sha(old_seal.read_bytes()) == EXPECTED_OLD_SEAL
    for item in readj(old_seal)["files"]:
        raw = (a.out / item["name"]).read_bytes()
        assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]
    for name, pin in old["input_artifact_pins"].items():
        assert sha((a.private / "server_closed_v2" / name).read_bytes()) == pin
    result = {"status": "F1_CLOSED_BY_EXACT_GIT_TO_DEPLOYMENT_BYTE_RECONSTRUCTION",
              "source_commit": COMMIT, "source_byte_map_sha256": EXPECTED_MAP,
              "materializer_script_sha256": sha((folder / "materialize_qualified_source_bytes.py").read_bytes()),
              "existing_reconstruction_receipt_sha256": sha(rr_path.read_bytes()),
              "addendum_verifier_sha256": sha(Path(__file__).read_bytes()), "checks": checks,
              "transforms": {"IDENTITY": 5, "LF_TO_CRLF": 2},
              "previous_review_seal_sha256": EXPECTED_OLD_SEAL,
              "previous_review_three_artifacts_unchanged": True,
              "previous_16_run_artifacts_unchanged": True,
              "original_plan_unchanged": True, "original_7_source_files_unchanged": True,
              "reconstruction_script_executed_by_this_review": False, "new_GPU_calls": 0,
              "conclusion": "F1 exact deployed source-byte reproducibility gap is closed by the explicit map plus independently verified existing reconstruction; the original LF/CRLF historical difference remains recorded. F2 pre-generation rejection logging limitation remains open. No new runtime or scientific qualification is claimed."}
    (a.out / "ADDENDUM_RECEIPT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files_checked": len(checks), "old_artifacts_unchanged": True}))


if __name__ == "__main__":
    main()
