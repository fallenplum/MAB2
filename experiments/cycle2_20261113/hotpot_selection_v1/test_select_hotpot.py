"""Synthetic records/parquet only; never opens benchmark files or real labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

import select_hotpot as s


def row(i):
    return {"id": f"id{i:04d}", "question": f"Synthetic question number {i}?",
            "context": {"title": ["Second", "First"],
                        "sentences": [["  A sentence.  ", "Another\nline."], ["尾句。"]]},
            "type": "bridge", "level": "hard"}


def exposures():
    return {"schema_version": "1.0", "id_namespace": "hotpot_raw", "ids": [],
            "question_group_hashes": [], "question_normalizer": s.NORMALIZER,
            "content_hashes": [], "content_hash_algorithm": s.CONTENT_HASH,
            "known_records_scope": "SYNTHETIC_FIXTURE_ONLY", "unknown_cross_project_exposure": "UNKNOWN"}


def write_json(path, obj):
    path.write_bytes(s.canonical_bytes(obj))
    return s.sha256_file(path)


class PureTests(unittest.TestCase):
    def test_nfkc_whitespace_casefold(self):
        self.assertEqual(s.question_key("  Ａ\u00a0Straße\n？ "), "a strasse ?")
        self.assertEqual(s.question_group_sha256("Ａ  Straße？"),
                         s.question_group_sha256("a strasse?"))

    def test_exposed_nonrepresentative_excludes_whole_group(self):
        fillers = [row(i) for i in range(384)]
        a, z = row(9000), row(9001)
        a["id"], z["id"] = "a-min", "z-exposed"
        a["question"], z["question"] = "Duplicate  question?", " ＤＵＰＬＩＣＡＴＥ\nQUESTION? "
        exp = exposures(); exp["ids"] = [z["id"]]
        chosen, _, stats = s.select_records(fillers + [a, z], exp)
        self.assertEqual({x["id"] for x in chosen}, {x["id"] for x in fillers})
        self.assertEqual(stats["exposed_groups_removed"], 1)
        self.assertEqual(stats["duplicate_question_groups"], 1)
        exp = exposures(); exp["question_group_hashes"] = [s.question_group_sha256(a["question"])]
        self.assertEqual(s.select_records(fillers + [a, z], exp)[2]["eligible_representatives"], 384)
        exp = exposures(); exp["content_hashes"] = [s.content_sha256(z)]
        self.assertEqual(s.select_records(fillers + [a, z], exp)[2]["eligible_representatives"], 384)

    def test_representative_preserves_original_context(self):
        a, z = row(9000), row(9001)
        a["id"], z["id"] = "a-min", "z-other"
        a["question"], z["question"] = "Dup?", "ＤＵＰ?"
        z["context"]["title"] = ["New first", "New second"]
        chosen, _, _ = s.select_records([row(i) for i in range(383)] + [z, a], exposures())
        got = {r["id"]: r for r in chosen}
        self.assertEqual(got["a-min"], a)
        self.assertNotIn("z-other", got)

    def test_order_independence_seed_determinism_and_balanced_edges(self):
        rows = [row(i) for i in range(410)]
        left = s.select_records(rows, exposures())
        right = s.select_records(reversed(rows), exposures())
        self.assertEqual(left, right)
        self.assertEqual(left[2]["questions_per_edge"], {"1": 96, "5": 96, "7": 96, "9": 96})
        self.assertTrue(all(set(m) == {"id", "question_group_sha256", "input_sha256", "content_sha256",
                                      "edge_index", "generation_seed1", "generation_seed2"} for m in left[1]))

    def test_duplicate_id_and_nonallowlisted_columns_rejected(self):
        rows = [row(i) for i in range(384)]
        for extra in (copy.deepcopy(rows[0]), dict(rows[0], question="conflicting")):
            with self.assertRaises(ValueError):
                s.select_records(rows + [extra], exposures())
        with self.assertRaises(ValueError):
            s.validate_row(dict(row(0), answer="SYNTHETIC_LABEL"))
        bad = row(0); bad["context"]["answer"] = "forbidden nested field"
        with self.assertRaises(ValueError):
            s.validate_row(bad)
        bad = row(0); bad["context"]["sentences"].pop()
        with self.assertRaises(ValueError):
            s.validate_row(bad)

    def test_legacy_hash_namespace_and_insufficient_groups_rejected(self):
        exp = exposures(); exp["question_normalizer"] = "lower_whitespace_old"
        with self.assertRaises(ValueError):
            s.select_records([row(i) for i in range(384)], exp)
        with self.assertRaises(ValueError):
            s.select_records([row(i) for i in range(383)], exposures())


class ParquetAndAdmissionTests(unittest.TestCase):
    def setUp(self):
        # This directory contains only fixtures made by this test. Validate the
        # resolved parent before TemporaryDirectory is ever allowed to clean it.
        self.stage = Path(__file__).resolve().parent
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic_selection_", dir=self.stage)
        self.base = Path(self.tmp.name).resolve()
        self.assertEqual(self.base.parent, self.stage)
        self.rows = [row(i) for i in range(390)]
        files = []
        for index, subset in enumerate((self.rows[:195], self.rows[195:])):
            target = self.base / f"train-{index:05d}-of-00002.parquet"
            # Fake labels exist solely to test projection; no real gold is used.
            full = [dict(r, answer="SYNTHETIC_DO_NOT_PROJECT",
                         supporting_facts={"title": ["Second"], "sent_id": [0]}) for r in subset]
            pq.write_table(pa.Table.from_pylist(full), target)
            files.append({"path": target.name, "sha256": s.sha256_file(target), "num_rows": 195})
        self.source = {"schema_version": "1.0", "status": "FROZEN", "dataset_id": "synthetic/hotpot",
                       "revision": "1"*40, "config": "distractor", "split": "train",
                       "source_audit_sha256": "a"*64, "read_columns": list(s.ALLOWLIST), "parquet_files": files}
        self.protocol = {"status": "FROZEN", "population": {"n_questions": 384, "edge_indices": [1,5,7,9],
                         "n_questions_per_edge": 96}, "seeds": dict(s.SEEDS),
                         "scientific_scope": {"task": "HotpotQA distractor train"}}
        self.source_path, self.proto_path, self.exp_path = [self.base/x for x in ("source.json", "protocol.json", "exposures.json")]
        self.kw = dict(source_path=self.source_path, source_sha=write_json(self.source_path, self.source),
                       expected_revision="1"*40, protocol_path=self.proto_path,
                       protocol_sha=write_json(self.proto_path, self.protocol), exposures_path=self.exp_path,
                       exposures_sha=write_json(self.exp_path, exposures()))
        self.output = self.base / "private_output"

    def tearDown(self):
        self.assertEqual(self.base.resolve().parent, self.stage)
        self.assertTrue(self.base.name.startswith("synthetic_selection_"))
        self.tmp.cleanup()

    def admission(self):
        obj = {"schema_version": "1.0", "action": "SELECT_HOTPOT_APPLICABILITY_384", "decision": "APPROVED",
               "source_descriptor_sha256": self.kw["source_sha"], "protocol_sha256": self.kw["protocol_sha"],
               "exposure_manifest_sha256": self.kw["exposures_sha"],
               "selector_sha256": s.sha256_file(Path(s.__file__)), "output_directory": str(self.output),
               "data_classification": "PRIVATE_NO_LABELS", "gpu_authorized": False}
        path = self.base / "admission.json"
        return dict(mode="select", admission_path=path, admission_sha=write_json(path, obj), output_directory=self.output)

    def test_default_preflight_does_not_decode_rows_or_create_output(self):
        with mock.patch.object(s, "iter_allowed_rows", side_effect=AssertionError("must not decode rows")):
            result = s.run_selection(**self.kw)
        self.assertEqual(result["status"], "PREFLIGHT_ONLY_NO_QUESTION_ROWS_READ")
        self.assertFalse(result["question_rows_decoded"])
        self.assertFalse(self.output.exists())

    def test_actual_two_parquet_projection_and_private_outputs(self):
        real = pq.ParquetFile
        projected = []
        class ProjectionGuard:
            def __init__(inner, path):
                inner.obj = real(path)
                inner.schema_arrow, inner.metadata = inner.obj.schema_arrow, inner.obj.metadata
            def iter_batches(inner, **kwargs):
                if kwargs["columns"] != list(s.ALLOWLIST):
                    raise AssertionError("forbidden source columns requested")
                projected.append(tuple(kwargs["columns"]))
                return inner.obj.iter_batches(**kwargs)
        with mock.patch.object(pq, "ParquetFile", ProjectionGuard):
            result = s.run_selection(**self.kw, **self.admission())
        self.assertEqual(len(projected), 2)
        self.assertFalse(result["labels_decoded"])
        self.assertFalse(result["label_vault_created"])
        selected = [json.loads(x) for x in (self.output/"questions.jsonl").read_text(encoding="utf-8").splitlines()]
        original = {r["id"]: r for r in self.rows}
        self.assertEqual(len(selected), 384)
        self.assertTrue(all(set(r) == set(s.ALLOWLIST) and r == original[r["id"]] for r in selected))
        manifest = json.loads((self.output/"SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual([m["id"] for m in manifest["questions"]], [r["id"] for r in selected])
        self.assertEqual(s.sha256_file(self.output/"questions.jsonl"), result["questions_jsonl_sha256"])
        before = s.sha256_file(self.output/"SELECTION_RECEIPT.json")
        with self.assertRaises(ValueError):
            s.run_selection(**self.kw, **self.admission())
        self.assertEqual(before, s.sha256_file(self.output/"SELECTION_RECEIPT.json"))

    def test_source_sha_revision_and_projection_policy_tampering_rejected(self):
        bad = dict(self.kw, expected_revision="2"*40)
        with self.assertRaises(ValueError): s.run_selection(**bad)
        source = copy.deepcopy(self.source); source["read_columns"].append("answer")
        bad = dict(self.kw, source_sha=write_json(self.source_path, source))
        with self.assertRaises(ValueError): s.run_selection(**bad)
        self.kw["source_sha"] = write_json(self.source_path, self.source)
        target = self.base/self.source["parquet_files"][0]["path"]
        with target.open("ab") as stream: stream.write(b"CHANGED")
        with self.assertRaises(ValueError): s.run_selection(**self.kw)

    def test_missing_candidate_or_wrong_admission_cannot_select(self):
        with self.assertRaises(ValueError):
            s.run_selection(**self.kw, mode="select", output_directory=self.output)
        ad = self.admission()
        obj = json.loads(ad["admission_path"].read_text(encoding="utf-8")); obj["protocol_sha256"] = "f"*64
        ad["admission_sha"] = write_json(ad["admission_path"], obj)
        with self.assertRaises(ValueError): s.run_selection(**self.kw, **ad)
        self.protocol["status"] = "CANDIDATE_NOT_FROZEN_NOT_ADMITTED"
        self.kw["protocol_sha"] = write_json(self.proto_path, self.protocol)
        with self.assertRaises(ValueError): s.run_selection(**self.kw, **self.admission())
        self.assertFalse(self.output.exists())

    def test_unexpected_source_column_rejected_even_with_matching_byte_hash(self):
        target = self.base/self.source["parquet_files"][0]["path"]
        full = [dict(r, answer="SYNTHETIC_LABEL", supporting_facts={"title": [], "sent_id": []},
                     unexpected="do not silently accept schema drift") for r in self.rows[:195]]
        pq.write_table(pa.Table.from_pylist(full), target)
        self.source["parquet_files"][0]["sha256"] = s.sha256_file(target)
        self.kw["source_sha"] = write_json(self.source_path, self.source)
        with self.assertRaises(ValueError): s.run_selection(**self.kw)

    def test_duplicate_descriptor_keys_rejected(self):
        path = self.base/"ambiguous.json"
        path.write_text('{"status":"FROZEN","status":"OTHER"}', encoding="utf-8")
        with self.assertRaises(ValueError): s._pinned_json(path, s.sha256_file(path))


if __name__ == "__main__":
    unittest.main()
