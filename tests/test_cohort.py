"""D5 tests use only synthetic patients; never load project data/ images."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.cohort import CohortError, assemble_cohort, save_cohort, export_templates, make_folds
from cmvd_demo.cohort.inputs import SCHEMAS, read_table
from cmvd_demo.cohort.perfusion import verified_ratios
from cmvd_demo.cohort.storage import load_frozen
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.io import ReadCancelled
from cmvd_demo.perfusion import load_json, mr_features, pair_mr, import_pet, build_bundle
from cmvd_demo.roi import document_hash


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = create_synthetic_project(ROOT, self.root/"fixtures")
        self.project = load_json(path)

    def rows(self, kind):
        return read_table(self.project["inputs"][kind], kind)[0]

    def put(self, kind, rows):
        with Path(self.project["inputs"][kind]).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEMAS[kind])
            writer.writeheader()
            writer.writerows(rows)

    def assemble(self):
        return assemble_cohort(ROOT, self.project)

    def test_absent_real_resources_create_zero_cases_and_report_every_missing_input(self):
        p = load_json(ROOT/"configs/cohort_project.template.json")
        p["inputs"] = {kind: str(self.root/(kind+".csv")) for kind in SCHEMAS}
        result = assemble_cohort(ROOT, p)
        self.assertEqual(result["cases"], [])
        self.assertEqual(result["samples"], [])
        self.assertTrue(all("input_missing:"+kind in result["report"]["issues"] for kind in SCHEMAS))
        self.assertFalse(result["report"]["freeze_ready"])
        self.assertFalse(result["report"]["training_inputs_ready"])

    def test_complete_fixture_seven_patients_eight_exams_six_development_rows(self):
        a = self.assemble()
        report = a["report"]
        self.assertEqual((report["manifest_case_count"], report["associated_patient_count"], report["development_included_count"]), (8, 7, 6))
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["freeze_ready"])
        self.assertTrue(report["development_matrix_ready"])
        self.assertFalse(report["training_inputs_ready"])
        self.assertFalse(report["training_prerequisites_ready"])
        for sample in a["samples"]:
            self.assertAlmostEqual(sample["features"]["pet_mfr"], 3.)
            self.assertAlmostEqual(sample["features"]["mr_mpri"], 2.)
        self.assertEqual(a["cases"][-1]["reasons"], ["non_index_exam_or_ambiguous_selection"])

    def test_labels_and_identifiers_never_enter_feature_columns(self):
        run = save_cohort(self.root/"out", self.assemble())
        with (run/"features.csv").open(encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["research_id"]+load_json(ROOT/"configs/cohort_contract.json")["feature_order"])
            self.assertEqual(len(list(reader)), 6)
        self.assertTrue((run/"labels.csv").exists())
        self.assertNotIn("cmvd_label", (run/"features.csv").read_text(encoding="utf-8-sig"))
        self.assertFalse(load_json(run/"audit.json")["training_inputs_ready"])

    def test_patient_folds_disjoint_balanced_repeatable_and_independent_excluded(self):
        a = self.assemble()
        result = make_folds(list(reversed(a["samples"])))
        self.assertEqual(result, a["report"]["folds"])
        validations = []
        for f in result["folds"]:
            train, valid = set(f["train_patients"]), set(f["validation_patients"])
            self.assertFalse(train & valid)
            self.assertEqual(len(train | valid), 6)
            self.assertNotIn("SYNTH-P007", train | valid)
            labels = {s["label"] for s in a["samples"] if s["patient_key"] in valid}
            self.assertEqual(labels, {0, 1})
            validations.extend(valid)
        self.assertEqual(len(set(validations)), len(validations))

    def test_small_minority_blocks_folds_without_silently_changing_number(self):
        a = self.assemble()
        self.assertFalse(make_folds(a["samples"][:4], 3)["ready"])
        with self.assertRaises(CohortError):
            make_folds(a["samples"]+[a["samples"][0]])
        bad = copy.deepcopy(a["samples"])
        bad[0]["partition"] = "independent_test"
        with self.assertRaises(CohortError):
            make_folds(bad)

    def test_independent_label_values_and_missing_feature_paths_are_never_evaluated(self):
        labels = self.rows("labels")
        labels.append({**labels[0], "patient_id": "SYNTH-E007", "cmvd_label": "DO-NOT-READ", "source_reference": "DO-NOT-EXPORT"})
        self.put("labels", labels)
        registry = self.rows("perfusion")
        registry.append({**registry[0], "research_id": "SYNTH-E007", "features_path": str(self.root/"must-not-open.json")})
        self.put("perfusion", registry)
        with patch("cmvd_demo.cohort.core.verified_ratios", wraps=verified_ratios) as verify:
            a = self.assemble()
            self.assertEqual(verify.call_count, 6)
            self.assertNotIn(str(self.root/"must-not-open.json"), repr(verify.call_args_list))
        run = save_cohort(self.root/"out", a)
        text = "".join(p.read_text(encoding="utf-8-sig") for p in run.iterdir())
        self.assertNotIn("DO-NOT-READ", text)
        self.assertNotIn("DO-NOT-EXPORT", text)
        self.assertEqual(a["cases"][6]["reasons"], ["reserved_independent_not_loaded"])

    def test_all_exams_of_previously_used_patient_prevent_independent_use(self):
        rows = self.rows("partitions")
        rows[0]["partition"] = "independent_test"
        self.put("partitions", rows)
        a = self.assemble()
        self.assertIn("independent_patient_used_or_unknown_development", a["report"]["issues"])
        self.assertFalse(a["report"]["freeze_ready"])
        self.assertNotIn("SYNTH-P001", {s["patient_key"] for s in a["samples"]})

    def test_no_implicit_selection_when_two_visits_are_marked_index(self):
        rows = self.rows("associations")
        rows[-1]["index_exam"] = "1"
        self.put("associations", rows)
        a = self.assemble()
        self.assertIn("index_exam_not_unique", a["report"]["issues"])
        self.assertFalse(a["report"]["freeze_ready"])

    def test_missing_and_nonfinite_clinical_values_are_excluded_never_zero_imputed(self):
        rows = self.rows("clinical")
        rows[0]["age_years"], rows[1]["ldl_c_mmol_l"], rows[2]["hypertension"] = "", "nan", "2"
        self.put("clinical", rows)
        self.put("missingness", [{"patient_id": "SYNTH-E001", "field": "age_years", "reason": "not_collected"}])
        a = self.assemble()
        self.assertEqual(len(a["samples"]), 3)
        self.assertEqual(a["missingness"], [{"research_id": "SYNTH-E001", "field": "age_years", "reason": "not_collected"}])
        for index in range(3):
            self.assertFalse(a["cases"][index]["included"])

    def test_legacy_directory_labels_and_demographics_are_discarded(self):
        path = Path(self.project["inputs"]["manifest"])
        rows = self.rows("manifest")
        fields = list(SCHEMAS["manifest"])+["CohortLabel", "Sex", "AgeDICOM", "PrivateName"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{**r, "CohortLabel": "POS", "Sex": "F", "AgeDICOM": "099Y", "PrivateName": "PRIVATE_SENTINEL"} for r in rows])
        labels = self.rows("labels")
        labels[0]["cmvd_label"] = ""
        self.put("labels", labels)
        a = self.assemble()
        self.assertNotIn("PRIVATE_SENTINEL", repr(a))
        self.assertIn("physician_label_missing_invalid_or_unreviewed", a["cases"][0]["reasons"])

    def test_wrong_label_source_definition_and_missing_review_block_rows(self):
        for field, value in (("label_source", "directory_prefix"), ("label_definition_version", "different"), ("reviewer_code", "")):
            original = self.rows("labels")
            modified = copy.deepcopy(original)
            modified[0][field] = value
            self.put("labels", modified)
            self.assertFalse(self.assemble()["cases"][0]["included"])
            self.put("labels", original)

    def test_processing_errors_are_preserved_as_exclusions(self):
        rows = self.rows("manifest")
        rows[0]["ProcessingErrors"] = "167"
        self.put("manifest", rows)
        self.assertIn("manifest_processing_failed_or_empty", self.assemble()["cases"][0]["reasons"])

    def test_duplicate_and_unmatched_ids_are_reported_without_reindexing(self):
        rows = self.rows("labels")
        self.put("labels", rows+[rows[0]])
        a = self.assemble()
        self.assertIn("input_invalid:labels:cohort_duplicate_association_key", a["report"]["issues"])
        self.assertEqual(a["cases"][0]["research_id"], "SYNTH-E001")
        self.put("labels", rows+[{**rows[0], "patient_id": "SYNTH-UNKNOWN"}])
        self.assertIn("unmatched_research_id:labels", self.assemble()["report"]["issues"])

    def test_count_mismatch_and_unreviewed_identity_prevent_freeze(self):
        self.project["policy"]["reported_case_count"] = 80
        a = self.assemble()
        self.assertIn("reported_manifest_count_mismatch", a["report"]["issues"])
        self.assertFalse(a["report"]["freeze_ready"])
        with self.assertRaises(CohortError):
            save_cohort(self.root/"out", a, freeze=True)

    def test_missing_clinical_review_blocks_matrix_but_not_patient_reservation(self):
        self.project["policy"]["clinical_schema_reviewed"] = False
        a = self.assemble()
        self.assertTrue(a["report"]["freeze_ready"])
        self.assertEqual(a["samples"], [])

    def test_true_mode_rejects_synthetic_identifiers(self):
        self.project["mode"] = "research"
        a = self.assemble()
        self.assertEqual(a["samples"], [])
        self.assertFalse(a["report"]["training_inputs_ready"])

    def test_research_contract_requires_freeze_before_training_readiness(self):
        # Handcrafted numerical contract fixture only. This test exercises the
        # positive research schema; these are not real annotations or diagnoses.
        self.project["mode"] = "research"
        for kind in SCHEMAS:
            rows = self.rows(kind)
            for row in rows:
                for key in ("ResearchID", "research_id", "patient_id", "patient_key"):
                    if key in row:
                        row[key] = row[key].replace("SYNTH-", "UNIT-")
            self.put(kind, rows)
        registry = self.rows("perfusion")
        for entry in registry:
            path = Path(entry["features_path"])
            original = load_json(path)
            case = original["case_token"]
            def reviewed_input(doc, curves):
                doc, curves = copy.deepcopy(doc), copy.deepcopy(curves)
                doc["purpose"] = "research_draft"
                curves["document_sha256"] = document_hash(doc)
                curves["qc"]["warnings"], curves["qc"]["feature_input_ready"] = [], True
                return doc, curves
            branches = []
            for condition in ("rest", "stress"):
                item = original["mr"][condition]
                doc, curves = reviewed_input(item["roi_document"], item["input_curves"])
                branches.append(mr_features(doc, curves, case, condition, "UNIT-PROTOCOL", baseline_frames=2,
                                             window_start=2, window_end=6, window_reviewed=True))
            mr = pair_mr(*branches, pairing_reviewed=True, protocol_comparable=True)
            local = {key: reviewed_input(item["roi_document"], item["input_curves"])
                     for key, item in original["pet"]["local_tacs"].items()}
            request = copy.deepcopy(original["pet"]["workstation_result"])
            from cmvd_demo.perfusion import pet_request
            fresh = pet_request(case, local)
            for condition in ("rest", "stress"):
                request["branches"][condition]["binding"] = fresh["branches"][condition]["binding"]
            ref = copy.deepcopy(original["mr_reference_check"]["reference"])
            ref["feature_sha256"] = document_hash(mr)
            bundle = build_bundle(case, mr=mr, pet=import_pet(request, case, local), mr_reference=ref)
            self.assertTrue(bundle["qc"]["perfusion_input_ready"])
            path.write_text(json.dumps(bundle), encoding="utf-8")
            entry["features_sha256"] = document_hash(bundle)
            marker = load_json(path.with_name("bundle.json"))
            marker["features_sha256"] = entry["features_sha256"]
            path.with_name("bundle.json").write_text(json.dumps(marker), encoding="utf-8")
        self.put("perfusion", registry)
        a = self.assemble()
        self.assertEqual(a["report"]["issues"], [])
        self.assertTrue(a["report"]["training_prerequisites_ready"])
        self.assertFalse(a["report"]["training_inputs_ready"])
        audit = save_cohort(self.root/"out", a)
        self.assertFalse(load_json(audit/"audit.json")["training_inputs_ready"])
        frozen = save_cohort(self.root/"out", a, freeze=True)
        self.assertTrue(load_json(frozen/"audit.json")["training_inputs_ready"])
        self.assertFalse(load_json(frozen/"audit.json")["model_trained"])

    def test_fixture_mode_requires_synthetic_identifiers(self):
        rows = self.rows("associations")
        rows[0]["patient_key"] = "NOT-SYNTHETIC"
        self.put("associations", rows)
        self.assertIn("fixture_requires_synthetic_identifiers", self.assemble()["report"]["issues"])

    def test_snapshot_hash_recomputed_measurement_and_qc_tampering_are_rejected(self):
        registry = self.rows("perfusion")[0]
        path = Path(registry["features_path"])
        original = load_json(path)
        token = self.rows("associations")[0]["case_token"]
        with self.assertRaises(CohortError):
            verified_ratios(path, "0"*64, token, mode="synthetic_fixture")
        modified = copy.deepcopy(original)
        modified["mr"]["measurements"]["mr_mpri"]["value"] = 999.
        path.write_text(json.dumps(modified), encoding="utf-8")
        marker = load_json(path.with_name("bundle.json"))
        marker["features_sha256"] = document_hash(modified)
        path.with_name("bundle.json").write_text(json.dumps(marker), encoding="utf-8")
        with self.assertRaisesRegex(CohortError, "recomputation_mismatch"):
            verified_ratios(path, document_hash(modified), token, mode="synthetic_fixture")

    def test_missing_branch_unit_or_reference_never_becomes_a_normal_ratio(self):
        rows = self.rows("perfusion")
        path = Path(rows[0]["features_path"])
        bundle = load_json(path)
        bundle["pet"] = None
        path.write_text(json.dumps(bundle), encoding="utf-8")
        rows[0]["features_sha256"] = document_hash(bundle)
        marker = load_json(path.with_name("bundle.json"))
        marker["features_sha256"] = document_hash(bundle)
        path.with_name("bundle.json").write_text(json.dumps(marker), encoding="utf-8")
        self.put("perfusion", rows)
        a = self.assemble()
        self.assertIn("cohort_perfusion_branch_missing", a["cases"][0]["reasons"])
        self.assertEqual(len(a["samples"]), 5)

    def test_engineering_fixture_can_never_pass_research_feature_gate(self):
        row = self.rows("perfusion")[0]
        token = self.rows("associations")[0]["case_token"]
        with self.assertRaisesRegex(CohortError, "perfusion_qc_pending"):
            verified_ratios(row["features_path"], row["features_sha256"], token, mode="research")

    def test_source_edit_after_check_requires_recheck_before_export(self):
        a = self.assemble()
        path = Path(self.project["inputs"]["clinical"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(CohortError, "source_changed"):
            save_cohort(self.root/"out", a, freeze=True)
        self.assertFalse((self.root/"out").exists())

    def test_audit_and_freeze_are_distinct_append_only_and_tampering_detected(self):
        a = self.assemble()
        audit = save_cohort(self.root/"out", a)
        self.assertTrue((audit/"partition_plan.json").exists())
        with self.assertRaises(CohortError):
            load_frozen(audit)
        frozen = save_cohort(self.root/"out", a, freeze=True)
        again = save_cohort(self.root/"out", a, freeze=True)
        self.assertNotEqual(frozen, again)
        plan = load_frozen(frozen)
        self.assertEqual(len(plan["patients"]), 7)
        self.assertFalse(load_json(frozen/"audit.json")["training_inputs_ready"])
        with (frozen/"features.csv").open("a") as stream:
            stream.write("tampered")
        with self.assertRaisesRegex(CohortError, "tampered"):
            load_frozen(frozen)

    def test_cooperative_cancel_during_read_and_during_publish_never_creates_marker(self):
        with self.assertRaises(ReadCancelled):
            assemble_cohort(ROOT, self.project, cancel=lambda: True)
        a = self.assemble()
        def interrupted(path, value):
            from cmvd_demo.perfusion.storage import write_json
            write_json(path, value)
            cancelled[0] = True
        cancelled = [False]
        with patch("cmvd_demo.cohort.storage.write_json", side_effect=interrupted), self.assertRaises(ReadCancelled):
            save_cohort(self.root/"out", a, freeze=True, cancel=lambda: cancelled[0])
        runs = list((self.root/"out").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertTrue((runs[0]/"audit.json").exists())
        self.assertFalse((runs[0]/"bundle.json").exists())

    def test_blank_forms_do_not_create_a_manifest_or_patient_rows(self):
        run = export_templates(ROOT, self.root/"requests")
        p = load_json(run/"cohort_project.json")
        self.assertFalse((run/"manifest.csv").exists())
        for kind in SCHEMAS:
            if kind != "manifest":
                self.assertEqual(read_table(p["inputs"][kind], kind)[0], [])
        self.assertEqual(p["mode"], "research")
        self.assertEqual(p["policy"]["clinical_schema_reviewed"], False)

    def test_csv_duplicate_headers_nonfinite_json_and_private_error_text_are_not_echoed(self):
        path = Path(self.project["inputs"]["labels"])
        path.write_text("patient_id,patient_id\nPRIVATE_SENTINEL,X", encoding="utf-8")
        a = self.assemble()
        self.assertNotIn("PRIVATE_SENTINEL", repr(a))
        self.assertIn("input_invalid:labels:cohort_csv_duplicate_or_missing_header", a["report"]["issues"])


if __name__ == "__main__":
    unittest.main()
