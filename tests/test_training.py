"""D6 uses only generated D4/D5 fixtures; no real images, labels or patients."""

import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cmvd_demo.cohort import assemble_cohort, save_cohort
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.io import ReadCancelled
from cmvd_demo.perfusion.storage import load_json
from cmvd_demo.roi import document_hash
from cmvd_demo.training import TrainingError, load_training_data, train_nested, train_snapshot, load_model_package, predict_features
from cmvd_demo.training.core import fit_outer, fit_pipeline, model_probabilities, validate_contract
from cmvd_demo.training.metrics import evaluate_oof
from cmvd_demo.training import storage


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_temp.cleanup)
        cls.base = Path(cls.fixture_temp.name)
        cls.project_path = create_synthetic_project(ROOT, cls.base/"fixtures", development_patients=12,
            protocol_version=2, reserve_independent=False, vary_features=True)
        cls.project = load_json(cls.project_path)
        cls.cohort = assemble_cohort(ROOT, cls.project)
        cls.snapshot = save_cohort(cls.base/"snapshots", cls.cohort, freeze=True)
        cls.feature = load_json(ROOT/"configs/cohort_contract.json")
        cls.contract = load_json(ROOT/"configs/training_contract.json")
        cls.contract["bootstrap_replicates"] = 5

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = load_training_data(self.snapshot, self.feature, allow_synthetic=True)

    def mutable_snapshot(self):
        target = self.root/"snapshot"
        shutil.copytree(self.snapshot, target)
        return target

    def rebind(self, run, *, audit=None, plan=None):
        if audit is not None:
            (run/"audit.json").write_text(json.dumps(audit), encoding="utf-8")
            plan = plan or load_json(run/"partitions.json")
            plan["audit_sha256"] = document_hash(audit)
        if plan is not None:
            (run/"partitions.json").write_text(json.dumps(plan), encoding="utf-8")
        marker = load_json(run/"bundle.json")
        marker["files_sha256"] = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in marker["files_sha256"]}
        (run/"bundle.json").write_text(json.dumps(marker), encoding="utf-8")

    def test_v2_development_only_freezes_without_inventing_independent_patients(self):
        report = self.cohort["report"]
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["freeze_ready"])
        self.assertEqual(report["population"]["independent_target_patients"], 40)
        self.assertEqual(report["reserved_independent_patients"], 0)
        self.assertFalse(report["population"]["independent_registered_to_target"])
        self.assertFalse(report["training_inputs_ready"])

    def test_population_registration_and_review_are_not_silently_assumed(self):
        p = copy.deepcopy(self.project)
        p["population_protocol"]["independent_registration_status"] = "registered"
        self.assertFalse(assemble_cohort(ROOT, p)["report"]["freeze_ready"])
        p["population_protocol"]["independent_registration_status"] = "pending"
        p["population_protocol"]["development_population_reviewed"] = False
        self.assertFalse(assemble_cohort(ROOT, p)["report"]["freeze_ready"])

    def test_future_independent_patient_keeps_all_measurements_out(self):
        path = create_synthetic_project(ROOT, self.root/"fixture", development_patients=12, protocol_version=2,
                                        reserve_independent=True, vary_features=True)
        c = assemble_cohort(ROOT, load_json(path))
        frozen = save_cohort(self.root/"cohort", c, freeze=True)
        data = load_training_data(frozen, self.feature, allow_synthetic=True)
        self.assertEqual(len(data.y), 12)
        self.assertNotIn("SYNTH-P013", data.patients)
        self.assertEqual(c["report"]["population"]["independent_registration_status"], "partial")

    def test_audit_or_fixture_without_opt_in_cannot_train(self):
        with self.assertRaisesRegex(TrainingError, "synthetic_explicit"):
            load_training_data(self.snapshot, self.feature)
        audit = save_cohort(self.root/"audit", self.cohort)
        with self.assertRaisesRegex(ValueError, "frozen_snapshot_required"):
            load_training_data(audit, self.feature, allow_synthetic=True)

    def test_nonready_research_gate_is_not_bypassed_by_snapshot_flag(self):
        run = self.mutable_snapshot()
        audit, plan, marker = (load_json(run/p) for p in ("audit.json", "partitions.json", "bundle.json"))
        audit["mode"], plan["mode"], marker["mode"] = "research", "research", "research"
        (run/"bundle.json").write_text(json.dumps(marker), encoding="utf-8")
        self.rebind(run, audit=audit, plan=plan)
        with self.assertRaisesRegex(TrainingError, "research_gate_blocked"):
            load_training_data(run, self.feature)

    def test_tampered_snapshot_is_rejected_before_fitting(self):
        run = self.mutable_snapshot()
        with (run/"features.csv").open("a", encoding="utf-8") as stream:
            stream.write("bad")
        with self.assertRaisesRegex(ValueError, "snapshot_tampered"):
            load_training_data(run, self.feature, allow_synthetic=True)

    def test_patient_crossing_or_duplicate_identity_is_rejected_even_with_new_hashes(self):
        run = self.mutable_snapshot()
        plan = load_json(run/"partitions.json")
        plan["patients"][1]["patient_key"] = plan["patients"][0]["patient_key"]
        self.rebind(run, plan=plan)
        with self.assertRaisesRegex(TrainingError, "duplicate_patient"):
            load_training_data(run, self.feature, allow_synthetic=True)

    def test_independent_row_in_features_is_rejected_even_with_new_hashes(self):
        run = self.mutable_snapshot()
        plan = load_json(run/"partitions.json")
        plan["patients"][0]["partition"] = "independent_test"
        self.rebind(run, plan=plan)
        with self.assertRaisesRegex(TrainingError, "independent_leakage"):
            load_training_data(run, self.feature, allow_synthetic=True)

    def test_csv_field_order_is_strict_and_missing_value_never_becomes_zero(self):
        run = self.mutable_snapshot()
        with (run/"features.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
        with (run/"features.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        self.rebind(run)
        with self.assertRaisesRegex(TrainingError, "columns_or_order"):
            load_training_data(run, self.feature, allow_synthetic=True)
        rows[0][1], rows[0][2] = rows[0][2], rows[0][1]
        rows[1][1] = ""
        with (run/"features.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(rows)
        self.rebind(run)
        with self.assertRaisesRegex(TrainingError, "nonfinite_or_missing"):
            load_training_data(run, self.feature, allow_synthetic=True)

    def test_fold_local_means_and_C_are_invariant_to_outer_holdout_changes(self):
        fold = self.data.folds[0]
        train = np.array([self.data.patients.index(p) for p in fold["train_patients"]])
        valid = np.array([self.data.patients.index(p) for p in fold["validation_patients"]])
        fields = self.feature["feature_order"]
        _, model_a, selection_a = fit_outer(self.data, train, valid, fields, self.contract)
        x, y = self.data.x.copy(), self.data.y.copy()
        x[valid, 0] = 110
        x[valid, -2:] = 999
        y[valid] = 1-y[valid]
        changed = replace(self.data, x=x, y=y)
        _, model_b, selection_b = fit_outer(changed, train, valid, fields, self.contract)
        self.assertEqual(selection_a, selection_b)
        self.assertEqual(model_a, model_b)
        for j, field in enumerate(fields):
            if field in self.contract["continuous_features"]:
                self.assertAlmostEqual(model_a["mean"][j], self.data.x[train, j].mean())
            else:
                self.assertEqual((model_a["mean"][j], model_a["scale"][j]), (0., 1.))
        for inner in selection_a["inner_folds"]:
            self.assertTrue(set(inner["train_patients"]) <= set(fold["train_patients"]))
            self.assertFalse(set(inner["train_patients"]) & set(inner["validation_patients"]))
            inner_indices = [self.data.patients.index(p) for p in inner["train_patients"]]
            for j, field in enumerate(fields):
                if field in self.contract["continuous_features"]:
                    self.assertAlmostEqual(inner["preprocessing_mean"][j], self.data.x[inner_indices, j].mean())

    def test_nested_oof_covers_each_patient_once_and_all_ablations_share_folds(self):
        result = train_nested(self.data, self.contract)
        patients = set(self.data.patients)
        for name in self.contract["variants"]:
            rows = [r for r in result["oof_predictions"] if r["variant"] == name]
            self.assertEqual(len(rows), len(patients))
            self.assertEqual({r["patient_key"] for r in rows}, patients)
            self.assertEqual([f["validation_patients"] for f in result["fold_audit"][name]["outer_folds"]],
                             [f["validation_patients"] for f in self.data.folds])
        fusion = result["models"]["fusion"]
        for j, field in enumerate(fusion["feature_order"]):
            if field in self.contract["continuous_features"]:
                self.assertAlmostEqual(fusion["mean"][j], self.data.x[:, j].mean())
        self.assertFalse(result["report"]["real_training_performed"])
        self.assertFalse(result["report"]["independent_evaluation_performed"])

    def test_tie_prefers_smallest_C_and_zero_variance_is_finite(self):
        data = replace(self.data, x=np.tile(self.data.x[0], (len(self.data.y), 1)))
        result = train_nested(data, self.contract)
        for model in result["models"].values():
            self.assertEqual(model["C"], .01)
            self.assertTrue(np.isfinite(model["coefficient"]).all())
            self.assertTrue(np.all(np.asarray(model["scale"]) == 1))

    def test_inner_class_shortage_and_patient_leakage_fail_before_any_fit(self):
        small_path = create_synthetic_project(ROOT, self.root/"small", protocol_version=2, reserve_independent=False)
        cohort = assemble_cohort(ROOT, load_json(small_path))
        run = save_cohort(self.root/"cohort", cohort, freeze=True)
        small = load_training_data(run, self.feature, allow_synthetic=True)
        with patch("cmvd_demo.training.core.fit_pipeline") as fit:
            with self.assertRaisesRegex(TrainingError, "inner_class_count_insufficient"):
                train_nested(small, self.contract)
            fit.assert_not_called()
        folds = copy.deepcopy(self.data.folds)
        folds[0]["train_patients"].append(folds[0]["validation_patients"][0])
        with self.assertRaisesRegex(TrainingError, "fold_leakage"):
            train_nested(replace(self.data, folds=folds), self.contract)

    def test_known_perfect_metrics_bootstrap_repeatable_and_threshold_inclusive(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])
        a, b = evaluate_oof(y, p, self.contract), evaluate_oof(y, p, self.contract)
        self.assertEqual(a, b)
        self.assertEqual(a["point_estimates"]["roc_auc"], 1.)
        self.assertEqual(a["point_estimates"]["average_precision"], 1.)
        self.assertEqual(a["point_estimates"]["confusion_matrix_tn_fp_fn_tp"], [2, 0, 0, 2])
        self.assertAlmostEqual(a["point_estimates"]["brier"], .025)
        _, model, _ = fit_outer(self.data, np.arange(12), np.array([], dtype=int), self.feature["feature_order"], self.contract)
        model["coefficient"], model["intercept"] = [0.]*8, 0.
        values = dict(zip(model["feature_order"], self.data.x[0]))
        prediction = predict_features(model, values, model["feature_units"],
            feature_contract_sha256=model["feature_contract_sha256"], allow_synthetic=True)
        self.assertEqual(prediction["probability_positive"], .5)
        self.assertEqual(prediction["cmvd_label"], 1)

    def test_repeat_runs_and_disk_reload_are_deterministic_but_never_overwrite(self):
        runs = [train_snapshot(ROOT, self.snapshot, self.root/"trained", allow_synthetic=True, contract=self.contract) for _ in range(2)]
        self.assertNotEqual(runs[0], runs[1])
        a, b = [load_model_package(run, allow_synthetic=True) for run in runs]
        self.assertEqual(a["models"], b["models"])
        self.assertEqual(a["report"], b["report"])
        model = a["models"]["fusion"]
        p = predict_features(model, dict(zip(model["feature_order"], self.data.x[0])), model["feature_units"],
            feature_contract_sha256=document_hash(self.feature), allow_synthetic=True)
        self.assertEqual(p["probability_positive"], model_probabilities(b["models"]["fusion"], self.data.x[:1])[0])
        self.assertEqual(len(list((self.root/"trained").glob("*/bundle.json"))), 2)

    def test_model_tamper_wrong_units_missing_fields_and_synthetic_default_rejected(self):
        run = train_snapshot(ROOT, self.snapshot, self.root/"trained", allow_synthetic=True, contract=self.contract)
        model = load_model_package(run, allow_synthetic=True)["models"]["fusion"]
        values, units, digest = dict(zip(model["feature_order"], self.data.x[0])), model["feature_units"], document_hash(self.feature)
        for args in ((values, units, "0"*64), (values, {**units, "pet_mfr": "BQML"}, digest), ({k:v for k,v in values.items() if k!="mr_mpri"}, units, digest)):
            with self.assertRaises(TrainingError):
                predict_features(model, args[0], args[1], feature_contract_sha256=args[2], allow_synthetic=True)
        with self.assertRaisesRegex(TrainingError, "synthetic_explicit"):
            load_model_package(run)
        with self.assertRaisesRegex(TrainingError, "synthetic_explicit"):
            predict_features(model, values, units, feature_contract_sha256=digest)
        with (run/"models.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(TrainingError, "package_tampered"):
            load_model_package(run, allow_synthetic=True)

    def test_convergence_failure_is_not_silently_accepted(self):
        def not_converged(*args, **kwargs):
            warnings.warn("synthetic convergence", ConvergenceWarning)
        with patch("cmvd_demo.training.core.Pipeline.fit", side_effect=not_converged):
            with self.assertRaisesRegex(TrainingError, "solver_not_converged"):
                fit_pipeline(self.data.x, self.data.y, self.feature["feature_order"], self.contract, .1)

    def test_cancel_during_fit_or_save_leaves_no_completion_marker(self):
        flag = [False]
        def progress(*args): flag[0] = True
        with self.assertRaises(ReadCancelled):
            train_snapshot(ROOT, self.snapshot, self.root/"cancel_fit", allow_synthetic=True, contract=self.contract,
                           progress=progress, cancel=lambda: flag[0])
        self.assertEqual(list((self.root/"cancel_fit").glob("*/bundle.json")), [])
        self.assertEqual(len(list((self.root/"cancel_fit").glob("*/failure.json"))), 1)
        real_write = storage.write_json
        def interrupted_save(path, value):
            real_write(path, value)
            if path.name == "models.json": raise OSError("SYNTHETIC-SAVE-FAILURE")
        with patch.object(storage, "write_json", side_effect=interrupted_save):
            with self.assertRaises(OSError):
                train_snapshot(ROOT, self.snapshot, self.root/"cancel_save", allow_synthetic=True, contract=self.contract)
        self.assertEqual(list((self.root/"cancel_save").glob("*/bundle.json")), [])

    def test_changed_source_before_publish_preserves_failed_attempt(self):
        run = self.mutable_snapshot()
        changed = [False]
        def progress(*args):
            if not changed[0]:
                with (run/"features.csv").open("a", encoding="utf-8") as stream: stream.write(" ")
                changed[0] = True
        with self.assertRaisesRegex(TrainingError, "source_changed"):
            train_snapshot(ROOT, run, self.root/"trained", allow_synthetic=True, contract=self.contract, progress=progress)
        self.assertEqual(list((self.root/"trained").glob("*/bundle.json")), [])

    def test_contract_cannot_switch_features_threshold_or_fold_policy_silently(self):
        contract = copy.deepcopy(self.contract)
        contract["threshold"] = .4
        with self.assertRaisesRegex(TrainingError, "threshold_policy"):
            validate_contract(contract, self.feature)
        contract = copy.deepcopy(self.contract)
        contract["random_seed"] += 1
        with self.assertRaisesRegex(TrainingError, "frozen_fold_policy"):
            train_nested(self.data, contract)


if __name__ == "__main__":
    unittest.main()
