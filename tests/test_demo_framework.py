"""Safety and data-integrity boundaries, using only synthetic inputs."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cmvd_demo.framework import ContractError, build_features, predict_fixture, validate_partitions


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((ROOT / "examples/synthetic/case.json").read_text(encoding="utf-8"))
        self.contract = json.loads((ROOT / "configs/demo_contract.json").read_text(encoding="utf-8"))
        self.model = json.loads((ROOT / "examples/synthetic/model_fixture.json").read_text(encoding="utf-8"))

    def test_simulation_is_explicit_and_ratios_are_derived(self):
        result = predict_fixture(self.case, self.contract, self.model)
        self.assertFalse(result["model_trained"])
        self.assertFalse(result["clinical_validation"])
        self.assertAlmostEqual(result["features"]["pet_mfr"], 1.7)
        self.assertAlmostEqual(result["features"]["mr_mpri"], 1.6)
        self.assertGreater(result["simulated_positive_probability"], 0)
        self.assertLess(result["simulated_positive_probability"], 1)

    def test_each_required_branch_is_checked(self):
        for group in self.contract["groups"]:
            case = deepcopy(self.case)
            del case[group]
            with self.subTest(group=group), self.assertRaises(ContractError):
                build_features(case, self.contract)

    def test_failed_qc_is_not_a_negative_diagnosis(self):
        for branch in ("pet_perfusion", "mr_perfusion"):
            case = deepcopy(self.case)
            case[branch]["provenance"]["qc_status"] = "failed"
            with self.assertRaises(ContractError):
                predict_fixture(case, self.contract, self.model)

    def test_numeric_missing_invalid_and_nonfinite_values_are_rejected(self):
        for invalid in (None, "72", True, float("nan"), float("inf"), -1):
            self.case["clinical_features"]["rest_heart_rate_bpm"] = invalid
            with self.subTest(value=str(invalid)), self.assertRaises(ContractError):
                build_features(self.case, self.contract)

    def test_zero_denominator_is_rejected(self):
        self.case["pet_perfusion"]["rest_mbf"] = 0
        with self.assertRaises(ContractError):
            build_features(self.case, self.contract)

    def test_unknown_input_and_label_are_rejected(self):
        self.case["cmvd_label"] = 1
        with self.assertRaises(ContractError):
            build_features(self.case, self.contract)

    def test_real_input_and_directory_ids_are_rejected(self):
        self.case["data_origin"] = "research"
        with self.assertRaises(ContractError):
            predict_fixture(self.case, self.contract, self.model)
        self.case["data_origin"] = "synthetic"
        self.case["patient_id"] = "POS-001"
        with self.assertRaises(ContractError):
            build_features(self.case, self.contract)

    def test_model_feature_order_and_trained_status_are_checked(self):
        self.model["feature_order"] = list(reversed(self.model["feature_order"]))
        with self.assertRaises(ContractError):
            predict_fixture(self.case, self.contract, self.model)
        self.model["feature_order"] = self.contract["feature_order"]
        self.model["trained"] = True
        with self.assertRaises(ContractError):
            predict_fixture(self.case, self.contract, self.model)

    def test_patient_partitions_reject_overlap_duplicates_and_empty_test(self):
        validate_partitions(["SYNTH-001", "SYNTH-002"], ["SYNTH-003"])
        for development, test in ((["P1"], ["P1"]), (["P1", "P1"], ["P2"]), (["P1"], [])):
            with self.subTest(development=development, test=test), self.assertRaises(ContractError):
                validate_partitions(development, test)


if __name__ == "__main__":
    unittest.main()
