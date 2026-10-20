"""Strict feature contract and deliberately untrained synthetic inference.

No image access, model training, clinical inference, or external network calls.
"""

import math
import re


class ContractError(ValueError):
    """Input is outside the currently supported contract."""


def _keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(f"{context}: fields do not match the contract")


def _number(value, context):
    if type(value) not in (int, float):
        raise ContractError(f"{context}: requires a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ContractError(f"{context}: requires a finite number")
    return value


def validate_partitions(development_ids, independent_test_ids):
    """Validate a proposed patient allocation; never create or rename IDs."""
    for ids in (development_ids, independent_test_ids):
        if not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ContractError("Both partitions require nonempty patient IDs")
        if len(set(ids)) != len(ids):
            raise ContractError("Duplicate patient in a partition")
    if set(development_ids) & set(independent_test_ids):
        raise ContractError("Patient leakage between development and test")


def build_features(case, contract):
    """Reject incomplete inputs; ratios are calculated, never label-derived."""
    groups = contract["groups"]
    _keys(case, {"schema_version", "data_origin", "patient_id", *groups}, "case")
    if case["schema_version"] != contract["schema_version"]:
        raise ContractError("Unsupported schema version")
    if case["data_origin"] != "synthetic":
        raise ContractError("Real research inference is not implemented")
    if not isinstance(case["patient_id"], str) or not re.fullmatch(
        r"SYNTH-[0-9]{3,}", case["patient_id"]
    ):
        raise ContractError("Fixture requires an explicit SYNTH patient ID")

    values = {}
    for group, fields in groups.items():
        perfusion = group in ("pet_perfusion", "mr_perfusion")
        _keys(case[group], {*fields, *({"provenance"} if perfusion else set())}, group)
        for field, spec in fields.items():
            value = _number(case[group][field], f"{group}.{field}")
            if "allowed" in spec and value not in spec["allowed"]:
                raise ContractError(f"{field}: unsupported category")
            if "min" in spec and value < spec["min"]:
                raise ContractError(f"{field}: below engineering range")
            if "max" in spec and value > spec["max"]:
                raise ContractError(f"{field}: above engineering range")
            if "min_exclusive" in spec and value <= spec["min_exclusive"]:
                raise ContractError(f"{field}: must be strictly positive")
            values[field] = value
        if perfusion:
            provenance = case[group]["provenance"]
            _keys(provenance, contract["perfusion_provenance_fields"], group)
            if any(not isinstance(v, str) or not v.strip() for v in provenance.values()):
                raise ContractError(f"{group}: incomplete provenance")
            if provenance["qc_status"] != "pass":
                raise ContractError(f"{group}: QC did not pass")
            if provenance["method"] != "synthetic_fixture":
                raise ContractError(f"{group}: only synthetic method supported")

    values["pet_mfr"] = values["stress_mbf"] / values["rest_mbf"]
    values["mr_mpri"] = (
        values["stress_normalized_upslope"] / values["rest_normalized_upslope"]
    )
    return {name: _number(values[name], name) for name in contract["feature_order"]}


def predict_fixture(case, contract, model):
    """Exercise inference wiring, with a hard gate against clinical usage."""
    features = build_features(case, contract)
    if (
        model.get("model_kind") != "fixed_synthetic_fixture"
        or model.get("trained") is not False
        or model.get("allowed_data_origin") != "synthetic"
        or model.get("schema_version") != contract["schema_version"]
        or model.get("feature_order") != contract["feature_order"]
    ):
        raise ContractError("Only the matching untrained synthetic fixture is supported")
    weights = model.get("coefficients")
    if not isinstance(weights, list) or len(weights) != len(features):
        raise ContractError("Fixture coefficient count mismatch")
    threshold = _number(model.get("threshold"), "threshold")
    if not 0 < threshold < 1:
        raise ContractError("Threshold must lie strictly between zero and one")
    logit = _number(model.get("intercept"), "intercept") + sum(
        value * _number(weight, "coefficient")
        for value, weight in zip(features.values(), weights)
    )
    _number(logit, "logit")
    if logit >= 0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probability = exp_logit / (1.0 + exp_logit)
    return {
        "status": "synthetic_simulation_only",
        "data_origin": "synthetic",
        "patient_id": case["patient_id"],
        "model_id": model["model_id"],
        "model_trained": False,
        "schema_version": contract["schema_version"],
        "features": features,
        "simulated_positive_probability": probability,
        "simulated_class": int(probability >= threshold),
        "class_mapping": {"0": "simulated_negative", "1": "simulated_positive"},
        "threshold": threshold,
        "clinical_validation": False,
        "notice": "Synthetic simulation with manually fixed coefficients; not a CMVD diagnosis.",
    }
