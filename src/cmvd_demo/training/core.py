"""Patient nested CV with fold-local preprocessing and JSON-only fitted models."""

import warnings

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmvd_demo.cohort import make_folds
from cmvd_demo.cohort.inputs import check_cancel
from cmvd_demo.roi import document_hash
from .data import TrainingError, require, validate_values, validate_folds
from .metrics import evaluate_oof


def validate_contract(contract, feature_contract):
    order = feature_contract["feature_order"]
    require(contract.get("schema_version") == 1 and contract.get("feature_contract_id") == feature_contract["contract_id"]
            and contract.get("missing_policy") == feature_contract["missing_policy"], "training_contract_mismatch")
    continuous, binary = contract["continuous_features"], contract["binary_features"]
    require(len(set(continuous+binary)) == len(order) and set(continuous+binary) == set(order), "training_preprocessing_fields")
    for key in binary:
        require(feature_contract["clinical_fields"].get(key, {}).get("allowed") == [0, 1], "training_binary_contract")
    variants = contract["variants"]
    clinical = [k for k in order if k in feature_contract["clinical_fields"]]
    require(variants == {"clinical": clinical, "clinical_pet": clinical+["pet_mfr"],
                         "clinical_mr": clinical+["mr_mpri"], "fusion": order}
            and contract.get("primary_model") == "fusion", "training_ablation_contract")
    require(all(type(contract[k]) is int and 2 <= contract[k] <= 10 for k in ("outer_splits", "inner_splits"))
            and type(contract["random_seed"]) is int, "training_fold_policy")
    cs = contract["candidate_C"]
    require(isinstance(cs, list) and 1 <= len(cs) <= 10 and all(type(c) in (int, float) and np.isfinite(c) and c > 0 for c in cs)
            and cs == sorted(set(cs)), "training_C_policy")
    require(contract.get("solver") == "lbfgs" and contract.get("l1_ratio") == 0
            and contract.get("selection_metric") == "mean_inner_log_loss" and contract.get("tie_break") == "smallest_C"
            and contract.get("threshold") == .5 and contract.get("threshold_basis") == "prespecified_engineering_rule_not_clinical_cutoff",
            "training_model_or_threshold_policy")
    require(type(contract.get("max_iter")) is int and 1 <= contract["max_iter"] <= 10000
            and type(contract.get("tol")) in (int, float) and 0 < contract["tol"] <= .01
            and type(contract.get("bootstrap_replicates")) is int and 1 <= contract["bootstrap_replicates"] <= 5000,
            "training_iteration_budget")


def build_pipeline(fields, contract, c):
    continuous = [j for j, key in enumerate(fields) if key in contract["continuous_features"]]
    binary = [j for j, key in enumerate(fields) if key in contract["binary_features"]]
    preprocessing = ColumnTransformer([("continuous", StandardScaler(), continuous), ("binary", "passthrough", binary)],
                                      remainder="drop", sparse_threshold=0)
    classifier = LogisticRegression(C=c, solver=contract["solver"], l1_ratio=0,
                                    max_iter=contract["max_iter"], tol=contract["tol"],
                                    class_weight=None, random_state=contract["random_seed"])
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def fit_pipeline(x, y, fields, contract, c, *, cancel=None):
    check_cancel(cancel)
    require(set(y) == {0, 1}, "training_fold_single_class")
    model = build_pipeline(fields, contract, c)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, y)
    except ConvergenceWarning:
        raise TrainingError("training_solver_not_converged") from None
    check_cancel(cancel)
    return model


def export_model(model, fields, data, contract, c):
    continuous = [i for i, key in enumerate(fields) if key in contract["continuous_features"]]
    binary = [i for i, key in enumerate(fields) if key in contract["binary_features"]]
    scaler = model.named_steps["preprocess"].named_transformers_["continuous"]
    mean, scale, coefficients = np.zeros(len(fields)), np.ones(len(fields)), np.zeros(len(fields))
    mean[continuous], scale[continuous] = scaler.mean_, scaler.scale_
    classifier = model.named_steps["classifier"]
    coefficients[continuous+binary] = classifier.coef_[0]
    rules = data.feature_contract["clinical_fields"] | data.feature_contract["perfusion_fields"]
    return {"schema_version": 1, "kind": "standardized_logistic_binary", "mode": data.audit["mode"],
            "feature_order": fields, "feature_units": {k: rules[k]["unit"] for k in fields},
            "feature_rules": {k: rules[k] for k in fields}, "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coefficients.tolist(), "intercept": float(classifier.intercept_[0]),
            "classes": [0, 1], "threshold": contract["threshold"], "threshold_basis": contract["threshold_basis"],
            "C": c, "feature_contract_sha256": document_hash(data.feature_contract),
            "training_contract_sha256": document_hash(contract), "source_cohort_bundle_sha256": data.snapshot_sha256,
            "clinical_validated": False}


def model_probabilities(model, x):
    n = len(model["feature_order"])
    mean, scale, coefficient = (np.asarray(model[k], dtype=np.float64) for k in ("mean", "scale", "coefficient"))
    require(all(v.shape == (n,) and np.isfinite(v).all() for v in (mean, scale, coefficient)) and (scale > 0).all()
            and np.isfinite(model["intercept"]), "training_model_parameters_invalid")
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == n and np.isfinite(x).all(), "training_prediction_shape_or_value")
    z = ((x-mean)/scale) @ coefficient + model["intercept"]
    require(np.isfinite(z).all(), "training_prediction_overflow")
    return np.exp(-np.logaddexp(0., -z))


def predict_features(model, features, units, *, feature_contract_sha256, allow_synthetic=False):
    require(model.get("kind") == "standardized_logistic_binary" and model.get("schema_version") == 1
            and model.get("mode") in ("research", "synthetic_fixture") and model.get("classes") == [0, 1]
            and model.get("threshold") == .5, "training_prediction_model_contract")
    synthetic = model["mode"] == "synthetic_fixture"
    require(not synthetic or allow_synthetic is True, "training_synthetic_explicit_opt_in_required")
    require(isinstance(features, dict) and set(features) == set(model["feature_order"]), "training_prediction_fields")
    require(units == model["feature_units"] and feature_contract_sha256 == model["feature_contract_sha256"],
            "training_prediction_units_or_contract")
    rules = {"clinical_fields": model["feature_rules"], "perfusion_fields": {}}
    try:
        x = validate_values([[float(features[k]) for k in model["feature_order"]]], model["feature_order"], rules)
    except (ValueError, TypeError, OverflowError):
        raise TrainingError("training_prediction_invalid_features") from None
    p = float(model_probabilities(model, x)[0])
    return {"probability_positive": p, "cmvd_label": int(p >= model["threshold"]),
            "synthetic": synthetic, "clinical_validated": False}


def indices_for(data, patients):
    lookup = {patient: i for i, patient in enumerate(data.patients)}
    return np.asarray([lookup[p] for p in patients], dtype=np.int64)


def select_c(data, indices, fields, contract, *, cancel=None):
    samples = [{"patient_key": data.patients[i], "partition": "development", "label": int(data.y[i])} for i in indices]
    folds = make_folds(samples, contract["inner_splits"], contract["random_seed"])
    require(folds["ready"], "training_inner_class_count_insufficient")
    columns = [data.feature_contract["feature_order"].index(key) for key in fields]
    candidates, audits = [], []
    for c in contract["candidate_C"]:
        losses = []
        for fold in folds["folds"]:
            train, valid = (indices_for(data, fold[k]) for k in ("train_patients", "validation_patients"))
            fitted = fit_pipeline(data.x[np.ix_(train, columns)], data.y[train], fields, contract, c, cancel=cancel)
            losses.append(float(log_loss(data.y[valid], fitted.predict_proba(data.x[np.ix_(valid, columns)])[:, 1], labels=[0, 1])))
            if c == contract["candidate_C"][0]:
                exported = export_model(fitted, fields, data, contract, c)
                audits.append({**fold, "preprocessing_mean": exported["mean"], "preprocessing_scale": exported["scale"]})
        candidates.append({"C": c, "fold_log_losses": losses, "mean_log_loss": float(np.mean(losses))})
    chosen = min(candidates, key=lambda r: (r["mean_log_loss"], r["C"]))["C"]
    return chosen, {"chosen_C": chosen, "candidates": candidates, "inner_folds": audits}


def fit_outer(data, train, valid, fields, contract, *, cancel=None):
    c, selection = select_c(data, train, fields, contract, cancel=cancel)
    columns = [data.feature_contract["feature_order"].index(key) for key in fields]
    fitted = fit_pipeline(data.x[np.ix_(train, columns)], data.y[train], fields, contract, c, cancel=cancel)
    model = export_model(fitted, fields, data, contract, c)
    expected = fitted.predict_proba(data.x[np.ix_(train, columns)])[:, 1]
    error = float(np.max(np.abs(expected-model_probabilities(model, data.x[np.ix_(train, columns)]))))
    require(error <= 1e-12, "training_numeric_export_mismatch")
    model["numeric_export_max_abs_error"] = error
    p = fitted.predict_proba(data.x[np.ix_(valid, columns)])[:, 1] if len(valid) else np.array([])
    return p, model, selection


def train_nested(data, contract, *, progress=None, cancel=None):
    validate_contract(contract, data.feature_contract)
    require(data.cv_policy == {"n_splits": contract["outer_splits"], "random_seed": contract["random_seed"]},
            "training_frozen_fold_policy_mismatch")
    validate_values(data.x, data.feature_contract["feature_order"], data.feature_contract)
    require(data.y.ndim == 1 and set(data.y) == {0, 1}, "training_label_values")
    validate_folds(data.folds, data.patients, n_splits=contract["outer_splits"])
    samples = [{"patient_key": p, "partition": "development", "label": int(y)} for p, y in zip(data.patients, data.y)]
    expected = make_folds(samples, contract["outer_splits"], contract["random_seed"])
    require(expected["ready"] and list(data.folds) == expected["folds"], "training_frozen_fold_policy_mismatch")
    require(len(set(data.patients)) == len(data.patients) == len(data.x) == len(data.y), "training_patient_uniqueness")
    # Preflight every inner split before the first fit, never silently reduce K.
    for fold in data.folds:
        train = indices_for(data, fold["train_patients"])
        inner = make_folds([samples[i] for i in train], contract["inner_splits"], contract["random_seed"])
        require(inner["ready"], "training_inner_class_count_insufficient")
    models, reports, audit, predictions, roundtrip = {}, {}, {}, [], {}
    for variant_index, (name, fields) in enumerate(contract["variants"].items()):
        p = np.full(len(data.y), np.nan)
        outer_audit = []
        for fold in data.folds:
            check_cancel(cancel)
            if progress:
                progress("nested_outer_fold", variant_index*len(data.folds)+fold["fold"], len(contract["variants"])*len(data.folds))
            train, valid = (indices_for(data, fold[k]) for k in ("train_patients", "validation_patients"))
            pv, model, selection = fit_outer(data, train, valid, fields, contract, cancel=cancel)
            p[valid] = pv
            outer_audit.append({**fold, "selection": selection, "refit_model": model})
            predictions.extend({"variant": name, "research_id": data.research_ids[i], "patient_key": data.patients[i],
                                "outer_fold": fold["fold"], "cmvd_label": int(data.y[i]),
                                "probability_positive": float(pi), "predicted_label": int(pi >= contract["threshold"])} for i, pi in zip(valid, pv))
        require(np.isfinite(p).all(), "training_oof_incomplete")
        if progress:
            progress("oof_metrics", variant_index+1, len(contract["variants"]))
        reports[name] = evaluate_oof(data.y, p, contract, cancel=cancel)
        all_indices = np.arange(len(data.y))
        _, model, final_selection = fit_outer(data, all_indices, np.array([], dtype=int), fields, contract, cancel=cancel)
        models[name] = model
        audit[name] = {"outer_folds": outer_audit, "final_development_selection": final_selection}
        roundtrip[name] = model["numeric_export_max_abs_error"]
    check_cancel(cancel)
    return {"models": models, "fold_audit": audit, "oof_predictions": predictions,
            "report": {"schema_version": 1, "kind": "nested_patient_logistic_training", "mode": data.audit["mode"],
                "development_patients": len(data.y), "class_counts": {str(y): int(np.sum(data.y == y)) for y in (0, 1)},
                "primary_model": contract["primary_model"], "threshold": contract["threshold"],
                "threshold_basis": contract["threshold_basis"], "population": data.audit.get("population"),
                "source_cohort_bundle_sha256": data.snapshot_sha256, "feature_contract_sha256": document_hash(data.feature_contract),
                "training_contract_sha256": document_hash(contract),
                "label_definition_version": data.audit["label_definition_version"],
                "label_definition_reference": data.audit["label_definition_reference"],
                "oof_metrics": reports, "numeric_export_max_abs_error": roundtrip,
                "synthetic_training_performed": data.synthetic, "real_training_performed": not data.synthetic,
                "independent_evaluation_performed": False, "clinical_validated": False,
                "limitations": ["OOF comparison uses the same complete-case patients and folds for every variant",
                    "Primary model prespecified; no winner selected using outer folds", "No independent test statistics used",
                    "Final refit uses all eligible development patients, not a validation score",
                    "Synthetic fixture performance has no clinical interpretation", "CNN/image training and desktop inference are not implemented"]}}
