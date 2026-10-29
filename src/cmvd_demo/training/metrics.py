"""Fixed-threshold OOF summaries and fixed-prediction patient bootstrap."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from cmvd_demo.cohort.inputs import check_cancel


def scores(y, p, threshold):
    predicted = p >= threshold
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tp = int(np.sum((y == 1) & predicted))
    sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity+specificity)/2,
            "brier": float(np.mean((y-p)**2)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp]}


def evaluate_oof(y, p, contract, *, cancel=None):
    metrics = scores(y, p, contract["threshold"])
    bins = []
    for i in range(5):
        mask = (p >= i/5) & ((p < (i+1)/5) if i < 4 else (p <= 1))
        bins.append({"lower": i/5, "upper": (i+1)/5, "n": int(mask.sum()),
                     "mean_prediction": float(p[mask].mean()) if mask.any() else None,
                     "observed_fraction": float(y[mask].mean()) if mask.any() else None})
    rng = np.random.default_rng(contract["random_seed"])
    values = {k: [] for k in metrics if k != "confusion_matrix_tn_fp_fn_tp"}
    skipped = 0
    for _ in range(contract["bootstrap_replicates"]):
        check_cancel(cancel)
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) != 2:
            skipped += 1
            continue
        sample = scores(y[indices], p[indices], contract["threshold"])
        for key in values:
            values[key].append(sample[key])
    intervals = {key: np.quantile(v, [.025, .975]).tolist() if v else None for key, v in values.items()}
    return {"point_estimates": metrics, "calibration_fixed_bins": bins,
            "bootstrap_95_percentile": intervals, "bootstrap_requested": contract["bootstrap_replicates"],
            "bootstrap_valid": contract["bootstrap_replicates"]-skipped, "bootstrap_single_class_skipped": skipped,
            "uncertainty_scope": "Patient resampling of fixed OOF predictions; excludes model fitting/selection uncertainty. Single-class resamples omitted.",
            "pr_summary": "average_precision_step_integral_not_trapezoidal_pr_auc"}
