"""Binary metrics with explicit undefined values and tie-aware average precision."""

import numpy as np


def distance_metrics(predictions, trained=True):
    valid = np.asarray(predictions["distance_valid"], dtype=bool)
    result = {"distance_head_trained": bool(trained), "distance_valid_count": int(valid.sum()),
              "distance_MAE_m": None}
    if trained and valid.any():
        errors = np.abs(np.asarray(predictions["distance_m"])[valid]
                        - np.asarray(predictions["target_distance_m"])[valid])
        result["distance_MAE_m"] = float(errors.mean())
    return result


def binary_metrics(probability, target, threshold=0.5):
    p, y = np.asarray(probability, dtype=float), np.asarray(target)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Expected equal finite probability/label vectors")
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid binary labels or probabilities")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    y = y.astype(int)
    accepted = p >= threshold
    tp, fp = int(np.sum(accepted & (y == 1))), int(np.sum(accepted & (y == 0)))
    positive, negative = int(np.sum(y == 1)), int(np.sum(y == 0))
    ap = None
    if positive:
        order = np.argsort(-p, kind="stable")
        cumulative = np.cumsum(y[order])
        ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
        hits = cumulative[ends]
        ap = float(np.sum((hits - np.r_[0, hits[:-1]]) / positive * hits / (ends + 1)))
    return {"count": len(y), "positive": positive, "negative": negative,
            "accepted": int(accepted.sum()), "tp": tp, "fp": fp,
            "recall": tp / positive if positive else None,
            "fpr": fp / negative if negative else None,
            "fdr": fp / (tp + fp) if tp + fp else None,
            "coverage": float(accepted.mean()) if len(y) else None,
            "brier": float(np.mean((p - y) ** 2)) if len(y) else None, "ap": ap}


def choose_threshold(probability, target, max_fdr=0.05, min_accepted=100):
    binary_metrics(probability, target)
    p, y = np.asarray(probability, dtype=float), np.asarray(target, dtype=int)
    # Validate even when the calibration set is empty or one-class.
    if not 0 <= max_fdr < 1 or min_accepted < 1 or len(np.unique(y)) != 2:
        raise ValueError("Threshold fitting requires both classes and a valid risk/coverage rule")
    order = np.argsort(-p, kind="stable")
    hits = np.cumsum(y[order])
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    candidates = []
    for end in ends:
        accepted, tp = int(end + 1), int(hits[end])
        fdr = (accepted - tp) / accepted
        if accepted >= min_accepted and fdr <= max_fdr:
            candidates.append((tp / y.sum(), -fdr, float(p[order[end]])))
    if not candidates:
        raise ValueError("Calibration failed: no threshold meets FDR and minimum acceptance")
    threshold = max(candidates)[2]
    return threshold, binary_metrics(p, y, threshold)
