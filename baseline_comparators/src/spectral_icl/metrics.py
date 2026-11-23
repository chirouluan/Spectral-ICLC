"""Invalid outputs remain in all denominators; bootstrap unit is physical ID."""
import numpy as np
import pandas as pd


def evaluate(rows, candidates):
    scored = [r for r in rows if r.get("eval_mode") != "not_applicable_unseen_class"]
    n = len(scored)
    if not n:
        return {"accuracy": None, "n_sample_time_predictions": 0}, [], []
    valid = np.array([bool(r["prediction_valid"]) for r in scored])
    truth = np.array([r["y_true"] for r in scored])
    pred = np.array([r["y_pred"] if r["prediction_valid"] else -999 for r in scored])
    correct = (truth == pred) & valid
    per_class, confusion = [], []
    for c in candidates:
        tp = int(((truth == c) & (pred == c)).sum())
        actual, predicted = int((truth == c).sum()), int((pred == c).sum())
        precision, recall = tp / predicted if predicted else 0, tp / actual if actual else 0
        per_class.append(dict(label=c, support=actual, precision=precision, recall=recall,
                              f1=2*precision*recall/(precision+recall) if precision+recall else 0))
        for p in list(candidates) + ["invalid"]:
            confusion.append(dict(y_true=c, y_pred=p,
                                  count=int(((truth == c) & (pred == (-999 if p == "invalid" else p))).sum())))
    seen = np.array([r["is_seen_class"] for r in scored], dtype=bool)
    sa = float(correct[seen].mean()) if seen.any() else None
    ua = float(correct[~seen].mean()) if (~seen).any() else None
    metrics = dict(n_physical_samples=len({r["physical_sample_id"] for r in scored}),
                   n_sample_time_predictions=n, n_valid_predictions=int(valid.sum()),
                   invalid_prediction_rate=float(1-valid.mean()), accuracy=float(correct.mean()),
                   macro_f1=float(np.mean([r["f1"] for r in per_class if r["support"]])),
                   balanced_accuracy=float(np.mean([r["recall"] for r in per_class if r["support"]])),
                   macro_precision=float(np.mean([r["precision"] for r in per_class if r["support"]])),
                   macro_class_scope="classes with ground-truth observations; quarantined unseen Train/Val classes excluded",
                   per_class_recall={str(r["label"]): r["recall"] for r in per_class},
                   ordinal_class_distance_mae_percentage_points=float(np.abs(truth[valid]-pred[valid]).mean()) if valid.any() else None,
                   screening_sensitivity_nonzero=float(((pred[truth != 0] != 0) & valid[truth != 0]).mean()) if (truth != 0).any() else None,
                   screening_specificity_zero=float(((pred[truth == 0] == 0) & valid[truth == 0]).mean()) if (truth == 0).any() else None,
                   seen_accuracy=sa, unseen_accuracy=ua,
                   seen_unseen_harmonic_mean=2*sa*ua/(sa+ua) if sa is not None and ua is not None and sa+ua else (0 if ua is not None else None),
                   total_input_tokens=sum(r.get("input_tokens", 0) or 0 for r in scored),
                   total_output_tokens=sum(r.get("output_tokens", 0) or 0 for r in scored),
                   total_estimated_cost_usd=None,
                   n_context_budget_exceptions=sum(bool(r.get("budget_exception")) for r in scored),
                   not_applicable_reasons={"quarantined": sum(r.get("eval_mode") == "not_applicable_unseen_class" for r in rows)},
                   ordinal_error_scope="valid outputs only; consult invalid_prediction_rate",
                   confidence_interval_95=None)
    return metrics, per_class, confusion


def cluster_bootstrap(rows, seed=12345, n_bootstrap=2000):
    frame = pd.DataFrame(rows)
    frame["correct"] = (frame.y_true == frame.y_pred) & frame.prediction_valid.astype(bool)
    groups = frame.groupby("physical_sample_id").correct.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sampled = groups.iloc[rng.integers(len(groups), size=len(groups))]
        values.append(sampled["sum"].sum() / sampled["count"].sum())
    return np.quantile(values, [.025, .975]).tolist()
