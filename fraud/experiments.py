"""The experiments behind the final model.

Every experiment fits on the earlier part of the training period and is judged
on the later part (`modeling.split_by_time`). The test set plays no role here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import modeling as M
from .config import RANDOM_STATE, TARGET, TRANSACTION_FEATURES

TABLE_COLUMNS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn", "threshold"]

# Small random search around the default LightGBM settings.
SEARCH_SPACE = {
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50, 100],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "subsample": [0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def _row(name: str, fitted: M.Fitted, val: pd.DataFrame) -> tuple[dict, np.ndarray]:
    scores = fitted.predict(val)
    metrics = M.evaluate(val[TARGET], scores)
    row = {"model": name, **{k: metrics[k] for k in TABLE_COLUMNS}, "trees": fitted.info.get("trees"), "fit_seconds": fitted.info["fit_seconds"]}
    return row, scores


def _run(candidates: dict, val: pd.DataFrame, log=print) -> tuple[pd.DataFrame, dict, dict]:
    rows, scores, fitted = [], {}, {}
    for name, fit_fn in candidates.items():
        fitted[name] = fit_fn()
        row, scores[name] = _row(name, fitted[name], val)
        rows.append(row)
        log(f"  {name:<42} PR-AUC {row['pr_auc']:.4f}  precision {row['precision']:.3f}  recall {row['recall']:.3f}  ({row['fit_seconds']}s)")
    return pd.DataFrame(rows).set_index("model"), scores, fitted


def compare_models(fit: pd.DataFrame, val: pd.DataFrame, log=print):
    """Which model family? Plus one ablation showing what the behaviour features add.

    Precision / recall / F1 are at each model's own F1-optimal validation threshold.
    """
    candidates = {
        "Logistic regression": lambda: M.fit_logreg(fit),
        "Random forest": lambda: M.fit_random_forest(fit),
        "XGBoost": lambda: M.fit_xgboost(fit, val),
        "LightGBM": lambda: M.fit_lightgbm(fit, val),
        "LightGBM, transaction features only": lambda: M.fit_lightgbm(fit, val, columns=TRANSACTION_FEATURES),
    }
    return _run(candidates, val, log)


def compare_imbalance_strategies(fit: pd.DataFrame, val: pd.DataFrame, log=print, reuse: dict | None = None):
    """Which way of handling the 170:1 class imbalance? Same model, same validation data.

    `reuse` maps a strategy to a model already fitted with it on the same data,
    to save fitting it again.
    """
    reuse = reuse or {}
    candidates = {
        label: (lambda s=strategy: reuse.get(s) or M.fit_lightgbm(fit, val, imbalance=s))
        for strategy, label in M.IMBALANCE_STRATEGIES.items()
    }
    table, scores, fitted = _run(candidates, val, log)
    table.insert(0, "strategy", list(M.IMBALANCE_STRATEGIES))
    table["train_rows"] = [fitted[name].info["train_rows"] for name in table.index]
    table["train_frauds"] = [fitted[name].info["train_frauds"] for name in table.index]
    return table, scores, fitted


def tune_lightgbm(fit: pd.DataFrame, val: pd.DataFrame, imbalance: str, n_trials: int = 6, log=print, default: M.Fitted | None = None):
    """Random search over `SEARCH_SPACE`; the default settings are always trial 0.

    `default` is the default-settings model if it has already been fitted on the
    same data with the same imbalance strategy.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    trials = [dict(M.LGBM_PARAMS)]
    while len(trials) < n_trials + 1:
        params = {**M.LGBM_PARAMS, **{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}}
        if params not in trials:
            trials.append(params)

    rows, best = [], None
    for i, params in enumerate(trials):
        fitted = default if i == 0 and default is not None else M.fit_lightgbm(fit, val, params=params, imbalance=imbalance)
        row, scores = _row(f"trial {i}" + (" (default)" if i == 0 else ""), fitted, val)
        rows.append({**row, **{k: params[k] for k in SEARCH_SPACE}})
        log(f"  {row['model']:<18} PR-AUC {row['pr_auc']:.4f}  F1 {row['f1']:.4f}  trees {row['trees']}  {({k: params[k] for k in SEARCH_SPACE})}")
        if best is None or row["pr_auc"] > best[0]:
            best = (row["pr_auc"], params, fitted, scores)
    return pd.DataFrame(rows).set_index("model"), best[1], best[2], best[3]


def performance_by_month(feats: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Metrics per calendar month at a fixed threshold, to show whether the model decays as it ages."""
    month = feats["trans_date_trans_time"].dt.to_period("M").astype(str).to_numpy()
    rows = []
    y = feats[TARGET].to_numpy()
    for m in np.unique(month):
        mask = month == m
        metrics = M.threshold_metrics(y[mask], scores[mask], threshold)
        frauds = metrics["tp"] + metrics["fn"]
        if not frauds:  # recall and PR-AUC are undefined for a month with no fraud in it
            metrics.update(recall=np.nan, f1=np.nan, precision=np.nan if not metrics["fp"] else 0.0)
        rows.append({
            "month": m,
            "transactions": int(mask.sum()),
            "frauds": frauds,
            "pr_auc": average_precision_score(y[mask], scores[mask]) if frauds else np.nan,
            **{k: metrics[k] for k in ["precision", "recall", "f1", "fp", "fn"]},
        })
    return pd.DataFrame(rows).set_index("month")


def unseen_card_check(fit: pd.DataFrame, val: pd.DataFrame, params: dict, imbalance: str, holdout: float = 0.2) -> pd.DataFrame:
    """Does the model work on cards it has never seen?

    The dataset has under 1,000 cards, and cardholder attributes such as age and
    city population nearly identify a card. A model could score well by
    remembering which customers were defrauded rather than learning what fraud
    looks like. Here a fifth of the cards is withheld from fitting entirely, and
    validation-period performance is compared between cards the model trained on
    and cards it has never seen.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    cards = np.sort(pd.concat([fit["cc_num"], val["cc_num"]]).unique())
    held_out = set(rng.choice(cards, size=int(len(cards) * holdout), replace=False))
    fit_seen = fit[~fit["cc_num"].isin(held_out)]
    val_seen = val[~val["cc_num"].isin(held_out)]
    val_unseen = val[val["cc_num"].isin(held_out)]

    fitted = M.fit_lightgbm(fit_seen, val_seen, params=params, imbalance=imbalance)
    threshold = M.best_f1_threshold(val_seen[TARGET], fitted.predict(val_seen))
    rows = []
    for label, part in [("Cards seen in training", val_seen), ("Cards never seen", val_unseen)]:
        metrics = M.evaluate(part[TARGET], fitted.predict(part), threshold)
        rows.append({
            "validation cards": label,
            "cards": part["cc_num"].nunique(),
            "transactions": len(part),
            "frauds": int(part[TARGET].sum()),
            **{k: metrics[k] for k in ["pr_auc", "roc_auc", "precision", "recall", "f1"]},
        })
    return pd.DataFrame(rows).set_index("validation cards")
