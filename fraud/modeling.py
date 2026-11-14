"""Design matrices, model fitting, class-imbalance strategies and metrics."""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import FEATURES, NUMERIC, RANDOM_STATE, TARGET, VALIDATION_START

# LightGBM gets a generous tree budget and stops once validation PR-AUC has gone
# PATIENCE rounds without improving by at least MIN_GAIN. Without the minimum
# gain it keeps adding trees for the sake of the fourth decimal place.
MAX_TREES = 3000
PATIENCE = 50
MIN_GAIN = 1e-4

LGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

# Right-skewed, non-negative features that logistic regression sees on a log scale.
HEAVY_TAILED = [
    "amt", "city_pop", "dist_km", "secs_since_prev", "prev_amt",
    "cnt_1h", "cnt_24h", "cnt_7d", "cnt_30d",
    "amt_sum_1h", "amt_sum_24h", "amt_sum_7d",
    "amt_ratio_30d", "amt_ratio_hist", "amt_ratio_max", "vel_ratio_24h",
]  # fmt: skip


# --------------------------------------------------------------------------- data


def split_by_time(train: pd.DataFrame, start: str = VALIDATION_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training period into an earlier fit set and a later validation set."""
    is_val = (train["trans_date_trans_time"] >= start).to_numpy()
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)


def tree_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    """Features as LightGBM takes them: `category` as an integer code."""
    X = feats[FEATURES].copy()
    X["category"] = feats["category"].cat.codes.astype(np.int32)
    return X


def onehot_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    """`category` one-hot encoded, for models without native categorical support."""
    dummies = pd.get_dummies(feats["category"], prefix="category", dtype=np.float64)
    return pd.concat([feats[NUMERIC], dummies], axis=1)


def linear_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    """One-hot matrix with heavy-tailed features log-scaled, for logistic regression."""
    X = onehot_matrix(feats)
    X[HEAVY_TAILED] = np.log1p(X[HEAVY_TAILED])
    return X


# ------------------------------------------------------------------------- models


@dataclass
class Fitted:
    """A trained model behind one interface: engineered features in, P(fraud) out."""

    name: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    info: dict = field(default_factory=dict)
    model: object = None


def _timed(fit_fn):
    """Record how long a fit took in the result's `info`."""

    @functools.wraps(fit_fn)
    def wrapper(*args, **kwargs):
        started = time.time()
        fitted = fit_fn(*args, **kwargs)
        fitted.info["fit_seconds"] = round(time.time() - started)
        return fitted

    return wrapper


@_timed
def fit_logreg(fit: pd.DataFrame) -> Fitted:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(linear_matrix(fit), fit[TARGET])
    return Fitted("Logistic regression", lambda f: model.predict_proba(linear_matrix(f))[:, 1], model=model)


@_timed
def fit_random_forest(fit: pd.DataFrame) -> Fitted:
    from sklearn.ensemble import RandomForestClassifier

    # Each tree sees a 25% sample to keep this baseline quick. A forest four times
    # the size took 10 minutes and gained under 0.01 PR-AUC, still well short of
    # gradient boosting.
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        max_samples=0.25,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(onehot_matrix(fit), fit[TARGET])
    return Fitted("Random forest", lambda f: model.predict_proba(onehot_matrix(f))[:, 1], model=model)


@_timed
def fit_xgboost(fit: pd.DataFrame, val: pd.DataFrame) -> Fitted:
    import xgboost as xgb

    # XGBoost builds a tree ~4x slower than LightGBM on this data, so it gets a
    # higher learning rate and a smaller tree budget to keep the comparison quick.
    model = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(onehot_matrix(fit), fit[TARGET], eval_set=[(onehot_matrix(val), val[TARGET])], verbose=False)
    return Fitted(
        "XGBoost",
        lambda f: model.predict_proba(onehot_matrix(f))[:, 1],
        info={"trees": int(model.best_iteration) + 1},
        model=model,
    )


@_timed
def fit_lightgbm(
    fit: pd.DataFrame,
    val: pd.DataFrame | None = None,
    *,
    params: dict | None = None,
    n_estimators: int | None = None,
    imbalance: str = "none",
    columns: list[str] | None = None,
    name: str = "LightGBM",
) -> Fitted:
    """Fit LightGBM.

    With `val`, trees are added until validation PR-AUC stops improving. Without
    it, exactly `n_estimators` trees are built (used for the final refit).
    `imbalance` picks one of the strategies in `rebalance`; `columns` restricts
    the model to a subset of the features (for ablations).
    """
    columns = list(columns or FEATURES)
    X, y, scale_pos_weight = rebalance(tree_matrix(fit)[columns], fit[TARGET].to_numpy(), imbalance)
    settings = {
        **LGBM_PARAMS,
        **(params or {}),
        "objective": "binary",
        "metric": "average_precision",
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }
    train_set = lgb.Dataset(X, y, categorical_feature=[c for c in columns if c == "category"])
    if val is not None:
        booster = lgb.train(
            settings,
            train_set,
            num_boost_round=n_estimators or MAX_TREES,
            valid_sets=[lgb.Dataset(tree_matrix(val)[columns], val[TARGET], reference=train_set)],
            callbacks=[lgb.early_stopping(PATIENCE, min_delta=MIN_GAIN, verbose=False)],
        )
        trees = int(booster.best_iteration)
    else:
        if n_estimators is None:
            raise ValueError("n_estimators is required when there is no validation set to stop on.")
        booster = lgb.train(settings, train_set, num_boost_round=n_estimators)
        trees = n_estimators
    return Fitted(
        name,
        lambda f: booster.predict(tree_matrix(f)[columns].to_numpy(np.float64), num_iteration=trees),
        info={"trees": trees, "train_rows": len(y), "train_frauds": int(y.sum())},
        model=booster,
    )


# ---------------------------------------------------------------- class imbalance

IMBALANCE_STRATEGIES = {
    "none": "No rebalancing",
    "class_weight": "Class weights",
    "undersample": "Random undersampling (1:1)",
    "smote": "SMOTE-NC oversampling (to 10%)",
}


def rebalance(X: pd.DataFrame, y: np.ndarray, strategy: str) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Apply a class-imbalance strategy to the *training* rows only.

    Returns the (possibly resampled) X and y plus the positive-class weight to
    train with. Validation and test data are never resampled: they keep the real
    fraud rate so the metrics mean what they would in production.
    """
    if strategy == "none":
        return X, y, 1.0
    if strategy == "class_weight":
        # Each fraud counts as much as (legit / fraud) legitimate transactions.
        return X, y, float((y == 0).sum() / (y == 1).sum())
    if strategy == "undersample":
        rng = np.random.default_rng(RANDOM_STATE)
        fraud, legit = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        keep = np.sort(np.r_[fraud, rng.choice(legit, size=len(fraud), replace=False)])
        return X.iloc[keep], y[keep], 1.0
    if strategy == "smote":
        from imblearn.over_sampling import SMOTENC

        # SMOTE-NC interpolates numeric features between neighbouring frauds and
        # takes the neighbours' majority vote for the categorical one.
        sampler = SMOTENC(
            categorical_features=[X.columns.get_loc("category")],
            sampling_strategy=0.1,
            random_state=RANDOM_STATE,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        X_res["category"] = X_res["category"].astype(np.int32)
        return X_res, np.asarray(y_res), 1.0
    raise ValueError(f"Unknown imbalance strategy: {strategy!r}")


# ------------------------------------------------------------------------ metrics


def best_f1_threshold(y, p) -> float:
    """Score threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def threshold_metrics(y, p, threshold: float, amount=None) -> dict:
    """Confusion-matrix metrics when every score >= threshold is flagged as fraud."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(p) >= threshold
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    if amount is not None:
        amount = np.asarray(amount, dtype=np.float64)
        out["fraud_amount_caught"] = float(amount[flagged & y].sum() / amount[y].sum()) if y.any() else 0.0
    return out


def evaluate(y, p, threshold: float | None = None, amount=None) -> dict:
    """Ranking metrics plus confusion-matrix metrics.

    With no `threshold`, the F1-optimal threshold for this very data is used.
    That is fine for comparing candidates on validation data, and is never how
    the test set is scored.
    """
    if threshold is None:
        threshold = best_f1_threshold(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        **threshold_metrics(y, p, threshold, amount),
    }


def threshold_table(y, p, thresholds, amount=None) -> pd.DataFrame:
    """Operating points: what each candidate threshold costs in missed fraud and false alarms."""
    return pd.DataFrame([threshold_metrics(y, p, t, amount) for t in thresholds])
