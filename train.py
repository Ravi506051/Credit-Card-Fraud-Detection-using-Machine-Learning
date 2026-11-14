"""Train, select and evaluate the fraud detection model end to end.

    python train.py [--train PATH] [--test PATH] [--trials N]

Writes the model bundle to models/, and metrics, tables and figures to reports/.
Model selection uses a time-based validation split of the training file; the
test file is scored exactly once, at the end, with the already-chosen threshold.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from fraud import experiments as E
from fraud import modeling as M
from fraud import plots as P
from fraud.config import DEFAULT_TEST, DEFAULT_TRAIN, FEATURES, MODEL_PATH, REPORTS_DIR, TARGET, VALIDATION_START
from fraud.features import load_features, load_raw

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 74 - len(title)), flush=True)


def log(message: str) -> None:
    print(message, flush=True)


def shap_importance(booster, feats: pd.DataFrame, sample: int = 50_000) -> pd.Series:
    """Mean absolute SHAP contribution of each feature over a random sample of rows."""
    rows = feats.sample(min(sample, len(feats)), random_state=0)
    contributions = booster.predict(M.tree_matrix(rows).to_numpy(np.float64), pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(contributions).mean(axis=0), index=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="training transactions CSV")
    parser.add_argument("--test", default=DEFAULT_TEST, help="held-out test transactions CSV")
    parser.add_argument("--trials", type=int, default=6, help="random-search trials for LightGBM tuning")
    args = parser.parse_args()
    started = time.time()
    P.use_style()
    REPORTS_DIR.mkdir(exist_ok=True)

    section("Data and features")
    train, test = load_features(args.train, args.test)
    fit, val = M.split_by_time(train)
    for name, part in [("train", train), ("  fit (before " + VALIDATION_START + ")", fit), ("  validation (from " + VALIDATION_START + ")", val), ("test", test)]:
        log(f"{name:<32} {len(part):>9,} transactions  {int(part[TARGET].sum()):>6,} frauds  ({part[TARGET].mean():.3%})")

    raw_train = load_raw(args.train)
    P.save(P.fraud_rate_by_hour(raw_train), "eda_fraud_rate_by_hour")
    P.save(P.fraud_rate_by_category(raw_train), "eda_fraud_rate_by_category")
    P.save(P.amount_distribution(raw_train), "eda_amount_distribution")
    del raw_train

    section("1. Model families (validation)")
    models, model_scores, fitted_models = E.compare_models(fit, val, log)
    models.to_csv(REPORTS_DIR / "model_comparison.csv")
    families = {k: v for k, v in model_scores.items() if "only" not in k}
    P.save(P.pr_curves(val[TARGET], families, "Gradient boosting separates fraud far better than the baselines", "Precision-recall on validation data"), "model_comparison_pr")

    section("2. Class-imbalance strategies (LightGBM, validation)")
    strategies, _, fitted_strategies = E.compare_imbalance_strategies(fit, val, log, reuse={"none": fitted_models["LightGBM"]})
    strategies.to_csv(REPORTS_DIR / "imbalance_strategies.csv")
    P.save(
        P.strategy_dotplot(strategies, "Rebalancing the training data does not improve the model", "Validation data; precision and recall at each variant's F1-optimal threshold. Axes are zoomed."),
        "imbalance_strategies",
    )
    best_strategy = strategies["pr_auc"].idxmax()
    imbalance = strategies.loc[best_strategy, "strategy"]
    log(f"-> best by PR-AUC: {best_strategy}")

    section("3. Hyperparameter search (validation)")
    tuning, params, tuned, val_scores = E.tune_lightgbm(fit, val, imbalance, args.trials, log, default=fitted_strategies[best_strategy])
    tuning.to_csv(REPORTS_DIR / "tuning.csv")
    log(f"-> selected: {params} with {tuned.info['trees']} trees")

    section("4. Decision threshold (validation)")
    threshold = M.best_f1_threshold(val[TARGET], val_scores)
    val_metrics = M.evaluate(val[TARGET], val_scores, threshold, val["amt"])
    grid = sorted({0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, round(threshold, 4)})
    val_tradeoff = M.threshold_table(val[TARGET], val_scores, grid, val["amt"])
    val_tradeoff.to_csv(REPORTS_DIR / "threshold_tradeoff_validation.csv", index=False)
    log(val_tradeoff[["threshold", "precision", "recall", "f1", "fp", "fn", "fraud_amount_caught"]].round(4).to_string(index=False))
    log(f"-> F1-optimal threshold: {threshold:.4f}")
    P.save(P.threshold_tradeoff(val[TARGET], val_scores, threshold), "threshold_tradeoff")

    section("5. Robustness: cards the model has never seen (validation)")
    unseen = E.unseen_card_check(fit, val, params, imbalance)
    unseen.to_csv(REPORTS_DIR / "unseen_card_check.csv")
    log(unseen.round(4).to_string())

    section("6. Final model: refit on all training data, score the test set once")
    # More data supports proportionally more trees than early stopping found on the fit split.
    final_trees = round(tuned.info["trees"] * len(train) / len(fit))
    final = M.fit_lightgbm(train, params=params, n_estimators=final_trees, imbalance=imbalance)
    test_scores = final.predict(test)
    test_metrics = M.evaluate(test[TARGET], test_scores, threshold, test["amt"])
    test_tradeoff = M.threshold_table(test[TARGET], test_scores, grid, test["amt"])
    test_tradeoff.to_csv(REPORTS_DIR / "threshold_tradeoff_test.csv", index=False)
    for name, m in [("validation", val_metrics), ("test", test_metrics)]:
        log(
            f"{name:<11} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
            f"F1 {m['f1']:.4f}  FP {m['fp']:,}  FN {m['fn']:,}  fraud $ caught {m['fraud_amount_caught']:.1%}"
        )
    by_month = E.performance_by_month(test, test_scores, threshold)
    by_month.to_csv(REPORTS_DIR / "test_by_month.csv")
    log("\nTest set by month:\n" + by_month.round(4).to_string())
    importance = shap_importance(final.model, test)
    importance.sort_values(ascending=False).to_csv(REPORTS_DIR / "feature_importance.csv", header=["mean_abs_shap"])
    P.save(P.confusion_matrix(test_metrics, f"Test set decisions at threshold {threshold:.3f}"), "test_confusion_matrix")
    P.save(P.test_pr_curve(test[TARGET], test_scores, test_metrics), "test_pr_curve")
    P.save(P.feature_importance(importance), "feature_importance")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": final.model.model_to_string(),
            "features": FEATURES,
            "categories": list(train["category"].cat.categories),
            "threshold": threshold,
            "test_metrics": test_metrics,
        },
        MODEL_PATH,
    )
    summary = {
        "data": {
            "train_rows": len(train), "train_frauds": int(train[TARGET].sum()),
            "fit_rows": len(fit), "validation_rows": len(val), "validation_start": VALIDATION_START,
            "test_rows": len(test), "test_frauds": int(test[TARGET].sum()),
        },
        "selected": {
            "model": "LightGBM", "imbalance_strategy": imbalance, "params": params,
            "trees_at_early_stopping": tuned.info["trees"], "trees_final": final_trees, "threshold": threshold,
        },
        "validation": val_metrics,
        "test": test_metrics,
    }  # fmt: skip
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"\nSaved model to {MODEL_PATH} and reports to {REPORTS_DIR}  ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
