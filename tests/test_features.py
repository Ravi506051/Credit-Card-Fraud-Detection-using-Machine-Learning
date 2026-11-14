"""Checks on the feature pipeline, run on small synthetic data (no CSVs needed).

The properties that matter for a fraud model:
  * behaviour features use only a card's past transactions, never the future;
  * they never depend on the fraud label;
  * the real-time scorer reproduces the batch features exactly.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud import modeling as M
from fraud.config import BEHAVIOUR_FEATURES, FEATURES, MAX_WINDOW, NIGHT_HOURS, TARGET, WINDOWS
from fraud.features import Z_CLIP, build_features
from fraud.realtime import RealTimeScorer

CATEGORIES = ["grocery_pos", "misc_net", "shopping_net", "travel"]


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n, cards = 4000, 15
    # 120 days of activity on a 30-second grid, so some transactions share a timestamp.
    unix_time = np.sort(1_325_376_000 + rng.integers(0, 120 * 86_400 // 30, n) * 30)
    card = rng.integers(0, cards, n)
    return pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(unix_time, unit="s"),
        "cc_num": 4_000_000_000_000 + card,
        "category": rng.choice(CATEGORIES, n),
        "amt": np.round(rng.lognormal(3.5, 1.2, n), 2),
        "lat": 30 + card * 0.7,
        "long": -100 + card * 1.1,
        "city_pop": 1000 * (card + 1),
        "dob": pd.Timestamp("1960-01-01") + pd.to_timedelta(card * 900, unit="D"),
        "trans_num": [f"t{i}" for i in range(n)],
        "unix_time": unix_time,
        "merch_lat": 30 + card * 0.7 + rng.uniform(-1, 1, n),
        "merch_long": -100 + card * 1.1 + rng.uniform(-1, 1, n),
        TARGET: (rng.random(n) < 0.05).astype(int),
    })  # fmt: skip


def reference_behaviour_features(df: pd.DataFrame, i: int) -> dict:
    """Behaviour features for row i, written the slow, obvious way."""
    row = df.iloc[i]
    past = df.iloc[:i]
    past = past[past["cc_num"] == row["cc_num"]]
    t, amt = row["unix_time"], row["amt"]
    out = {}
    for name, seconds in WINDOWS.items():
        window = past[past["unix_time"] >= t - seconds]
        out[f"cnt_{name}"] = len(window)
        out[f"amt_sum_{name}"] = round(window["amt"].sum(), 2)
    mean_30d = out["amt_sum_30d"] / out["cnt_30d"] if out["cnt_30d"] else 0.0
    out["amt_ratio_30d"] = amt / mean_30d if mean_30d > 0 else 1.0
    out["vel_ratio_24h"] = out["cnt_24h"] / ((out["cnt_30d"] + 1) / 30)
    if len(past):
        std = past["amt"].std(ddof=0)
        out["secs_since_prev"] = min(t - past["unix_time"].iloc[-1], MAX_WINDOW)
        out["prev_amt"] = past["amt"].iloc[-1]
        out["amt_ratio_hist"] = amt / past["amt"].mean()
        out["amt_z_hist"] = float(np.clip((amt - past["amt"].mean()) / std, -Z_CLIP, Z_CLIP)) if len(past) >= 2 and std > 0 else 0.0
        out["amt_ratio_max"] = amt / past["amt"].max()
        out["cat_share_hist"] = (past["category"] == row["category"]).mean()
        out["night_share_hist"] = ((past["unix_time"] % 86_400) // 3_600).isin(NIGHT_HOURS).mean()
    else:
        out.update(secs_since_prev=MAX_WINDOW, prev_amt=0.0, amt_ratio_hist=1.0, amt_z_hist=0.0, amt_ratio_max=1.0, cat_share_hist=0.0, night_share_hist=0.0)
    return {name: out[name] for name in BEHAVIOUR_FEATURES}


def test_behaviour_features_match_reference(transactions):
    feats = build_features(transactions)
    for i in np.random.default_rng(1).choice(len(transactions), 300, replace=False):
        expected = reference_behaviour_features(transactions, i)
        actual = feats.loc[i, BEHAVIOUR_FEATURES].astype(float).to_dict()
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), f"row {i}"


def test_features_never_look_at_the_future(transactions):
    """Features for the first k transactions are the same whether or not later ones exist."""
    full = build_features(transactions, categories=CATEGORIES)
    for k in (1, 500, 2500):
        prefix = build_features(transactions.iloc[:k], categories=CATEGORIES)
        pd.testing.assert_frame_equal(prefix[FEATURES], full[FEATURES].iloc[:k], check_exact=False, rtol=1e-12)


def test_features_do_not_depend_on_the_label(transactions):
    flipped = transactions.assign(**{TARGET: 1 - transactions[TARGET]})
    pd.testing.assert_frame_equal(build_features(transactions)[FEATURES], build_features(flipped)[FEATURES])


def test_realtime_scorer_matches_batch_pipeline(transactions):
    feats = build_features(transactions, categories=CATEGORIES)
    X = M.tree_matrix(feats)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 15, "min_data_in_leaf": 5, "seed": 0},
        lgb.Dataset(X, feats[TARGET], categorical_feature=["category"]),
        num_boost_round=30,
    )
    bundle = {"model": booster.model_to_string(), "features": FEATURES, "categories": CATEGORIES, "threshold": 0.5}

    # Warm up on the first half, then stream the second half one transaction at a time.
    half = len(transactions) // 2
    scorer = RealTimeScorer(bundle)
    scorer.warm_up(transactions.iloc[:half])
    results = [scorer.score(row._asdict()) for row in transactions.iloc[half:].itertuples(index=False)]

    X_batch = X.iloc[half:].to_numpy(np.float64)
    np.testing.assert_allclose(np.vstack([r["features"] for r in results]), X_batch, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose([r["fraud_probability"] for r in results], booster.predict(X_batch), rtol=1e-9)
