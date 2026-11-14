"""Batch feature engineering.

Raw transactions go in, one row of model features per transaction comes out.

The behaviour features describe how a transaction compares with the same card's
own history. They are built from transactions that happened strictly *before*
the one being described, so they are exactly what a live system would know at
authorisation time, and they never look at the fraud label. `fraud.realtime`
computes the same features one transaction at a time; `realtime_demo.py` checks
that the two implementations agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, MAX_WINDOW, NIGHT_HOURS, TARGET, WINDOWS

RAW_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "category",
    "amt",
    "lat",
    "long",
    "city_pop",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]
KEY_COLUMNS = ["trans_num", "cc_num", "trans_date_trans_time", "unix_time"]
Z_CLIP = 20.0  # a card with two near-identical past amounts would otherwise give absurd z-scores


def clock_features(unix_time):
    """Hour of day and day of week (0 = Monday) from the `unix_time` column.

    `unix_time` is the clock for everything time-related. The human-readable
    `trans_date_trans_time` column is not a consistent timeline: it repeats
    2019-02-28 and skips 2020-02-29, so intervals and weekdays taken from it
    are off by a day for part of the data. `unix_time` is continuous.
    """
    hour = (unix_time % 86_400) // 3_600
    day_of_week = (unix_time // 86_400 + 3) % 7  # 1970-01-01 was a Thursday
    return hour, day_of_week


def load_raw(path, columns=RAW_COLUMNS) -> pd.DataFrame:
    """Read a transactions CSV. `columns=None` loads every column (used for EDA)."""
    return pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["trans_date_trans_time", "dob"],
    )


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def _safe_ratio(num, den, default):
    """num / den where den > 0, otherwise `default` (no history to compare against)."""
    out = np.full(len(num), default, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def build_features(raw: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Engineer model features for every transaction in `raw`.

    Pass train and test concatenated in time order so that cards carry their
    history across the boundary, as they would in production. Rows come back in
    the same order they went in.

    `categories` fixes the category encoding; by default it is the sorted set of
    categories present in `raw`.
    """
    df = raw.reset_index(drop=True)
    n = len(df)
    t_all = df["unix_time"].to_numpy(np.int64)
    hour_all, dow_all = clock_features(t_all)

    # Work in (card, time) order. Ties keep arrival order, which is also the order
    # a live system would process them in.
    order = np.lexsort((np.arange(n), t_all, df["cc_num"].to_numpy()))
    card = df["cc_num"].to_numpy()[order]
    t = t_all[order]
    amt = df["amt"].to_numpy(np.float64)[order]
    is_night = np.isin(hour_all, NIGHT_HOURS).astype(np.float64)[order]

    if categories is None:
        categories = sorted(df["category"].unique())
    category = pd.Categorical(df["category"], categories=categories)
    if (category.codes < 0).any():
        raise ValueError("Transactions contain a category the model was not trained on.")
    cat_code = category.codes[order]

    idx = np.arange(n)
    new_card = np.r_[True, card[1:] != card[:-1]]
    grp = np.cumsum(new_card) - 1
    grp_start = np.maximum.accumulate(np.where(new_card, idx, 0))
    hist_cnt = (idx - grp_start).astype(np.float64)  # transactions this card made before this one

    def prior_cumsum(x):
        """Per-card running total of x over the transactions before each row."""
        inclusive = pd.Series(x).groupby(grp, sort=False).cumsum().to_numpy()
        prior = np.r_[0.0, inclusive[:-1]]
        prior[new_card] = 0.0
        return prior

    def prior_value(x, first):
        """x of the card's previous transaction (`first` when there is none)."""
        prev = np.r_[first, x[:-1]].astype(np.float64)
        prev[new_card] = first
        return prev

    sum_amt = prior_cumsum(amt)
    sum_amt_sq = prior_cumsum(amt * amt)

    out = {}

    # --- recency -------------------------------------------------------------
    prev_t = prior_value(t, 0)
    out["secs_since_prev"] = np.where(new_card, MAX_WINDOW, np.minimum(t - prev_t, MAX_WINDOW))
    out["prev_amt"] = prior_value(amt, 0.0)

    # --- velocity: activity in rolling windows before this transaction --------
    # Rows are sorted by (card, time), so a single composite key lets one binary
    # search find where each card's look-back window starts.
    key = grp.astype(np.int64) * 10**11 + t
    for name, seconds in WINDOWS.items():
        start = np.searchsorted(key, key - seconds, side="left")
        out[f"cnt_{name}"] = (idx - start).astype(np.float64)
        out[f"amt_sum_{name}"] = np.round(sum_amt - sum_amt[start], 2)

    # --- is this amount unusual for this card? --------------------------------
    mean_30d = _safe_ratio(out["amt_sum_30d"], out["cnt_30d"], 0.0)
    out["amt_ratio_30d"] = _safe_ratio(amt, mean_30d, 1.0)

    hist_mean = _safe_ratio(sum_amt, hist_cnt, 0.0)
    hist_var = _safe_ratio(sum_amt_sq, hist_cnt, 0.0) - hist_mean**2
    hist_std = np.sqrt(np.maximum(hist_var, 0.0))
    out["amt_ratio_hist"] = _safe_ratio(amt, hist_mean, 1.0)
    z = _safe_ratio(amt - hist_mean, np.where(hist_cnt >= 2, hist_std, 0.0), 0.0)
    out["amt_z_hist"] = np.clip(z, -Z_CLIP, Z_CLIP)

    running_max = pd.Series(amt).groupby(grp, sort=False).cummax().to_numpy()
    out["amt_ratio_max"] = _safe_ratio(amt, prior_value(running_max, 0.0), 1.0)

    # --- is this pace / category / hour unusual for this card? ----------------
    out["vel_ratio_24h"] = out["cnt_24h"] / ((out["cnt_30d"] + 1.0) / 30.0)
    same_cat_before = pd.Series(idx).groupby([grp, cat_code], sort=False).cumcount().to_numpy()
    out["cat_share_hist"] = _safe_ratio(same_cat_before.astype(np.float64), hist_cnt, 0.0)
    out["night_share_hist"] = _safe_ratio(prior_cumsum(is_night), hist_cnt, 0.0)

    # Back to the caller's row order.
    feats = pd.DataFrame(index=np.arange(n))
    for name, values in out.items():
        col = np.empty(n, dtype=np.float64)
        col[order] = values
        feats[name] = col

    # --- the transaction itself ------------------------------------------------
    feats["amt"] = df["amt"].to_numpy(np.float64)
    feats["hour"] = hour_all.astype(np.float64)
    feats["day_of_week"] = dow_all.astype(np.float64)
    feats["is_night"] = np.isin(hour_all, NIGHT_HOURS).astype(np.float64)
    feats["category"] = category
    feats["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days.to_numpy(np.float64) / 365.25
    feats["city_pop"] = df["city_pop"].to_numpy(np.float64)
    feats["dist_km"] = haversine_km(
        df["lat"].to_numpy(), df["long"].to_numpy(), df["merch_lat"].to_numpy(), df["merch_long"].to_numpy()
    )

    feats = feats[FEATURES].copy()
    for col in KEY_COLUMNS:
        feats[col] = df[col].to_numpy()
    if TARGET in df:
        feats[TARGET] = df[TARGET].to_numpy()
    return feats


def load_features(train_path, test_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs and engineer features over the combined timeline.

    Test transactions get behaviour features that include the card's activity
    during the training period (history only, never labels), which is what a
    deployed system would have on hand.
    """
    train_raw = load_raw(train_path)
    test_raw = load_raw(test_path)
    feats = build_features(pd.concat([train_raw, test_raw], ignore_index=True))
    return feats.iloc[: len(train_raw)].reset_index(drop=True), feats.iloc[len(train_raw) :].reset_index(drop=True)
