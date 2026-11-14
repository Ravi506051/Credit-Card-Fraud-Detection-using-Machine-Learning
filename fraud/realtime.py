"""Real-time scoring: one transaction in, one decision out.

The model's behaviour features need each card's recent history. `CardState`
holds exactly that, updated after every transaction, so scoring a new
transaction never touches the full dataset. Here the state lives in a Python
dict; a production deployment would keep the same fields in a low-latency
key-value store keyed by card number.

Transactions must arrive in time order per card, as they do at authorisation.
"""
from __future__ import annotations

import math
import time
from collections import deque

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import FEATURES, MAX_WINDOW, MODEL_PATH, NIGHT_HOURS, WINDOWS
from .features import Z_CLIP, clock_features, haversine_km

W_1H, W_24H, W_7D = WINDOWS["1h"], WINDOWS["24h"], WINDOWS["7d"]

# How each feature is worded when it is one of the reasons a transaction was flagged.
REASON_TEXT = {
    "amt": "amount ${v:,.2f}",
    "hour": "made at {v:.0f}:00",
    "day_of_week": "day of week {v:.0f} (0 = Monday)",
    "is_night": "night-time transaction (22:00-03:59)",
    "category": "merchant category '{v}'",
    "age": "cardholder age {v:.0f}",
    "city_pop": "cardholder city population {v:,.0f}",
    "dist_km": "merchant {v:.0f} km from cardholder's home",
    "secs_since_prev": "{v:.1f} h since the card's previous transaction",
    "prev_amt": "previous transaction on the card was ${v:,.2f}",
    "cnt_1h": "{v:.0f} transactions on the card in the previous hour",
    "cnt_24h": "{v:.0f} transactions on the card in the previous 24 h",
    "cnt_7d": "{v:.0f} transactions on the card in the previous 7 days",
    "cnt_30d": "{v:.0f} transactions on the card in the previous 30 days",
    "amt_sum_1h": "${v:,.2f} spent on the card in the previous hour",
    "amt_sum_24h": "${v:,.2f} spent on the card in the previous 24 h",
    "amt_sum_7d": "${v:,.2f} spent on the card in the previous 7 days",
    "amt_ratio_30d": "amount is {v:.1f}x the card's 30-day average",
    "amt_ratio_hist": "amount is {v:.1f}x the card's all-time average",
    "amt_z_hist": "amount is {v:+.1f} standard deviations from the card's average",
    "amt_ratio_max": "amount is {v:.2f}x the card's previous largest transaction",
    "vel_ratio_24h": "24 h activity is {v:.1f}x the card's usual daily pace",
    "cat_share_hist": "category makes up {v:.0%} of the card's past transactions",
    "night_share_hist": "{v:.0%} of the card's past transactions were at night",
}


class CardState:
    """Everything remembered about one card between transactions."""

    __slots__ = (
        "recent", "count", "amt_sum", "amt_sum_sq", "amt_max",
        "night_count", "category_counts", "last_time", "last_amt",
    )  # fmt: skip

    def __init__(self):
        self.recent = deque()  # (unix_time, amount) of transactions in the last 30 days
        self.count = 0
        self.amt_sum = 0.0
        self.amt_sum_sq = 0.0
        self.amt_max = 0.0
        self.night_count = 0
        self.category_counts = {}
        self.last_time = 0
        self.last_amt = 0.0

    def remember(self, t: int, amt: float, cat: int, is_night: bool) -> None:
        recent = self.recent
        recent.append((t, amt))
        while recent[0][0] < t - MAX_WINDOW:
            recent.popleft()
        self.count += 1
        self.amt_sum += amt
        self.amt_sum_sq += amt * amt
        if amt > self.amt_max:
            self.amt_max = amt
        self.night_count += is_night
        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
        self.last_time = t
        self.last_amt = amt


class RealTimeScorer:
    """Scores transactions one at a time against a trained model bundle."""

    def __init__(self, bundle: dict):
        self.booster = lgb.Booster(model_str=bundle["model"])
        self.threshold = float(bundle["threshold"])
        self.categories = list(bundle["categories"])
        self.category_codes = {name: code for code, name in enumerate(self.categories)}
        if list(bundle["features"]) != FEATURES:
            raise ValueError("Model bundle was trained with a different feature list.")
        self.cards: dict[int, CardState] = {}

    @classmethod
    def load(cls, path=MODEL_PATH) -> "RealTimeScorer":
        return cls(joblib.load(path))

    # ----------------------------------------------------------------- state

    def warm_up(self, history: pd.DataFrame) -> None:
        """Replay past transactions (in time order) into card state, without scoring them."""
        t = history["unix_time"].to_numpy(np.int64)
        hour, _ = clock_features(t)
        night = np.isin(hour, NIGHT_HOURS).tolist()
        codes = history["category"].map(self.category_codes).to_numpy()
        cards = self.cards
        for cc, ti, amt, cat, is_night in zip(
            history["cc_num"].tolist(), t.tolist(), history["amt"].tolist(), codes.tolist(), night
        ):
            state = cards.get(cc)
            if state is None:
                state = cards[cc] = CardState()
            state.remember(ti, amt, cat, is_night)

    # -------------------------------------------------------------- features

    def features(self, txn: dict) -> np.ndarray:
        """Feature vector for a transaction, given the card's history so far.

        Mirrors `fraud.features.build_features` exactly, one row at a time.
        """
        t = int(txn["unix_time"])
        amt = float(txn["amt"])
        hour, day_of_week = clock_features(t)
        state = self.cards.get(txn["cc_num"])
        if state is None:
            state = self.cards[txn["cc_num"]] = CardState()
        n = state.count

        # Rolling windows over the card's last 30 days, newest first.
        recent = state.recent
        while recent and recent[0][0] < t - MAX_WINDOW:
            recent.popleft()
        cnt_1h = cnt_24h = cnt_7d = 0
        sum_1h = sum_24h = sum_7d = sum_30d = 0.0
        for t_past, amt_past in reversed(recent):
            elapsed = t - t_past
            sum_30d += amt_past
            if elapsed <= W_7D:
                cnt_7d += 1
                sum_7d += amt_past
                if elapsed <= W_24H:
                    cnt_24h += 1
                    sum_24h += amt_past
                    if elapsed <= W_1H:
                        cnt_1h += 1
                        sum_1h += amt_past
        cnt_30d = len(recent)

        mean_30d = round(sum_30d, 2) / cnt_30d if cnt_30d else 0.0
        hist_mean = state.amt_sum / n if n else 0.0
        hist_std = math.sqrt(max(state.amt_sum_sq / n - hist_mean**2, 0.0)) if n >= 2 else 0.0
        z = (amt - hist_mean) / hist_std if hist_std > 0 else 0.0
        trans_time, dob = pd.Timestamp(txn["trans_date_trans_time"]), pd.Timestamp(txn["dob"])

        values = {
            "amt": amt,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_night": hour in NIGHT_HOURS,
            "category": self.category_codes[txn["category"]],
            "age": (trans_time - dob).days / 365.25,
            "city_pop": txn["city_pop"],
            "dist_km": haversine_km(txn["lat"], txn["long"], txn["merch_lat"], txn["merch_long"]),
            "secs_since_prev": min(t - state.last_time, MAX_WINDOW) if n else MAX_WINDOW,
            "prev_amt": state.last_amt,
            "cnt_1h": cnt_1h,
            "cnt_24h": cnt_24h,
            "cnt_7d": cnt_7d,
            "cnt_30d": cnt_30d,
            "amt_sum_1h": round(sum_1h, 2),
            "amt_sum_24h": round(sum_24h, 2),
            "amt_sum_7d": round(sum_7d, 2),
            "amt_ratio_30d": amt / mean_30d if mean_30d > 0 else 1.0,
            "amt_ratio_hist": amt / hist_mean if hist_mean > 0 else 1.0,
            "amt_z_hist": min(max(z, -Z_CLIP), Z_CLIP),
            "amt_ratio_max": amt / state.amt_max if state.amt_max > 0 else 1.0,
            "vel_ratio_24h": cnt_24h / ((cnt_30d + 1.0) / 30.0),
            "cat_share_hist": state.category_counts.get(self.category_codes[txn["category"]], 0) / n if n else 0.0,
            "night_share_hist": state.night_count / n if n else 0.0,
        }
        return np.array([values[name] for name in FEATURES], dtype=np.float64)

    # --------------------------------------------------------------- scoring

    def score(self, txn: dict) -> dict:
        """Score one incoming transaction, then add it to the card's history."""
        started = time.perf_counter()
        x = self.features(txn)
        row = x.reshape(1, -1)
        probability = float(self.booster.predict(row, num_threads=1)[0])
        flagged = probability >= self.threshold
        reasons = self._reasons(row) if flagged else []

        self.cards[txn["cc_num"]].remember(
            int(txn["unix_time"]), float(txn["amt"]), int(x[FEATURES.index("category")]), bool(x[FEATURES.index("is_night")])
        )
        return {
            "trans_num": txn.get("trans_num"),
            "fraud_probability": probability,
            "flagged": bool(flagged),
            "reasons": reasons,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "features": x,
        }

    def _reasons(self, row: np.ndarray, top: int = 3) -> list[str]:
        """The features that pushed this transaction's score up the most.

        Uses the model's own per-prediction SHAP contributions (in log-odds).
        """
        contributions = self.booster.predict(row, pred_contrib=True, num_threads=1)[0][:-1]
        reasons = []
        for i in np.argsort(contributions)[::-1][:top]:
            if contributions[i] <= 0:
                break
            name, value = FEATURES[i], row[0, i]
            if name == "category":
                value = self.categories[int(value)]
            elif name == "secs_since_prev":
                value = value / 3600
            reasons.append(REASON_TEXT[name].format(v=value))
        return reasons
