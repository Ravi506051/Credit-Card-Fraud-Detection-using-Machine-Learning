"""Paths and constants shared by training, evaluation and real-time scoring."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = Path.home() / "Downloads" / "fraudTrain.csv"
DEFAULT_TEST = Path.home() / "Downloads" / "fraudTest.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "is_fraud"
RANDOM_STATE = 42

# Transactions on or after this date are held out of training for model selection
# and threshold tuning. The split is by time because fraud detection is always
# scored on transactions that happen after the ones the model learned from.
VALIDATION_START = "2020-03-01"

# Look-back windows (seconds) for per-card behaviour features.
WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800, "30d": 2_592_000}
MAX_WINDOW = WINDOWS["30d"]

NIGHT_HOURS = (22, 23, 0, 1, 2, 3)

CATEGORICAL = ["category"]

# What is known about the transaction itself at authorisation time.
TRANSACTION_FEATURES = [
    "amt",
    "hour",
    "day_of_week",
    "is_night",
    "category",
    "age",
    "city_pop",
    "dist_km",
]

# How this transaction compares with the same card's own past. Every one of these
# uses only transactions that happened strictly before the one being scored.
BEHAVIOUR_FEATURES = [
    "secs_since_prev",
    "prev_amt",
    "cnt_1h",
    "cnt_24h",
    "cnt_7d",
    "cnt_30d",
    "amt_sum_1h",
    "amt_sum_24h",
    "amt_sum_7d",
    "amt_ratio_30d",
    "amt_ratio_hist",
    "amt_z_hist",
    "amt_ratio_max",
    "vel_ratio_24h",
    "cat_share_hist",
    "night_share_hist",
]

FEATURES = TRANSACTION_FEATURES + BEHAVIOUR_FEATURES
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
