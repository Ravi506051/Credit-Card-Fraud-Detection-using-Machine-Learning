"""Replay test transactions through the real-time scorer, one at a time.

    python realtime_demo.py [--n 20000] [--train PATH] [--test PATH]

1. Warm up per-card state from the training-period history (no scoring).
2. Stream the first N test transactions through `RealTimeScorer.score`, the way
   an authorisation system would call it.
3. Report latency, the decisions made, and a few alerts with their reasons.
4. Check the streamed features and scores against the batch pipeline used for
   training, to prove the two code paths agree.

Run `python train.py` first to create the model bundle.
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from fraud import modeling as M
from fraud.config import DEFAULT_TEST, DEFAULT_TRAIN, FEATURES, TARGET
from fraud.features import build_features, load_raw
from fraud.realtime import RealTimeScorer

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="historical transactions used to warm up card state")
    parser.add_argument("--test", default=DEFAULT_TEST, help="transactions to stream")
    parser.add_argument("--n", type=int, default=20_000, help="how many test transactions to stream (0 = all)")
    args = parser.parse_args()

    scorer = RealTimeScorer.load()
    history = load_raw(args.train)
    stream = load_raw(args.test)
    if args.n:
        stream = stream.head(args.n)

    started = time.perf_counter()
    scorer.warm_up(history)
    print(f"Warmed up state for {len(scorer.cards):,} cards from {len(history):,} historical transactions in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    results = [scorer.score(row._asdict()) for row in stream.itertuples(index=False)]
    elapsed = time.perf_counter() - started
    probability = np.array([r["fraud_probability"] for r in results])
    latency = np.array([r["latency_ms"] for r in results])

    print(f"\nStreamed {len(stream):,} transactions in {elapsed:.1f}s  ({len(stream) / elapsed:,.0f} transactions/s, single thread)")
    print("Latency per transaction (features + model + state update), ms: " + "  ".join(f"p{q} {np.percentile(latency, q):.2f}" for q in (50, 95, 99)) + f"  max {latency.max():.2f}")

    m = M.threshold_metrics(stream[TARGET], probability, scorer.threshold)
    print(f"\nDecisions at threshold {scorer.threshold:.3f}: {m['tp']} frauds caught, {m['fn']} missed, {m['fp']} false alarms among {m['fp'] + m['tn']:,} legitimate transactions")
    print(f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")

    print("\nFirst alerts raised:")
    flagged = [i for i, r in enumerate(results) if r["flagged"]][:5]
    for i in flagged:
        txn, r = stream.iloc[i], results[i]
        actual = "FRAUD" if txn[TARGET] else "legitimate"
        print(f"  {txn['trans_date_trans_time']}  card ...{str(txn['cc_num'])[-4:]}  ${txn['amt']:>8,.2f}  {txn['category']:<14} P(fraud)={r['fraud_probability']:.3f}  actual: {actual}")
        for reason in r["reasons"]:
            print(f"      - {reason}")

    # Parity with the batch pipeline: same features, same scores.
    batch = build_features(pd.concat([history, stream], ignore_index=True), categories=scorer.categories).tail(len(stream))
    X_batch = M.tree_matrix(batch).to_numpy(np.float64)
    X_stream = np.vstack([r["features"] for r in results])
    feature_gap = np.abs(X_stream - X_batch) / np.maximum(1.0, np.abs(X_batch))
    score_gap = np.abs(probability - scorer.booster.predict(X_batch))
    worst = FEATURES[int(feature_gap.max(axis=0).argmax())]
    decisions_differ = int(((probability >= scorer.threshold) != (scorer.booster.predict(X_batch) >= scorer.threshold)).sum())
    print("\nParity with the batch (training) pipeline:")
    print(f"  largest relative feature difference: {feature_gap.max():.1e}  (feature: {worst})")
    print(f"  largest fraud-probability difference: {score_gap.max():.1e}")
    print(f"  decisions that differ: {decisions_differ} of {len(stream):,}")


if __name__ == "__main__":
    main()
