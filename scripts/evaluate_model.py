"""Evaluasi kejujuran model: holdout yang dipisah per pemain.

Model yang di-ship (`model_rf_market_value.pkl`) dilatih memakai seluruh
baris CSV, jadi skor apa pun yang dihitung di data yang sama adalah
in-sample dan pasti terlihat bagus (R² 0.97). Skrip ini melatih ulang
Random Forest dengan hyperparameter yang sama pada 80% pemain lalu menguji
di 20% pemain yang belum pernah dilihat.

Pemisahan dibuat per `player_key`, bukan per baris, karena satu pemain
punya beberapa snapshot umur. Kalau dipisah per baris, snapshot umur 23
bisa masuk train dan umur 24 masuk test — nilainya hampir identik dan
skor jadi menipu.

Jalankan:  python scripts/evaluate_model.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fcv import data as fcv_data  # noqa: E402
from fcv import model as fcv_model  # noqa: E402
from fcv.config import METRICS_JSON  # noqa: E402

TEST_SIZE = 0.2
SEED = 42


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ape = np.abs(y_pred - y_true) / y_true * 100
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "r2_log": round(float(r2_score(np.log(y_true), np.log(np.clip(y_pred, 1, None)))), 4),
        "mae_eur": int(mean_absolute_error(y_true, y_pred)),
        "median_ape_pct": round(float(np.median(ape)), 2),
        "p90_ape_pct": round(float(np.percentile(ape, 90)), 2),
        "within_20pct": round(float((ape <= 20).mean() * 100), 1),
        "n": int(len(y_true)),
    }


def main() -> None:
    print("memuat data...")
    df, dq = fcv_data.build_dataset()
    shipped = fcv_model.load_model()
    features = fcv_model.feature_names(shipped)

    X, y, groups = df[features], df["value_eur"].to_numpy(), df["player_key"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    print(f"train {len(train_idx):,} baris / test {len(test_idx):,} baris "
          f"({groups.iloc[test_idx].nunique():,} pemain uji)")

    print("melatih ulang Random Forest pembanding...")
    started = time.time()
    fresh = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    fresh.fit(X.iloc[train_idx], y[train_idx])
    elapsed = round(time.time() - started, 1)

    honest = score(y[test_idx], fresh.predict(X.iloc[test_idx]))
    shipped_on_test = score(y[test_idx], shipped.predict(X.iloc[test_idx]))
    shipped_on_all = score(y, shipped.predict(X))

    baseline_pred = np.full(len(test_idx), np.median(y[train_idx]))
    baseline = score(y[test_idx], baseline_pred)

    per_group = []
    test_df = df.iloc[test_idx].copy()
    test_df["pred"] = fresh.predict(X.iloc[test_idx])
    for name, part in test_df.groupby("position_group"):
        stats = score(part["value_eur"].to_numpy(), part["pred"].to_numpy())
        per_group.append({"position_group": name, **stats})

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rows_used": dq.rows_used,
        "players": dq.players,
        "features": features,
        "split": {"type": "GroupShuffleSplit by player_key", "test_size": TEST_SIZE, "seed": SEED},
        "train_seconds": elapsed,
        "holdout_fresh_model": honest,
        "shipped_model_on_test_rows": shipped_on_test,
        "shipped_model_on_all_rows": shipped_on_all,
        "baseline_median": baseline,
        "per_position_group": per_group,
        "catatan": ("holdout_fresh_model adalah satu-satunya angka out-of-sample. "
                    "Dua entri shipped_model_* dihitung di data latihnya sendiri, "
                    "jadi optimistis dan hanya untuk pembanding."),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== HASIL ===")
    print(f"holdout (model baru, pemain belum terlihat) : R2 {honest['r2']:.3f} | "
          f"median APE {honest['median_ape_pct']:.1f}% | dalam ±20%: {honest['within_20pct']:.0f}%")
    print(f"model ter-ship di baris uji (in-sample)     : R2 {shipped_on_test['r2']:.3f} | "
          f"median APE {shipped_on_test['median_ape_pct']:.1f}%")
    print(f"baseline median harga                      : R2 {baseline['r2']:.3f} | "
          f"median APE {baseline['median_ape_pct']:.1f}%")
    print(f"\ntersimpan -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
