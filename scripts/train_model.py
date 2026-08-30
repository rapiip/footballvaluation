"""Latih ulang model valuasi dari CSV — jaring pengaman kalau berkas .pkl hilang.

Model yang ada di repo (`model_rf_market_value.pkl`) sudah cukup untuk
menjalankan app. Skrip ini berguna kalau berkasnya hilang, rusak, atau kamu
ingin bereksperimen dengan hyperparameter lain.

Catatan kejujuran: hasilnya tidak akan identik byte-per-byte dengan model
asli, karena seed dan versi scikit-learn saat training pertama tidak
tercatat. Secara statistik setara — silakan cek dengan
`python scripts/evaluate_model.py` sesudahnya.

Jalankan:  python scripts/train_model.py
           python scripts/train_model.py --out model_eksperimen.pkl --trees 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fcv.config import DATA_CSV, MODEL_PKL  # noqa: E402

# 25 fitur yang dipakai model asli, urutannya penting.
FEATURES = [
    "age", "pace", "movement_acceleration", "movement_sprint_speed",
    "movement_agility", "movement_reactions", "movement_balance",
    "physic", "power_stamina", "power_strength", "mentality_composure",
    "shooting", "attacking_finishing", "power_shot_power", "power_long_shots",
    "passing", "attacking_short_passing", "mentality_vision", "attacking_crossing",
    "dribbling", "skill_dribbling", "skill_ball_control",
    "defending", "defending_standing_tackle", "mentality_interceptions",
]
TARGET = "value_eur"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(MODEL_PKL), help="berkas model keluaran")
    ap.add_argument("--trees", type=int, default=100, help="jumlah pohon (n_estimators)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compress", type=int, default=3,
                    help="level kompresi joblib; 0 = tanpa kompresi (~153 MB), 3 = ~28 MB")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(DATA_CSV)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    print(f"data  : {len(df):,} baris × {len(FEATURES)} fitur")

    started = time.time()
    model = RandomForestRegressor(n_estimators=args.trees, random_state=args.seed, n_jobs=-1)
    model.fit(X, y)
    print(f"latih : {time.time() - started:.1f} detik, {args.trees} pohon")

    pred = model.predict(X)
    ape = np.abs(pred - y) / y * 100
    print(f"skor in-sample (bukan ukuran generalisasi!): R² {r2_score(y, pred):.3f} | "
          f"median error {np.median(ape):.1f}%")

    started = time.time()
    joblib.dump(model, args.out, compress=args.compress)
    size = Path(args.out).stat().st_size / 1024 / 1024
    print(f"simpan: {args.out} ({size:.1f} MB, compress={args.compress}, "
          f"{time.time() - started:.1f} detik)")
    print("\nLangkah berikutnya: python scripts/evaluate_model.py "
          "untuk mengukur akurasi out-of-sample yang jujur.")


if __name__ == "__main__":
    main()
