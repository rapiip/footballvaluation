"""Pembungkus model Random Forest: load, prediksi massal, dan metrik jujur."""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from . import ratings
from .config import METRICS_JSON, MODEL_PKL


def load_model(path=MODEL_PKL):
    """Muat model. Warning versi scikit-learn ditekan agar UI tetap bersih."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise AttributeError("Model tidak menyimpan nama fitur.")


def attach_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Prediksi seluruh baris sekaligus (26 ribu baris ~0.1 detik).

    Ini yang membuka fitur market scanner: valuasi tidak lagi dihitung
    untuk satu pemain terpilih saja, tapi untuk seluruh dataset.
    """
    features = feature_names(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(df[features])

    out = df.copy()
    out["pred_eur"] = pred
    out["pred_m"] = out["pred_eur"] / 1_000_000
    out["gap_eur"] = out["pred_eur"] - out["value_eur"]
    out["gap_pct"] = out["gap_eur"] / out["value_eur"] * 100
    out["log_ratio"] = np.log(out["pred_eur"] / out["value_eur"])

    low, high = ratings.verdict_thresholds(out["log_ratio"])
    out["verdict"] = ratings.verdicts(out["log_ratio"], low, high)
    out["residual_z"] = ratings.residual_zscore(out["log_ratio"], out["position_group"])
    out.attrs["verdict_low"] = low
    out.attrs["verdict_high"] = high
    out.attrs["features"] = features
    return out


def importance_table(model) -> pd.DataFrame:
    names = feature_names(model)
    imp = pd.DataFrame({
        "feature": names,
        "importance": model.feature_importances_,
    })
    imp["share"] = imp["importance"] / imp["importance"].sum() * 100
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def load_metrics(path=METRICS_JSON) -> dict | None:
    """Metrik holdout hasil scripts/evaluate_model.py (kalau sudah dijalankan)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
