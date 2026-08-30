"""Layer rating: estimasi OVR ala EA FC, tier kartu, percentile, dan verdict harga.

Semua fungsi di sini murni (tanpa I/O) supaya bisa diuji langsung.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CARD_STATS, TIER_BINS, VERDICT_HIGH_Q, VERDICT_LOW_Q)

# Bobot enam stat kartu per posisi utama. Disusun mengikuti pola bobot
# atribut FIFA/EA FC (bek dinilai dari defending+physic, winger dari
# pace+dribbling, dst). Setiap baris berjumlah 1.0.
POSITION_WEIGHTS: dict[str, dict[str, float]] = {
    "ST":  dict(pace=.18, shooting=.32, passing=.08, dribbling=.18, defending=.03, physic=.21),
    "CF":  dict(pace=.18, shooting=.28, passing=.14, dribbling=.22, defending=.03, physic=.15),
    "LW":  dict(pace=.24, shooting=.22, passing=.14, dribbling=.28, defending=.03, physic=.09),
    "RW":  dict(pace=.24, shooting=.22, passing=.14, dribbling=.28, defending=.03, physic=.09),
    "LM":  dict(pace=.22, shooting=.16, passing=.22, dribbling=.24, defending=.07, physic=.09),
    "RM":  dict(pace=.22, shooting=.16, passing=.22, dribbling=.24, defending=.07, physic=.09),
    "CAM": dict(pace=.14, shooting=.20, passing=.26, dribbling=.26, defending=.05, physic=.09),
    "CM":  dict(pace=.10, shooting=.13, passing=.28, dribbling=.20, defending=.16, physic=.13),
    "CDM": dict(pace=.07, shooting=.06, passing=.22, dribbling=.12, defending=.33, physic=.20),
    "LWB": dict(pace=.22, shooting=.05, passing=.19, dribbling=.16, defending=.24, physic=.14),
    "RWB": dict(pace=.22, shooting=.05, passing=.19, dribbling=.16, defending=.24, physic=.14),
    "LB":  dict(pace=.19, shooting=.04, passing=.16, dribbling=.12, defending=.32, physic=.17),
    "RB":  dict(pace=.19, shooting=.04, passing=.16, dribbling=.12, defending=.32, physic=.17),
    "CB":  dict(pace=.08, shooting=.02, passing=.07, dribbling=.05, defending=.45, physic=.33),
    "GK":  dict(pace=.10, shooting=.10, passing=.20, dribbling=.10, defending=.20, physic=.30),
}
DEFAULT_WEIGHTS = dict(pace=.17, shooting=.17, passing=.17, dribbling=.17, defending=.16, physic=.16)

# Dataset tidak punya kolom `overall`, jadi OVR direkonstruksi.
# `movement_reactions` adalah proksi overall terkuat di data FIFA
# (korelasi 0.83 terhadap log harga), sisanya dari komposit posisi.
# REACTION_BLEND & CONTRAST_GAIN dikalibrasi supaya puncak distribusi
# mendarat di kisaran 91-94 (Messi/Ronaldo/Mbappé) seperti kartu aslinya.
REACTION_BLEND = 0.45
CONTRAST_GAIN = 1.12
OVR_MIN, OVR_MAX = 40, 99


def position_weight_matrix(positions: pd.Series) -> np.ndarray:
    """Matriks bobot (n_baris x 6) sesuai posisi utama tiap baris."""
    rows = [POSITION_WEIGHTS.get(p, DEFAULT_WEIGHTS) for p in positions]
    return np.array([[w[s] for s in CARD_STATS] for w in rows], dtype=float)


def estimate_ovr(df: pd.DataFrame, position_col: str = "position_main") -> pd.Series:
    """Estimasi rating keseluruhan (OVR) skala EA FC untuk setiap baris."""
    weights = position_weight_matrix(df[position_col])
    composite = (df[CARD_STATS].to_numpy(dtype=float) * weights).sum(axis=1)
    blended = REACTION_BLEND * df["movement_reactions"].to_numpy(dtype=float) + (
        1 - REACTION_BLEND
    ) * composite
    centre = blended.mean()
    stretched = centre + (blended - centre) * CONTRAST_GAIN
    return pd.Series(
        np.clip(np.rint(stretched), OVR_MIN, OVR_MAX).astype(int), index=df.index
    )


def card_tier(ovr: int) -> str:
    """Tier kartu (icon/gold/silver/bronze) dari OVR."""
    for threshold, name in TIER_BINS:
        if ovr >= threshold:
            return name
    return TIER_BINS[-1][1]


def card_tiers(ovr: pd.Series) -> pd.Series:
    return ovr.map(card_tier)


def percentile_within(df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
    """Percentile 0-100 tiap kolom dibanding pemain segrup posisi.

    Dipakai untuk radar: angka 88 defending pada bek dan pada penyerang
    punya arti sangat berbeda, percentile menyetarakan konteksnya.
    """
    out = {}
    grouped = df.groupby(group_col, observed=True)
    for col in cols:
        out[f"pct_{col}"] = grouped[col].rank(pct=True) * 100
    return pd.DataFrame(out, index=df.index)


def verdict_thresholds(log_ratio: pd.Series,
                       low_q: float = VERDICT_LOW_Q,
                       high_q: float = VERDICT_HIGH_Q) -> tuple[float, float]:
    """Ambang verdict dari sebaran residual model, bukan angka flat.

    Model punya bias +5.8% dan sebaran lebar; memakai ±10% flat membuat
    hampir separuh pemain ter-flag hanya karena derau model.
    """
    return float(log_ratio.quantile(low_q)), float(log_ratio.quantile(high_q))


def verdict(log_ratio: float, low: float, high: float) -> str:
    """BARGAIN / FAIR / OVERPRICED untuk satu baris."""
    if log_ratio >= high:
        return "BARGAIN"
    if log_ratio <= low:
        return "OVERPRICED"
    return "FAIR"


def verdicts(log_ratio: pd.Series, low: float, high: float) -> pd.Series:
    labels = np.where(log_ratio >= high, "BARGAIN",
                      np.where(log_ratio <= low, "OVERPRICED", "FAIR"))
    return pd.Series(labels, index=log_ratio.index)


def residual_zscore(log_ratio: pd.Series, group: pd.Series) -> pd.Series:
    """Z-score residual di dalam grup posisi (robust: median & MAD)."""
    frame = pd.DataFrame({"lr": log_ratio, "g": group})
    med = frame.groupby("g", observed=True)["lr"].transform("median")
    mad = frame.groupby("g", observed=True)["lr"].transform(
        lambda s: (s - s.median()).abs().median()
    )
    scale = (mad * 1.4826).replace(0, np.nan)
    return ((frame["lr"] - med) / scale).fillna(0.0)
