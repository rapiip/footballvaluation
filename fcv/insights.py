"""Analitik turunan: market scanner, pemain serupa, kurva umur, lini waktu karier."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CARD_STATS

SIMILARITY_COLS = CARD_STATS + ["movement_reactions", "mentality_composure", "age"]


def market_pool(df: pd.DataFrame,
                leagues: list[str] | None = None,
                groups: list[str] | None = None,
                age_range: tuple[int, int] | None = None,
                value_range: tuple[float, float] | None = None,
                min_ovr: int = 0,
                latest_only: bool = True) -> pd.DataFrame:
    """Satu sumber hasil filter. None = semua, pilihan [] = tidak ada hasil."""
    scan = df
    if latest_only:
        scan = scan.loc[scan["is_latest"]]
    if leagues is not None:
        scan = scan.loc[scan["league_name"].isin(leagues)]
    if groups is not None:
        scan = scan.loc[scan["position_group"].isin(groups)]
    if age_range:
        scan = scan.loc[scan["age"].between(*age_range)]
    if value_range:
        scan = scan.loc[scan["value_m"].between(*value_range)]
    if min_ovr:
        scan = scan.loc[scan["ovr"] >= min_ovr]
    return scan.copy()


def market_scan(df: pd.DataFrame,
                leagues: list[str] | None = None,
                groups: list[str] | None = None,
                age_range: tuple[int, int] | None = None,
                value_range: tuple[float, float] | None = None,
                min_ovr: int = 0,
                latest_only: bool = True,
                side: str = "BARGAIN",
                sort_by: str = "abs",
                limit: int = 25) -> pd.DataFrame:
    """Peringkat hanya berisi verdict yang diminta, sesudah filter pasar.

    Euro menyorot selisih nominal; persen menyorot selisih relatif.
    Gunakan market_pool untuk ringkasan semua verdict dan peta pasar.
    """
    scan = market_pool(df, leagues, groups, age_range, value_range, min_ovr, latest_only)
    scan = scan.loc[scan["verdict"] == side]

    column = "gap_eur" if sort_by == "abs" else "log_ratio"
    ascending = side == "OVERPRICED"
    return scan.sort_values(column, ascending=ascending).head(limit).copy()


def similar_players(df: pd.DataFrame, row: pd.Series, k: int = 6) -> pd.DataFrame:
    """Pemain dengan profil atribut terdekat (jarak euclidean ter-standardisasi).

    Hanya membandingkan dalam grup posisi yang sama supaya sebanding.
    """
    pool = df.loc[df["is_latest"] & (df["position_group"] == row["position_group"])]
    pool = pool.loc[pool["player_key"] != row["player_key"]]
    if pool.empty:
        return pool

    matrix = pool[SIMILARITY_COLS].to_numpy(dtype=float)
    centre = matrix.mean(axis=0)
    spread = np.where(matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0))
    target = (row[SIMILARITY_COLS].to_numpy(dtype=float) - centre) / spread
    dist = np.sqrt((((matrix - centre) / spread - target) ** 2).sum(axis=1))

    out = pool.copy()
    out["distance"] = dist
    out["similarity"] = (100 / (1 + dist)).round(1)
    return out.nsmallest(k, "distance")


def age_curve(df: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    """Median harga & OVR per umur — memanfaatkan seluruh snapshot, bukan satu per pemain."""
    scan = df if groups is None else df.loc[df["position_group"].isin(groups)]
    curve = scan.groupby("age").agg(
        median_value_m=("value_m", "median"),
        p90_value_m=("value_m", lambda s: s.quantile(0.9)),
        median_ovr=("ovr", "median"),
        snapshots=("value_m", "size"),
    ).reset_index()
    return curve.loc[curve["snapshots"] >= 20]


def peak_age(curve: pd.DataFrame) -> int:
    if curve.empty:
        return 0
    return int(curve.loc[curve["median_value_m"].idxmax(), "age"])


def career_timeline(df: pd.DataFrame, player_key: str) -> pd.DataFrame:
    """Seluruh snapshot satu pemain, terurut umur."""
    cols = ["age", "club_name", "league_name", "value_m", "pred_m", "ovr",
            "verdict", "gap_pct", "position_main"]
    return df.loc[df["player_key"] == player_key, cols].sort_values("age")


def league_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan per liga dari snapshot terbaru."""
    latest = df.loc[df["is_latest"]]
    table = latest.groupby(["league_name", "country"]).agg(
        pemain=("player_key", "nunique"),
        klub=("club_name", "nunique"),
        ovr_median=("ovr", "median"),
        umur_median=("age", "median"),
        nilai_median_m=("value_m", "median"),
        nilai_total_m=("value_m", "sum"),
        bargain=("verdict", lambda s: (s == "BARGAIN").mean() * 100),
    ).reset_index().sort_values("nilai_total_m", ascending=False)
    return table


def club_table(df: pd.DataFrame, league: str, limit: int = 20) -> pd.DataFrame:
    latest = df.loc[df["is_latest"] & (df["league_name"] == league)]
    return (latest.groupby("club_name").agg(
        pemain=("player_key", "nunique"),
        ovr_median=("ovr", "median"),
        skuad_m=("value_m", "sum"),
        umur_median=("age", "median"),
    ).reset_index().sort_values("skuad_m", ascending=False).head(limit))


def stat_leaders(df: pd.DataFrame, stat: str, limit: int = 10,
                 groups: list[str] | None = None) -> pd.DataFrame:
    latest = df.loc[df["is_latest"]]
    if groups is not None:
        latest = latest.loc[latest["position_group"].isin(groups)]
    return latest.nlargest(limit, stat)[
        list(dict.fromkeys(["short_name", "club_name", "position_main", "age", "ovr", stat, "value_m"]))
    ]


def comparison_table(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Kolom A/B tetap unik meskipun dua entri memiliki nama pendek sama."""
    from .config import ATTR_LABEL

    values_a = a[CARD_STATS].astype(int).to_numpy()
    values_b = b[CARD_STATS].astype(int).to_numpy()
    return pd.DataFrame({
        "Atribut": [ATTR_LABEL[stat] for stat in CARD_STATS],
        f"A · {a['short_name']}": values_a,
        f"B · {b['short_name']}": values_b,
        "Selisih A − B": values_a - values_b,
    })


def value_distribution(df: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    latest = df.loc[df["is_latest"]]
    counts, edges = np.histogram(np.log10(latest["value_eur"]), bins=bins)
    return pd.DataFrame({
        "value_eur": 10 ** ((edges[:-1] + edges[1:]) / 2),
        "count": counts,
    })
