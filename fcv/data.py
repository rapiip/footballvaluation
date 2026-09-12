"""Pembersihan dan pengayaan dataset.

Prinsip: tidak ada baris yang dibuang tanpa alasan. Versi lama app hanya
memakai 8.746 dari 26.397 baris (drop_duplicates by short_name); di sini
seluruh snapshot dipakai karena tiap baris adalah satu musim/edisi pemain,
yang justru berguna untuk kurva umur dan lini waktu karier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ratings
from .config import (CARD_STATS, CLUB_LEAGUE_FIX, DATA_CSV, GROUP_LABEL,
                     LEAGUE_SHORT, POSITION_GROUP, TOP5_COUNTRY)

TOP5_LEAGUES = tuple(TOP5_COUNTRY.keys())


@dataclass
class DataQuality:
    """Ringkasan apa saja yang dibereskan, ditampilkan di panel Data Audit."""
    rows_raw: int = 0
    rows_used: int = 0
    players: int = 0
    clubs: int = 0
    leagues_before: int = 0
    leagues_after: int = 0
    clubs_relabelled: int = 0
    relabel_detail: dict[str, int] = field(default_factory=dict)
    gk_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    same_age_collisions: int = 0
    age_min: int = 0
    age_max: int = 0


def load_raw(path=DATA_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def _fix_leagues(df: pd.DataFrame, dq: DataQuality) -> pd.DataFrame:
    dq.leagues_before = df["league_name"].nunique()

    fixed_league = df["club_name"].map(lambda c: CLUB_LEAGUE_FIX.get(c, (None, None))[0])
    fixed_country = df["club_name"].map(lambda c: CLUB_LEAGUE_FIX.get(c, (None, None))[1])

    df["league_name"] = fixed_league.fillna(df["league_name"])
    df["country"] = fixed_country.fillna(df["league_name"].map(TOP5_COUNTRY))
    df["country"] = df["country"].fillna("Lainnya")
    df["league_short"] = df["league_name"].map(LEAGUE_SHORT).fillna("—")
    df["is_top5"] = df["league_name"].isin(TOP5_LEAGUES)

    relabelled = df.loc[fixed_league.notna(), "club_name"]
    dq.clubs_relabelled = int(relabelled.nunique())
    dq.relabel_detail = (
        df.loc[fixed_league.notna()].groupby("league_name")["club_name"].nunique().to_dict()
    )
    dq.leagues_after = df["league_name"].nunique()
    return df


def _derive_positions(df: pd.DataFrame) -> pd.DataFrame:
    positions = df["player_positions"].astype(str)
    df["position_main"] = positions.str.split(",").str[0].str.strip()
    df["position_all"] = positions.str.replace(", ", " / ", regex=False)
    df["position_count"] = positions.str.count(",") + 1
    df["position_group"] = df["position_main"].map(POSITION_GROUP).fillna("MID")
    df["position_group_label"] = df["position_group"].map(GROUP_LABEL)
    return df


def build_dataset(path=DATA_CSV) -> tuple[pd.DataFrame, DataQuality]:
    """Baca CSV lalu hasilkan dataframe siap pakai + laporan kualitas data."""
    raw = load_raw(path)
    dq = DataQuality(rows_raw=len(raw))

    df = raw.copy()
    for col in ("short_name", "club_name", "league_name", "player_positions"):
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()
    dq.duplicate_rows_dropped = before - len(df)

    df = _fix_leagues(df, dq)
    df = _derive_positions(df)

    # Kiper hanya 1 baris dan enam stat kartunya tidak bermakna untuk GK
    # (dataset tidak menyertakan atribut goalkeeping), jadi dikeluarkan.
    gk_mask = df["position_group"] == "GK"
    dq.gk_rows_dropped = int(gk_mask.sum())
    df = df.loc[~gk_mask].copy()

    # Identitas pemain. short_name saja ambigu (ada 3 "Danilo" berbeda);
    # kombinasi nama + posisi utama menyisakan 27 tabrakan umur dari 26 ribu baris.
    df["player_key"] = df["short_name"] + "|" + df["position_main"]
    dq.same_age_collisions = int(df.duplicated(subset=["player_key", "age"]).sum())

    df["ovr"] = ratings.estimate_ovr(df)
    df["tier"] = ratings.card_tiers(df["ovr"])

    pct_cols = CARD_STATS + ["ovr", "movement_reactions", "value_eur"]
    df = pd.concat([df, ratings.percentile_within(df, pct_cols, "position_group")], axis=1)

    df["value_m"] = df["value_eur"] / 1_000_000
    df["log_value"] = np.log(df["value_eur"])

    # Snapshot terbaru per pemain (umur tertinggi) = kartu default.
    df = df.sort_values(["player_key", "age"], kind="stable")
    df["snapshot_index"] = df.groupby("player_key").cumcount() + 1
    df["snapshot_total"] = df.groupby("player_key")["age"].transform("size")
    # Jika umur sama, pilih baris terakhir secara deterministik. Semua snapshot
    # tetap tersedia di picker; tahun edisi memang tidak tersedia di sumber.
    df["is_latest"] = df["snapshot_index"] == df["snapshot_total"]

    # Label yang enak dibaca di dropdown: nama, klub snapshot terbaru, posisi.
    latest = df.loc[df["is_latest"]].drop_duplicates("player_key").set_index("player_key")
    label = (latest["short_name"] + "  ·  " + latest["club_name"]
             + "  ·  " + latest["position_main"])
    df["player_label"] = df["player_key"].map(label)

    df = df.reset_index(drop=True)

    dq.rows_used = len(df)
    dq.players = int(df["player_key"].nunique())
    dq.clubs = int(df["club_name"].nunique())
    dq.age_min, dq.age_max = int(df["age"].min()), int(df["age"].max())
    return df, dq


def player_options(df: pd.DataFrame) -> pd.Series:
    """Mapping label -> player_key, terurut dari OVR tertinggi."""
    latest = df.loc[df["is_latest"]].sort_values("ovr", ascending=False)
    latest = latest.drop_duplicates("player_key")
    return pd.Series(latest["player_key"].to_numpy(), index=latest["player_label"].to_numpy())
