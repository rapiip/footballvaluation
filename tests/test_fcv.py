"""Uji logika murni: rating, pembersihan data, verdict, dan format tampilan."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcv import data as fcv_data
from fcv import insights, ratings, ui
from fcv.config import CARD_STATS


def make_row(**over) -> dict:
    base = dict(
        short_name="A. Tester", club_name="Santos", league_name="Serie A",
        player_positions="ST, LW", value_eur=5_000_000.0, age=24,
        pace=80, movement_acceleration=80, movement_sprint_speed=80,
        movement_agility=75, movement_reactions=78, movement_balance=75,
        physic=70, power_stamina=70, power_strength=70, mentality_composure=70,
        shooting=80, attacking_finishing=80, power_shot_power=75, power_long_shots=70,
        passing=65, attacking_short_passing=70, mentality_vision=65, attacking_crossing=60,
        dribbling=80, skill_dribbling=80, skill_ball_control=80,
        defending=40, defending_standing_tackle=40, mentality_interceptions=40,
    )
    base.update(over)
    return base


def make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["position_main"] = df["player_positions"].str.split(",").str[0].str.strip()
    df["position_group"] = df["position_main"].map(
        {"ST": "ATT", "LW": "ATT", "CB": "DEF", "CM": "MID"})
    return df


# ------------------------------------------------------------------ rating
def test_position_weights_sum_to_one():
    for pos, weights in ratings.POSITION_WEIGHTS.items():
        assert set(weights) == set(CARD_STATS), pos
        assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0, pos


def test_estimate_ovr_stays_in_card_range():
    weak = make_row(**{s: 20 for s in CARD_STATS}, )
    weak["movement_reactions"] = 20
    strong = make_row(**{s: 99 for s in CARD_STATS})
    strong["movement_reactions"] = 99
    df = make_df([weak, strong, make_row()])
    ovr = ratings.estimate_ovr(df)
    assert ovr.between(ratings.OVR_MIN, ratings.OVR_MAX).all()
    assert ovr.iloc[1] > ovr.iloc[0]


def test_estimate_ovr_respects_position_weights():
    """Bek dengan defending tinggi harus kalah rating sebagai striker."""
    stopper = make_row(player_positions="CB", defending=90, physic=88, shooting=30,
                       pace=60, dribbling=45, passing=55)
    striker = dict(stopper)
    striker["player_positions"] = "ST"
    df = make_df([stopper, striker])
    ovr = ratings.estimate_ovr(df)
    assert ovr.iloc[0] > ovr.iloc[1]


def test_card_tier_boundaries():
    assert ratings.card_tier(90) == "icon"
    assert ratings.card_tier(85) == "icon"
    assert ratings.card_tier(84) == "gold"
    assert ratings.card_tier(75) == "gold"
    assert ratings.card_tier(74) == "silver"
    assert ratings.card_tier(65) == "silver"
    assert ratings.card_tier(64) == "bronze"
    assert ratings.card_tier(40) == "bronze"


def test_percentile_within_is_group_relative():
    rows = [
        make_row(player_positions="CB", defending=60),
        make_row(player_positions="CB", defending=90),
        make_row(player_positions="ST", defending=50),
    ]
    df = make_df(rows)
    pct = ratings.percentile_within(df, ["defending"], "position_group")
    assert pct.loc[1, "pct_defending"] == 100.0          # bek terbaik di grupnya
    assert pct.loc[2, "pct_defending"] == 100.0          # satu-satunya striker
    assert pct.loc[0, "pct_defending"] == 50.0


# ----------------------------------------------------------------- verdict
def test_verdict_thresholds_follow_distribution():
    lr = pd.Series(np.linspace(-1, 1, 101))
    low, high = ratings.verdict_thresholds(lr, 0.15, 0.85)
    assert low == pytest.approx(-0.7, abs=1e-9)
    assert high == pytest.approx(0.7, abs=1e-9)


def test_verdict_labels():
    assert ratings.verdict(0.5, -0.1, 0.2) == "BARGAIN"
    assert ratings.verdict(-0.5, -0.1, 0.2) == "OVERPRICED"
    assert ratings.verdict(0.0, -0.1, 0.2) == "FAIR"
    assert ratings.verdict(0.2, -0.1, 0.2) == "BARGAIN"      # batas atas inklusif


def test_verdicts_vectorised_matches_scalar():
    lr = pd.Series([-0.4, -0.05, 0.0, 0.3])
    out = ratings.verdicts(lr, -0.1, 0.2).tolist()
    assert out == [ratings.verdict(v, -0.1, 0.2) for v in lr]


def test_residual_zscore_centred_per_group():
    lr = pd.Series([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    grp = pd.Series(["ATT"] * 3 + ["DEF"] * 3)
    z = ratings.residual_zscore(lr, grp)
    assert z.iloc[1] == pytest.approx(0.0)      # median grup ATT
    assert z.iloc[4] == pytest.approx(0.0)      # median grup DEF
    assert z.iloc[2] > 0 and z.iloc[0] < 0


def test_residual_zscore_survives_zero_spread():
    lr = pd.Series([0.3, 0.3, 0.3])
    z = ratings.residual_zscore(lr, pd.Series(["MID"] * 3))
    assert (z == 0).all()


# -------------------------------------------------------------- pembersihan
def test_league_fix_moves_foreign_clubs_out_of_top5():
    dq = fcv_data.DataQuality()
    df = pd.DataFrame({
        "club_name": ["Santos", "Juventus", "Zenit St. Petersburg", "Rapid Wien", "Arsenal"],
        "league_name": ["Serie A", "Serie A", "Premier League", "Bundesliga", "Premier League"],
    })
    out = fcv_data._fix_leagues(df, dq)
    assert out.loc[0, "league_name"] == "Brasileirão Série A"
    assert out.loc[0, "country"] == "Brasil"
    assert out.loc[1, "league_name"] == "Serie A" and out.loc[1, "country"] == "Italia"
    assert out.loc[2, "league_name"] == "Russian Premier League"
    assert out.loc[3, "league_name"] == "Austrian Bundesliga"
    assert out.loc[4, "is_top5"] and not out.loc[0, "is_top5"]
    assert dq.clubs_relabelled == 3


def test_position_derivation():
    df = make_df([make_row(player_positions="CDM, CM, RB")])
    out = fcv_data._derive_positions(df)
    assert out.loc[0, "position_main"] == "CDM"
    assert out.loc[0, "position_group"] == "MID"
    assert out.loc[0, "position_count"] == 3
    assert out.loc[0, "position_all"] == "CDM / CM / RB"


# --------------------------------------------------------------- integrasi
@pytest.fixture(scope="module")
def real_dataset():
    return fcv_data.build_dataset()


def test_dataset_keeps_every_usable_row(real_dataset):
    df, dq = real_dataset
    assert dq.rows_raw == 26_397
    # hanya baris kiper yang dibuang (atribut GK tidak ada di dataset)
    assert dq.rows_used == dq.rows_raw - dq.gk_rows_dropped - dq.duplicate_rows_dropped
    assert dq.rows_used > 26_000
    assert dq.players > 11_000


def test_dataset_separates_contaminated_leagues(real_dataset):
    df, dq = real_dataset
    assert dq.leagues_before == 5
    assert dq.leagues_after == 10
    assert dq.clubs_relabelled == 73
    top5 = df.loc[df["is_top5"], "club_name"].unique()
    assert "Santos" not in top5
    assert "Zenit St. Petersburg" not in top5
    assert "Juventus" in top5
    # setelah dibersihkan, jumlah klub tiap liga top 5 masuk akal (17-22 klub)
    counts = df.loc[df["is_top5"]].groupby("league_name")["club_name"].nunique()
    assert counts.between(17, 22).all(), counts.to_dict()


def test_every_player_has_one_latest_snapshot(real_dataset):
    df, _ = real_dataset
    latest = df.loc[df["is_latest"]].drop_duplicates("player_key")
    assert len(latest) == df["player_key"].nunique()
    assert df["player_label"].notna().all()


def test_ovr_matches_reputation_of_elite_players(real_dataset):
    df, _ = real_dataset
    best = df.loc[df["is_latest"]].nlargest(25, "ovr")["short_name"].tolist()
    assert any("Messi" in n for n in best)
    assert any("Mbappé" in n for n in best)
    assert df["ovr"].max() <= 99 and df["ovr"].min() >= 40
    corr = np.corrcoef(df["ovr"], df["log_value"])[0, 1]
    assert corr > 0.85


def test_similar_players_stay_in_same_line(real_dataset):
    df, _ = real_dataset
    df = df.assign(pred_m=0.0, gap_pct=0.0)          # kolom tidak dipakai fungsi ini
    row = df.loc[df["is_latest"]].nlargest(1, "ovr").iloc[0]
    out = insights.similar_players(df, row, k=5)
    assert len(out) == 5
    assert (out["position_group"] == row["position_group"]).all()
    assert row["player_key"] not in out["player_key"].tolist()
    assert out["similarity"].is_monotonic_decreasing


def test_market_scan_sort_modes_differ():
    """Urutan persen menyorot pemain murah, urutan euro menyorot yang material."""
    base = make_row()
    rows = []
    for i, (value, pred) in enumerate([(120_000, 1_050_000), (73_000_000, 103_500_000)]):
        row = dict(base)
        row.update(short_name=f"P{i}", value_eur=float(value))
        rows.append(row)
    df = make_df(rows)
    df["is_latest"] = True
    df["gap_eur"] = [930_000.0, 30_500_000.0]
    df["log_ratio"] = np.log([1_050_000 / 120_000, 103_500_000 / 73_000_000])
    df["value_m"] = df["value_eur"] / 1e6

    by_pct = insights.market_scan(df, side="BARGAIN", sort_by="pct", limit=1)
    by_abs = insights.market_scan(df, side="BARGAIN", sort_by="abs", limit=1)
    assert by_pct.iloc[0]["short_name"] == "P0"
    assert by_abs.iloc[0]["short_name"] == "P1"


def test_age_curve_has_a_plausible_peak(real_dataset):
    df, _ = real_dataset
    curve = insights.age_curve(df)
    peak = insights.peak_age(curve)
    assert 22 <= peak <= 30, peak
    assert curve["age"].is_monotonic_increasing


# ---------------------------------------------------------------- tampilan
@pytest.mark.parametrize("value,expected", [
    (181_500_000, "€181,5 jt"),
    (2_600_000, "€2,6 jt"),
    (850_000, "€850 rb"),
    (1_450_000_000, "€1,45 M"),
    (-15_000_000, "−€15,0 jt"),
    (-420_000, "−€420 rb"),
])
def test_money_format_indonesian(value, expected):
    assert ui.money(value) == expected


def test_mini_card_shows_signed_gap():
    row = pd.Series(make_row(short_name="R. Sterling", club_name="Manchester City") | {
        "ovr": 87, "tier": "gold", "position_main": "RW", "league_short": "EPL",
        "gap_eur": 30_500_000.0, "gap_pct": 42.0,
    })
    html = ui.mini_card(row)
    assert "+€30,5 jt" in html and "+42%" in html and "gap-up" in html
    assert "vs AI" not in html          # label berulang dihapus, sudah dijelaskan di seksi
    row["gap_eur"], row["gap_pct"] = -47_700_000.0, -44.0
    html = ui.mini_card(row)
    assert "−€47,7 jt" in html and "gap-down" in html


def test_gap_bar_grows_with_gap_and_flips_side():
    """Bar selisih: nol di tengah, kanan untuk positif, kiri untuk negatif."""
    small = ui._gap_bar(6.0)
    big = ui._gap_bar(60.0)
    negative = ui._gap_bar(-30.0)

    assert "left:50%" in small and "left:50%" in big
    assert "right:50%" in negative and 'class="neg"' in negative

    def width(html: str) -> float:
        return float(html.split("width:")[1].split("%")[0])

    assert width(small) < width(big)
    assert width(big) == pytest.approx(50.0)      # dipatok setengah lebar
    assert width(ui._gap_bar(500.0)) == pytest.approx(50.0)
    assert width(ui._gap_bar(0.0)) == pytest.approx(ui.GAP_BAR_MIN)   # tetap terlihat
    assert width(ui._gap_bar(1.0)) >= ui.GAP_BAR_MIN


def test_tiles_support_muted_variant():
    normal = ui.tiles([("R² holdout", "0.868", "5.323 baris uji")])
    muted = ui.tiles([("R² in-sample", "0.973", "angka yang menipu", True)])
    assert "fc-tile--muted" not in normal
    assert "fc-tile--muted" in muted


def test_tab_intro_and_section_are_distinct_levels():
    """Judul tab tidak boleh diduplikasi sebagai judul seksi."""
    source = (Path(ui.__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    for tab_name in ("Market scanner", "Head to head", "Data audit"):
        assert f'ui.section("{tab_name}"' not in source


def test_pct_and_initials():
    assert ui.pct(38.24) == "+38,2%"
    assert ui.pct(-9.0, 0) == "−9%"          # minus tipografis, bukan hyphen
    assert ui.num(-0.55, 2) == "−0,55"
    assert ui.count(26396) == "26.396"
    assert ui.initials("K. Mbappé") == "KM"
    assert ui.initials("Neymar") == "NE"
    assert ui.initials("") == "?"


def test_player_card_html_is_single_line_and_escaped():
    row = pd.Series(make_row(short_name="<script>x</script>", club_name="Santos") | {
        "ovr": 88, "tier": "gold", "position_main": "ST", "league_short": "BRA",
        "verdict": "BARGAIN",
    })
    html = ui.player_card(row)
    assert "\n" not in html
    assert "<script>" not in html
    assert "fc-card--gold" in html
    assert ">88<" in html
