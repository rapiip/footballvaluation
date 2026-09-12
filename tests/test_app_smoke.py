"""Smoke test: jalankan app.py penuh lewat AppTest dan pastikan tidak ada exception.

Lambat (memuat model 157 MB), tapi ini satu-satunya cara memverifikasi seluruh
tab benar-benar ter-render tanpa error.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    return at


def test_app_runs_without_exception(app: AppTest):
    assert not app.exception, [e.value for e in app.exception]


def test_all_tabs_present(app: AppTest):
    labels = [t.label for t in app.tabs]
    assert labels == ["PLAYER HUB", "MARKET SCANNER", "HEAD TO HEAD", "INSIGHTS",
                      "DATA AUDIT"]


def test_controls_belong_to_their_results(app: AppTest):
    assert any(s.label == "Cari pemain" for s in app.tabs[0].selectbox)
    assert any(s.label == "Liga" for s in app.tabs[1].multiselect)
    assert any(b.label == "Reset filter pasar" for b in app.tabs[1].button)
    assert any(s.label == "Lini untuk analisis" for s in app.tabs[3].multiselect)


def test_charts_and_cards_rendered(app: AppTest):
    # lima chart utama + kartu HTML
    assert len(app.get("plotly_chart")) >= 5
    html = " ".join(m.value for m in app.markdown)
    assert "fc-card" in html and "fc-verdict" in html and "fc-tile" in html


def test_switching_player_keeps_app_healthy(app: AppTest):
    app.selectbox(key="hub_player").select_index(5).run()
    assert not app.exception, [e.value for e in app.exception]


def test_same_player_comparison_shows_guidance(app: AppTest):
    comparison_selects = [s for s in app.selectbox if s.label in {"Pemain A", "Pemain B"}]
    comparison_selects[1].select(comparison_selects[0].value).run()
    assert any("dua pemain berbeda" in warning.value for warning in app.warning)
    comparison_selects[1].select_index(1).run()


@pytest.mark.parametrize("empty_key", ["market_leagues", "market_groups"])
def test_empty_market_selection_and_reset(app: AppTest, empty_key: str):
    player_before = app.selectbox(key="hub_player").value
    insight_before = app.multiselect(key="insight_groups").value.copy()
    app.multiselect(key=empty_key).set_value([]).run()
    scanner = app.tabs[1]
    assert not app.exception
    assert any("0 pemain cocok" in m.value for m in scanner.markdown)
    assert not scanner.get("plotly_chart") and not scanner.dataframe
    assert any("Belum ada pemain" in item.value for item in scanner.info)
    app.button(key="empty_reset").click().run()
    assert not app.exception
    assert len(app.multiselect(key="market_leagues").value) == 5
    assert len(app.multiselect(key="market_groups").value) == 3
    assert app.selectbox(key="hub_player").value == player_before
    assert app.multiselect(key="insight_groups").value == insight_before
    assert app.tabs[1].get("plotly_chart")


def test_no_matches_at_extreme_rating_and_reset(app: AppTest):
    app.slider(key="market_ovr").set_value(99).run()
    assert any("0 pemain cocok" in m.value for m in app.tabs[1].markdown)
    app.button(key="empty_reset").click().run()
    assert app.slider(key="market_ovr").value == 60
    assert not app.exception


def test_insights_empty_selection_and_ovr_leaderboard(app: AppTest):
    app.multiselect(key="insight_groups").set_value([]).run()
    assert any("Pilih minimal satu lini" in i.value for i in app.tabs[3].info)
    assert not app.exception
    app.multiselect(key="insight_groups").set_value(["DEF", "MID", "ATT"]).run()
    app.selectbox(key="stat_pick").set_value("ovr").run()
    assert not app.exception
    leaders = app.tabs[3].dataframe[0].value
    assert leaders.columns.is_unique
    app.selectbox(key="stat_pick").select_index(0).run()
