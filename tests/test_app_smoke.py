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


def test_sidebar_controls_exist(app: AppTest):
    assert any(s.label == "Cari pemain" for s in app.sidebar.selectbox)
    assert any(s.label == "Liga" for s in app.sidebar.multiselect)
    assert any(b.label == "Reset filter pasar" for b in app.sidebar.button)


def test_charts_and_cards_rendered(app: AppTest):
    # lima chart utama + kartu HTML
    assert len(app.get("plotly_chart")) >= 5
    html = " ".join(m.value for m in app.markdown)
    assert "fc-card" in html and "fc-verdict" in html and "fc-tile" in html


def test_switching_player_keeps_app_healthy(app: AppTest):
    app.sidebar.selectbox[0].select_index(5).run()
    assert not app.exception, [e.value for e in app.exception]


def test_same_player_comparison_shows_guidance(app: AppTest):
    comparison_selects = [s for s in app.selectbox if s.label in {"Pemain A", "Pemain B"}]
    comparison_selects[1].select(comparison_selects[0].value).run()
    assert any("dua pemain berbeda" in warning.value for warning in app.warning)
