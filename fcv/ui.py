"""Komponen tampilan: kartu HTML bergaya EA FC + chart Plotly bertema gelap.

Semua HTML dirakit sebagai satu baris tanpa indentasi, karena Streamlit
memproses markdown lebih dulu (indentasi 4 spasi akan jadi blok kode).
"""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import (CARD_STAT_LABEL, CARD_STATS, CSS_FILE, GROUP_LABEL,
                     TIER_LABEL)

GREEN = "#00f5a0"
CYAN = "#19d3fa"
RED = "#ff5a78"
DIM = "#93a3c2"
DIM_2 = "#6f7f9e"
INK = "#eef3ff"
LIME = "#c9ff2e"
# Display font hanya untuk teks >= 16px (judul, angka besar). Tick dan label
# chart memakai body font karena font condensed pada 11-12px sulit dibaca.
DISPLAY_FONT = "Barlow Semi Condensed, Arial Narrow, sans-serif"
BODY_FONT = "Inter, Segoe UI, system-ui, sans-serif"

# Modebar Plotly (kamera, zoom, pan) menumpuk judul & legenda di kanan atas,
# jadi disembunyikan; interaksi hover/zoom masih jalan.
CHART_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}

VERDICT_TEXT = {
    "BARGAIN": ("BARGAIN", "Model menilai pemain ini lebih mahal dari harga pasarnya — kandidat incaran."),
    "FAIR": ("FAIR VALUE", "Harga pasar berada di rentang wajar menurut model."),
    "OVERPRICED": ("OVERPRICED", "Harga pasar di atas estimasi model — hati-hati membayar premi."),
}


# ---------------------------------------------------------------- utilitas
def inject_css() -> None:
    css = CSS_FILE.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def money(eur: float) -> str:
    """Format ala Indonesia: koma desimal, satuan juta/miliar, minus di depan €."""
    if eur is None or pd.isna(eur):
        return "—"
    sign = "−" if eur < 0 else ""
    juta = abs(eur) / 1_000_000
    if juta >= 1000:
        body = f"€{juta / 1000:,.2f} M".replace(",", "@").replace(".", ",").replace("@", ".")
    elif juta >= 1:
        body = f"€{juta:,.1f} jt".replace(",", "@").replace(".", ",").replace("@", ".")
    else:
        body = f"€{abs(eur) / 1000:,.0f} rb".replace(",", ".")
    return sign + body


def pct(x: float, digits: int = 1) -> str:
    """Persen gaya Indonesia dengan tanda minus tipografis (−), bukan hyphen."""
    if x is None or pd.isna(x):
        return "—"
    body = f"{abs(x):.{digits}f}".replace(".", ",")
    return f"{'+' if x >= 0 else '−'}{body}%"


def num(x: float, digits: int = 1) -> str:
    body = f"{abs(x):,.{digits}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return body if x >= 0 else f"−{body}"


def count(n: int) -> str:
    """Pemisah ribuan titik, sesuai kaidah Indonesia (26.396, bukan 26,396)."""
    return f"{int(n):,}".replace(",", ".")


def initials(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def gap_class(value: float) -> str:
    if value > 3:
        return "gap-up"
    if value < -3:
        return "gap-down"
    return "gap-flat"


# ---------------------------------------------------------------- komponen
def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="fc-sec"><span class="fc-sec__bar"></span>'
        f'<span class="fc-sec__title">{escape(title)}</span>'
        f'<span class="fc-sec__note">{escape(note)}</span></div>',
        unsafe_allow_html=True,
    )


def tab_intro(html: str) -> None:
    """Kalimat pembuka tab.

    Menggantikan judul seksi yang tadinya mengulang nama tab persis di
    atasnya (tab "MARKET SCANNER" + judul "MARKET SCANNER"), sehingga
    hierarkinya jadi tab > seksi > isi, bukan dua judul sederajat.
    """
    st.markdown(f'<div class="fc-intro">{html}</div>', unsafe_allow_html=True)


def app_header(dq, model_note: str) -> None:
    st.markdown(
        '<div class="fc-top"><div class="fc-top__mark">FCV</div><div>'
        '<div class="fc-top__title">Football Valuation Centre</div>'
        '<div class="fc-top__sub">Scouting &amp; market value intelligence</div></div>'
        '<div class="fc-top__spacer"></div>'
        f'<div class="fc-top__pill">Snapshot <b>{count(dq.rows_used)}</b></div>'
        f'<div class="fc-top__pill">Pemain <b>{count(dq.players)}</b></div>'
        f'<div class="fc-top__pill">Liga <b>{dq.leagues_after}</b></div>'
        f'<div class="fc-top__pill">Error model <b>{escape(model_note)}</b></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def grid(cards: list[str], columns: int = 3) -> str:
    """Bungkus kartu dalam satu CSS grid.

    Dipakai agar tinggi tiap baris rata: kalau tiap kartu dirender sebagai
    st.columns terpisah, tile dengan jumlah baris berbeda membuat tampilan
    bertangga.
    """
    return (f'<div class="fc-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr))">'
            f'{"".join(cards)}</div>')


def stack(cards: list[str]) -> str:
    """Daftar kartu vertikal dengan jarak seragam 9px."""
    return f'<div class="fc-stack">{"".join(cards)}</div>'


def player_card(row: pd.Series, show_ribbon: bool = True) -> str:
    tier = row["tier"]
    stats = "".join(
        f'<div class="fc-stat"><span class="v">{int(row[s])}</span>'
        f'<span class="l">{CARD_STAT_LABEL[s]}</span></div>'
        for s in CARD_STATS
    )
    ribbon = ""
    if show_ribbon and "verdict" in row:
        key = str(row["verdict"]).lower()
        ribbon = (f'<div class="fc-card__ribbon fc-card__ribbon--{key}">'
                  f'{escape(VERDICT_TEXT[row["verdict"]][0])}</div>')
    return (
        f'<article class="fc-card fc-card--{tier}" aria-label="Kartu pemain '
        f'{escape(str(row["short_name"]))}, OVR {int(row["ovr"])}">{ribbon}'
        f'<div class="fc-card__body"><div class="fc-card__shine"></div>'
        f'<div class="fc-card__head"><div class="fc-card__id">'
        f'<div class="fc-card__ovr">{int(row["ovr"])}</div>'
        f'<div class="fc-card__pos">{escape(str(row["position_main"]))}</div>'
        f'<div class="fc-card__sep"></div>'
        f'<div class="fc-card__chip">{int(row["age"])} TH</div>'
        f'<div class="fc-card__chip">{escape(str(row["league_short"]))}</div>'
        f'<div class="fc-card__chip">{escape(TIER_LABEL[tier])}</div>'
        f'</div><div class="fc-card__portrait">'
        f'<span>{escape(initials(row["short_name"]))}</span></div></div>'
        f'<div class="fc-card__name">{escape(str(row["short_name"]))}</div>'
        f'<div class="fc-card__club">{escape(str(row["club_name"]))}</div>'
        f'<div class="fc-card__stats">{stats}</div>'
        f'<div class="fc-card__foot">{escape(str(row.get("league_name", "")))}</div>'
        "</div></article>"
    )


def verdict_block(row: pd.Series) -> str:
    """Verdict dipecah tiga baris: label, penjelasan, lalu metadata angka.
    Sebelumnya semuanya digabung satu baris panjang dengan pemisah '. —'."""
    label, note = VERDICT_TEXT[row["verdict"]]
    key = str(row["verdict"]).lower()
    meta = (f'selisih {pct(row["gap_pct"])} vs harga pasar &nbsp;·&nbsp; '
            f'z-residual {num(row["residual_z"], 2)} &nbsp;·&nbsp; '
            f'grup {GROUP_LABEL[row["position_group"]]}')
    return (
        f'<div class="fc-verdict fc-verdict--{key}"><span class="fc-verdict__dot" '
        f'aria-hidden="true"></span>'
        f'<div><div class="fc-verdict__label">{escape(label)}</div>'
        f'<div class="fc-verdict__note">{escape(note)}</div>'
        f'<div class="fc-verdict__meta">{meta}</div></div></div>'
    )


def tiles(items: list[tuple]) -> str:
    """Baris tile KPI. Item boleh 3 elemen (label, nilai, sub) atau 4 elemen
    dengan elemen keempat `True` untuk menandai tile redup."""
    parts = []
    for item in items:
        key, value, sub = item[0], item[1], item[2]
        muted = " fc-tile--muted" if len(item) > 3 and item[3] else ""
        parts.append(
            f'<div class="fc-tile{muted}"><div class="fc-tile__k">{escape(key)}</div>'
            f'<div class="fc-tile__v">{escape(value)}</div>'
            f'<div class="fc-tile__s">{escape(sub)}</div></div>')
    return f'<div class="fc-tiles">{"".join(parts)}</div>'


def percentile_bars(row: pd.Series, cols: list[str] | None = None) -> str:
    cols = cols or CARD_STATS
    rows = []
    for col in cols:
        raw = int(row[col])
        p = float(row.get(f"pct_{col}", 0))
        cls = "hot" if p >= 80 else ("cold" if p < 40 else "")
        rows.append(
            f'<div class="fc-bar"><span class="fc-bar__k">{CARD_STAT_LABEL.get(col, col)}</span>'
            f'<span class="fc-bar__track" role="img" aria-label="Persentil {p:.0f} dari 100">'
            f'<span class="fc-bar__fill {cls}" '
            f'style="width:{p:.0f}%"></span></span>'
            f'<span class="fc-bar__v">{raw}<small> / {p:.0f}p</small></span></div>'
        )
    return f'<div class="fc-bars">{"".join(rows)}</div>'


GAP_BAR_SCALE = 60.0   # persen selisih yang mengisi setengah lebar bar
GAP_BAR_MIN = 3.5      # lebar minimum agar selisih kecil tetap terlihat sebagai tick


def _gap_bar(gap_pct: float) -> str:
    """Bar selisih dengan nol di tengah, agar besarannya bisa dibandingkan
    antar baris tanpa harus membaca angkanya satu per satu."""
    if gap_pct is None or pd.isna(gap_pct):
        return '<span class="fc-mini__bar"></span>'
    span = min(abs(float(gap_pct)) / GAP_BAR_SCALE, 1.0) * 50
    span = max(span, GAP_BAR_MIN)
    side = "left:50%" if gap_pct >= 0 else "right:50%"
    cls = "" if gap_pct >= 0 else ' class="neg"'
    return f'<span class="fc-mini__bar"><i{cls} style="{side};width:{span:.1f}%"></i></span>'


def mini_card(row: pd.Series, note: str | None = None) -> str:
    """Kartu ringkas satu baris. `note` (mis. skor kemiripan) diletakkan di
    dalam kartu, bukan sebagai caption terpisah di bawahnya, supaya tinggi
    tiap kartu dalam satu baris grid tetap sama."""
    cls = gap_class(row["gap_pct"])
    gap = money(row["gap_eur"])
    if row["gap_eur"] >= 0:
        gap = "+" + gap
    note_html = f'<div class="fc-mini__note">{escape(note)}</div>' if note else ""
    return (
        f'<article class="fc-mini" aria-label="{escape(str(row["short_name"]))}, '
        f'OVR {int(row["ovr"])}, harga {escape(money(row["value_eur"]))}">'
        f'<div class="fc-mini__ovr t-{row["tier"]}">{int(row["ovr"])}'
        f'<small>{escape(str(row["position_main"]))}</small></div>'
        f'<div class="fc-mini__main"><div class="fc-mini__name">{escape(str(row["short_name"]))}</div>'
        f'<div class="fc-mini__meta">{escape(str(row["club_name"]))} · {int(row["age"])} th · '
        f'{escape(str(row["league_short"]))}</div>{note_html}</div>'
        f'{_gap_bar(row["gap_pct"])}'
        f'<div class="fc-mini__val"><div class="now">{escape(money(row["value_eur"]))}</div>'
        f'<div class="gap {cls}">{escape(gap)}</div>'
        f'<div class="gap {cls}">{escape(pct(row["gap_pct"], 0))}</div></div></article>'
    )


# ---------------------------------------------------------------- chart
def _dark(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    """Tema chart: font body untuk semua teks kecil, ruang atas cukup untuk legenda."""
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=54 if legend else 26, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=INK, size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    bgcolor="rgba(0,0,0,0)", itemsizing="constant",
                    font=dict(family=BODY_FONT, size=11.5, color=DIM)),
        hoverlabel=dict(bgcolor="#0a0f1e", bordercolor="rgba(255,255,255,.2)",
                        font=dict(family=BODY_FONT, size=12, color=INK)),
    )
    axis = dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)",
                linecolor="rgba(255,255,255,.16)",
                tickfont=dict(family=BODY_FONT, size=11.5, color=DIM),
                title_font=dict(family=BODY_FONT, size=11.5, color=DIM_2),
                title_standoff=12)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def radar(row: pd.Series, benchmark: pd.Series | None = None,
          benchmark_name: str = "Median posisi") -> go.Figure:
    labels = [CARD_STAT_LABEL[s] for s in CARD_STATS]
    values = [float(row[s]) for s in CARD_STATS]
    fig = go.Figure()
    if benchmark is not None:
        bvals = [float(benchmark[s]) for s in CARD_STATS]
        fig.add_trace(go.Scatterpolar(
            r=bvals + bvals[:1], theta=labels + labels[:1], name=benchmark_name,
            line=dict(color="rgba(200,214,240,.55)", width=1.6, dash="dot"),
            fill="toself", fillcolor="rgba(200,214,240,.07)",
        ))
    fig.add_trace(go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1], name=str(row["short_name"]),
        line=dict(color=GREEN, width=2.6), fill="toself", fillcolor="rgba(0,245,160,.22)",
    ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(255,255,255,.02)",
        radialaxis=dict(range=[0, 100], showline=False, gridcolor="rgba(255,255,255,.09)",
                        showticklabels=False, ticks=""),
        angularaxis=dict(gridcolor="rgba(255,255,255,.09)",
                         tickfont=dict(family=DISPLAY_FONT, color=INK, size=15)),
    ))
    return _dark(fig, 360, legend=True)


def timeline(tl: pd.DataFrame, name: str) -> go.Figure:
    """Tanpa judul di dalam chart — judul seksi di atasnya sudah menjelaskan.

    Sebelumnya judul chart dan legenda menempati koordinat yang sama sehingga
    teksnya saling menumpuk.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tl["age"], y=tl["pred_m"], name="Estimasi model", mode="lines+markers",
        line=dict(color=CYAN, width=2, dash="dot"), marker=dict(size=7),
        hovertemplate="Umur %{x}<br>Estimasi €%{y:.1f} jt<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=tl["age"], y=tl["value_m"], name="Harga pasar", mode="lines+markers",
        line=dict(color=GREEN, width=3), marker=dict(size=9, line=dict(color="#04060d", width=1.5)),
        customdata=tl[["club_name", "ovr"]],
        hovertemplate="Umur %{x} · %{customdata[0]}<br>Pasar €%{y:.1f} jt · OVR %{customdata[1]}<extra></extra>",
    ))
    fig.update_xaxes(title="Umur", dtick=1)
    fig.update_yaxes(title="Nilai pasar (juta €)")
    # pemain dengan sedikit snapshot tidak butuh kanvas tinggi
    return _dark(fig, 260 if len(tl) < 4 else 340, legend=True)


def age_curve_chart(curve: pd.DataFrame, peak: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["age"], y=curve["p90_value_m"], name="Persentil 90",
        line=dict(color="rgba(25,211,250,.45)", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(25,211,250,.06)",
    ))
    fig.add_trace(go.Scatter(
        x=curve["age"], y=curve["median_value_m"], name="Median",
        line=dict(color=GREEN, width=3), mode="lines+markers", marker=dict(size=7),
    ))
    if peak:
        fig.add_vline(x=peak, line=dict(color=LIME, width=1.5, dash="dash"),
                      annotation_text=f"puncak {peak} th", annotation_position="top right",
                      annotation_font=dict(family=BODY_FONT, color=LIME, size=11))
    fig.update_xaxes(title="Umur", dtick=2)
    fig.update_yaxes(title="Nilai pasar (juta €)")
    return _dark(fig, 340, legend=True)


def market_scatter(df: pd.DataFrame, highlight: pd.Series | None = None) -> go.Figure:
    colours = {"BARGAIN": GREEN, "FAIR": "rgba(139,155,189,.45)", "OVERPRICED": RED}
    fig = go.Figure()
    for verdict_name, colour in colours.items():
        part = df.loc[df["verdict"] == verdict_name]
        fig.add_trace(go.Scattergl(
            x=part["value_m"], y=part["pred_m"], mode="markers", name=verdict_name,
            marker=dict(size=5, color=colour, opacity=.7),
            text=part["short_name"] + " · " + part["club_name"],
            hovertemplate="%{text}<br>Pasar €%{x:.1f} jt · AI €%{y:.1f} jt<extra></extra>",
        ))
    lim = [df["value_m"].min() * .8, df["value_m"].max() * 1.2]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Harga = estimasi",
                             line=dict(color="rgba(255,255,255,.35)", width=1.4, dash="dash")))
    if highlight is not None:
        fig.add_trace(go.Scatter(
            x=[highlight["value_m"]], y=[highlight["pred_m"]], mode="markers+text",
            name=str(highlight["short_name"]), text=[str(highlight["short_name"])],
            textposition="top center", textfont=dict(family=BODY_FONT, color=LIME, size=12),
            marker=dict(size=15, color=LIME, line=dict(color="#04060d", width=2), symbol="diamond"),
        ))
    fig.update_xaxes(title="Harga pasar (juta €)", type="log")
    fig.update_yaxes(title="Estimasi model (juta €)", type="log")
    return _dark(fig, 430, legend=True)


def importance_chart(imp: pd.DataFrame, labels: dict[str, str], top: int = 12) -> go.Figure:
    """Label nilai ditaruh di ujung bar supaya fitur berkontribusi kecil
    (1-5%) tetap terbaca angkanya, bukan cuma jadi garis tipis."""
    head = imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=head["share"], y=[labels.get(f, f) for f in head["feature"]], orientation="h",
        marker=dict(color=head["share"], colorscale=[[0, "#1d5f7a"], [.5, CYAN], [1, GREEN]],
                    line=dict(width=0)),
        text=[f"{v:.1f}%" for v in head["share"]], textposition="outside",
        textfont=dict(family=BODY_FONT, size=11, color=DIM),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title="Kontribusi terhadap prediksi (%)",
                     range=[0, head["share"].max() * 1.12])
    return _dark(fig, 330)


def compare_bars(a: pd.Series, b: pd.Series) -> go.Figure:
    labels = [CARD_STAT_LABEL[s] for s in CARD_STATS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[a[s] for s in CARD_STATS], name=str(a["short_name"]),
                         marker_color=GREEN, opacity=.9))
    fig.add_trace(go.Bar(x=labels, y=[b[s] for s in CARD_STATS], name=str(b["short_name"]),
                         marker_color=CYAN, opacity=.9))
    fig.update_layout(barmode="group", bargap=.28)
    fig.update_yaxes(title="Nilai atribut", range=[0, 100])
    return _dark(fig, 330, legend=True)


def value_hist(dist: pd.DataFrame, marker: float | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=dist["value_eur"], y=dist["count"],
                           marker=dict(color="rgba(0,245,160,.55)"),
                           hovertemplate="±%{x:,.0f} €<br>%{y} pemain<extra></extra>"))
    if marker:
        fig.add_vline(x=marker, line=dict(color=LIME, width=2))
    fig.update_xaxes(title="Harga pasar (€, skala log)", type="log")
    fig.update_yaxes(title="Jumlah pemain")
    return _dark(fig, 300)
