"""Football Valuation Centre — scouting & market value intelligence.

Tampilan bergaya EA FC Ultimate Team di atas dataset 26.397 snapshot pemain
lima liga (plus liga lain yang labelnya diperbaiki) dan model Random Forest.

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import math
from html import escape

import pandas as pd
import streamlit as st

from fcv import data as fcv_data
from fcv import insights, ratings, ui
from fcv import model as fcv_model
from fcv.config import (ATTR_LABEL, CARD_STATS, GROUP_LABEL, STAT_DETAIL,
                        TIER_LABEL)

st.set_page_config(page_title="Football Valuation Centre", page_icon="⚽",
                   layout="wide")


# ----------------------------------------------------------------- loader
@st.cache_resource(show_spinner="Menyiapkan estimasi pemain...")
def get_model():
    return fcv_model.load_model()


@st.cache_data(show_spinner="Membersihkan data & menghitung rating...")
def get_dataset():
    df, dq = fcv_data.build_dataset()
    scored = fcv_model.attach_predictions(df, get_model())
    return scored, dq


@st.cache_data(show_spinner=False)
def get_importance():
    return fcv_model.importance_table(get_model())


try:
    DF, DQ = get_dataset()
except FileNotFoundError as exc:
    st.error(f"Berkas data/model tidak ditemukan: {exc}")
    st.stop()

METRICS = fcv_model.load_metrics()
LOW, HIGH = ratings.verdict_thresholds(DF["log_ratio"])
LATEST = DF.loc[DF["is_latest"]]
OPTIONS = fcv_data.player_options(DF)

LEAGUE_CHOICES = sorted(LATEST["league_name"].unique())
DEFAULT_LEAGUES = [
    league for league in LEAGUE_CHOICES
    if league in {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
]
DEFAULT_GROUPS = ["DEF", "MID", "ATT"]
DEFAULT_AGE_RANGE = (DQ.age_min, DQ.age_max)
DEFAULT_VALUE_RANGE = (0.0, 200.0)
DEFAULT_MIN_OVR = 60


def reset_market_filters() -> None:
    """Kembalikan filter pasar tanpa mengubah pemain yang sedang dilihat."""
    st.session_state.market_leagues = DEFAULT_LEAGUES
    st.session_state.market_groups = DEFAULT_GROUPS
    st.session_state.market_age = DEFAULT_AGE_RANGE
    st.session_state.market_value = DEFAULT_VALUE_RANGE
    st.session_state.market_ovr = DEFAULT_MIN_OVR

ui.inject_css()
model_note = (f"{METRICS['holdout_fresh_model']['median_ape_pct']:.0f}%"
              if METRICS else "Belum tersedia")
ui.app_header(DQ, model_note)


# ---------------------------------------------------------- scoped controls
def render_player_picker():
    with st.container(border=True, key="player_picker"):
        c1, c2 = st.columns([1.5, 1], gap="medium")
        with c1:
            label = st.selectbox("Cari pemain", OPTIONS.index, key="hub_player",
                                 help="Ketik nama atau klub. Urutan awal berdasarkan OVR estimasi.")
        key = OPTIONS[label]
        rows = DF.loc[DF["player_key"] == key].sort_values("snapshot_index")
        with c2:
            snapshot = st.selectbox(
                "Snapshot pemain", rows.index, index=len(rows) - 1,
                key=f"snapshot_{key}",
                format_func=lambda i: f"{int(rows.loc[i, 'age'])} tahun · "
                f"{rows.loc[i, 'club_name']} · #{int(rows.loc[i, 'snapshot_index'])}",
                help="Default: umur tertinggi. Nomor snapshot membedakan baris berumur sama; tahun edisi tidak tersedia.",
            )
        st.caption("Pilih pemain dan snapshot untuk melihat profil. OVR adalah estimasi; data bukan harga transfer terkini.")
    return label, key, rows.loc[snapshot]


def render_market_filters() -> pd.DataFrame:
    with st.expander("Filter pasar", expanded=True):
        st.caption("Hanya berlaku di Market Scanner. Pilih minimal satu liga dan satu lini.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            leagues = st.multiselect("Liga", LEAGUE_CHOICES, default=DEFAULT_LEAGUES,
                                     key="market_leagues", placeholder="Pilih liga")
        with c2:
            groups = st.multiselect("Lini", DEFAULT_GROUPS, default=DEFAULT_GROUPS,
                                    format_func=lambda g: GROUP_LABEL[g], key="market_groups",
                                    placeholder="Pilih lini")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            age_range = st.slider("Rentang umur", DQ.age_min, DQ.age_max,
                                  DEFAULT_AGE_RANGE, key="market_age")
        with c2:
            value_range = st.slider("Harga pasar (juta €)", 0.0,
                                    float(LATEST["value_m"].max().round()), DEFAULT_VALUE_RANGE,
                                    step=0.5, key="market_value")
        with c3:
            ovr = st.slider("OVR minimum", 40, 99, DEFAULT_MIN_OVR, key="market_ovr")
        st.button("Reset filter pasar", on_click=reset_market_filters, key="reset_market")
    return insights.market_pool(DF, leagues=leagues, groups=groups,
                                age_range=age_range, value_range=value_range, min_ovr=ovr)


# ----------------------------------------------------------------- tab 1
def render_player_hub() -> None:
    ui.tab_intro(f"Kartu <b>{escape(selected['short_name'])}</b> pada snapshot umur "
                 f"{int(selected['age'])}, beserta posisi harganya terhadap estimasi model.")
    left, right = st.columns([0.86, 1.5], gap="large")

    with left:
        st.markdown(ui.player_card(selected), unsafe_allow_html=True)
        st.markdown(
            f'<div class="fc-note" style="text-align:center;margin-top:14px">'
            f'{selected["position_all"]} · {selected["league_name"]} '
            f'({selected["country"]})<br>Snapshot {int(selected["snapshot_index"])} '
            f'dari {int(selected["snapshot_total"])} · tier {TIER_LABEL[selected["tier"]]}'
            f'</div>', unsafe_allow_html=True)

    with right:
        st.markdown(ui.verdict_block(selected), unsafe_allow_html=True)
        st.markdown(ui.tiles([
            ("Harga pasar", ui.money(selected["value_eur"]),
             f"umur {int(selected['age'])} · persentil {selected['pct_value_eur']:.0f} di lini"),
            ("Estimasi model", ui.money(selected["pred_eur"]),
             f"selisih {ui.money(selected['gap_eur'])}"),
            ("Gap", ui.pct(selected["gap_pct"]), "estimasi vs harga pasar"),
            ("OVR est.", str(int(selected["ovr"])),
             f"persentil {selected['pct_ovr']:.0f} di antara "
             f"{ui.count((DF['position_group'] == selected['position_group']).sum())} snapshot selini"),
        ]), unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            benchmark = group_peers[CARD_STATS].median()
            radar_fig = ui.radar(selected, benchmark,
                                 f"Median {GROUP_LABEL[selected['position_group']]}")
            st.plotly_chart(radar_fig, width="stretch", key="radar_hub",
                            config=ui.CHART_CONFIG)
        with c2:
            st.markdown('<div class="fc-note" style="margin:10px 0 14px">'
                        'Nilai atribut · P = persentil di lini yang sama. '
                        'P90 berarti melampaui sekitar 90% snapshot pembanding.</div>',
                        unsafe_allow_html=True)
            st.markdown(ui.percentile_bars(selected), unsafe_allow_html=True)

    timeline_df = insights.career_timeline(DF, player_key)
    if len(timeline_df) > 1:
        ui.section("Lini waktu karier", f"{len(timeline_df)} snapshot · diurutkan berdasarkan umur, bukan tahun edisi")
        st.plotly_chart(ui.timeline(timeline_df, selected["short_name"]),
                        width="stretch", key="timeline_hub", config=ui.CHART_CONFIG)

    ui.section("Rincian atribut", "kelompok detail di balik enam stat kartu")
    detail_cards = []
    for stat, children in STAT_DETAIL.items():
        rows = "".join(
            f'<div class="fc-bar fc-bar--compact"><span class="fc-bar__k">{ATTR_LABEL[c]}</span>'
            f'<span class="fc-bar__track"><span class="fc-bar__fill" '
            f'style="width:{float(selected[c]):.0f}%"></span></span>'
            f'<span class="fc-bar__v">{int(selected[c])}</span></div>'
            for c in children
        )
        detail_cards.append(
            f'<div class="fc-tile">'
            f'<div class="fc-tile__k">{ATTR_LABEL[stat]} · {int(selected[stat])}</div>'
            f'<div class="fc-bars" style="margin-top:10px">{rows}</div></div>')
    st.markdown(ui.grid(detail_cards, 3), unsafe_allow_html=True)

    ui.section("Profil serupa", "enam profil terdekat berdasarkan atribut dan lini bermain")
    similar = insights.similar_players(DF, selected, k=6)
    if similar.empty:
        st.info("Tidak ada pembanding di lini ini.")
    else:
        cards = [ui.mini_card(row, note=f"kemiripan {row['similarity']:.0f}% · "
                                        f"{row['position_all']}")
                 for _, row in similar.iterrows()]
        st.markdown(ui.grid(cards, 2), unsafe_allow_html=True)


# ----------------------------------------------------------------- tab 2
def render_scanner() -> None:
    ui.tab_intro("Temukan kandidat dari snapshot umur tertinggi setiap pemain. "
                 "Atur filter, lalu bandingkan harga pasar dengan estimasi model.")
    pool = render_market_filters()
    st.markdown(ui.filter_result(len(pool), len(LATEST)), unsafe_allow_html=True)
    if pool.empty:
        st.info("Belum ada pemain yang cocok. Pilih minimal satu liga dan lini, "
                "turunkan OVR minimum, atau lebarkan rentang umur dan harga.")
        st.button("Kembalikan filter awal", on_click=reset_market_filters, key="empty_reset",
                  type="primary")
        return

    sort_mode = st.radio("Urutkan", ["abs", "pct"], horizontal=True, key="scan_sort",
                         format_func=lambda m: "Selisih €" if m == "abs" else "Selisih %",
                         help="Euro mengutamakan selisih nominal. Persen mengutamakan selisih relatif dan sering menyorot pemain murah.")

    bargains = insights.market_scan(pool, side="BARGAIN", sort_by=sort_mode, limit=12)
    overpriced = insights.market_scan(pool, side="OVERPRICED", sort_by=sort_mode, limit=12)
    st.markdown(ui.tiles([
        ("Pemain terfilter", ui.count(len(pool)), "snapshot terbaru"),
        ("Bargain", ui.count((pool["verdict"] == "BARGAIN").sum()),
         f"{(pool['verdict'] == 'BARGAIN').mean() * 100:.0f}% dari pool"),
        ("Overpriced", ui.count((pool["verdict"] == "OVERPRICED").sum()),
         f"{(pool['verdict'] == 'OVERPRICED').mean() * 100:.0f}% dari pool"),
        ("Fair value", ui.count((pool["verdict"] == "FAIR").sum()), "di antara dua ambang verdict"),
        ("Nilai pasar pool", ui.money(pool["value_eur"].sum()), "total"),
    ]), unsafe_allow_html=True)
    st.caption("Kartu menampilkan harga pasar, selisih € dan %. Verdict adalah sinyal relatif model; lihat batasannya di Insights.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        ui.section("Bargain", f"{len(bargains)} kandidat · gap minimal {ui.pct(math.expm1(HIGH) * 100, 0)}")
        if bargains.empty:
            st.info("Tidak ada kandidat bargain pada filter ini. Pemain fair value tetap ada di peta pasar.")
        else:
            st.markdown(ui.stack([ui.mini_card(row) for _, row in bargains.iterrows()]),
                        unsafe_allow_html=True)
    with col_b:
        ui.section("Overpriced", f"{len(overpriced)} kandidat · gap maksimal {ui.pct(math.expm1(LOW) * 100, 0)}")
        if overpriced.empty:
            st.info("Tidak ada kandidat overpriced pada filter ini. Pemain fair value tetap ada di peta pasar.")
        else:
            st.markdown(ui.stack([ui.mini_card(row) for _, row in overpriced.iterrows()]),
                        unsafe_allow_html=True)

    ui.section("Peta pasar", "garis putus = harga sama dengan estimasi model")
    st.plotly_chart(ui.market_scatter(pool), width="stretch", key="scatter_scan",
                    config=ui.CHART_CONFIG)

    ui.section("Papan liga", "mengikuti filter pasar yang sama")
    table = insights.league_table(pool)
    st.dataframe(
        table.rename(columns={
            "league_name": "Liga", "country": "Negara", "pemain": "Pemain", "klub": "Klub",
            "ovr_median": "OVR median", "umur_median": "Umur median",
            "nilai_median_m": "Nilai median (jt €)", "nilai_total_m": "Total (jt €)",
            "bargain": "% bargain"}),
        width="stretch", hide_index=True,
        column_config={
            "Nilai median (jt €)": st.column_config.NumberColumn(format="%.1f"),
            "Total (jt €)": st.column_config.NumberColumn(format="%.0f"),
            "% bargain": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                         max_value=100),
        })


# ----------------------------------------------------------------- tab 3
def render_compare() -> None:
    ui.tab_intro("Bandingkan dua entri pemain pada snapshot umur tertinggi masing-masing. "
                 "Pilihan di sini terpisah dari filter pasar.")
    labels = list(OPTIONS.index)
    c1, c2 = st.columns(2)
    with c1:
        label_a = st.selectbox("Pemain A", labels, index=0,
                               key="cmp_a")
    with c2:
        label_b = st.selectbox("Pemain B", labels, index=1,
                               key="cmp_b")

    row_a = LATEST.loc[LATEST["player_key"] == OPTIONS[label_a]].iloc[0]
    row_b = LATEST.loc[LATEST["player_key"] == OPTIONS[label_b]].iloc[0]

    if label_a == label_b:
        st.warning("Pilih dua pemain berbeda agar perbandingan atribut dan valuasi bermakna.")
        return

    card_a, card_b = st.columns(2, gap="large")
    with card_a:
        ui.section("Pemain A", f"{row_a['short_name']} · {int(row_a['age'])} tahun")
        st.markdown(ui.player_card(row_a), unsafe_allow_html=True)
    with card_b:
        ui.section("Pemain B", f"{row_b['short_name']} · {int(row_b['age'])} tahun")
        st.markdown(ui.player_card(row_b), unsafe_allow_html=True)
    st.markdown(ui.tiles([
            ("Harga A", ui.money(row_a["value_eur"]), str(row_a["short_name"])),
            ("Harga B", ui.money(row_b["value_eur"]), str(row_b["short_name"])),
            ("Selisih harga", ui.money(row_a["value_eur"] - row_b["value_eur"]), "A − B"),
            ("Beda OVR", f"{int(row_a['ovr']) - int(row_b['ovr']):+d}", "A − B"),
            ("Gap A vs estimasi", ui.pct(row_a["gap_pct"]), row_a["verdict"]),
            ("Gap B vs estimasi", ui.pct(row_b["gap_pct"]), row_b["verdict"]),
        ]), unsafe_allow_html=True)

    ui.section("Perbandingan atribut", "enam stat kartu, angka pastinya ada di tabel")
    st.plotly_chart(ui.compare_bars(row_a, row_b), width="stretch", key="cmp_bars",
                    config=ui.CHART_CONFIG)

    diff = insights.comparison_table(row_a, row_b)
    st.dataframe(diff, width="stretch", hide_index=True)


# ----------------------------------------------------------------- tab 4
def render_insights() -> None:
    ui.tab_intro("Pola agregat dari seluruh dataset, plus laporan seberapa akurat "
                 "model ini sebenarnya.")
    insight_groups = st.multiselect("Lini untuk analisis", DEFAULT_GROUPS,
                                    default=DEFAULT_GROUPS, key="insight_groups",
                                    format_func=lambda g: GROUP_LABEL[g],
                                    help="Hanya memengaruhi kurva umur dan papan atas atribut.")
    ui.section("Kurva umur", "median harga per umur — memakai seluruh 26 ribu snapshot")
    curve = insights.age_curve(DF, insight_groups)
    peak = insights.peak_age(curve)
    if curve.empty:
        st.info("Pilih minimal satu lini untuk menampilkan kurva umur.")
    else:
        st.plotly_chart(ui.age_curve_chart(curve, peak), width="stretch", key="age_curve",
                        config=ui.CHART_CONFIG)

    col1, col2 = st.columns([1.1, 1], gap="large")
    with col1:
        ui.section("Pendorong harga", "kontribusi fitur di model Random Forest")
        st.plotly_chart(ui.importance_chart(get_importance(), ATTR_LABEL),
                        width="stretch", key="imp_chart", config=ui.CHART_CONFIG)
        top = get_importance().iloc[0]
        st.markdown(
            f'<div class="fc-note">{ATTR_LABEL.get(top["feature"], top["feature"])} '
            f'menyumbang {top["share"]:.0f}% keputusan model. Konsentrasi sebesar ini '
            f'berarti model praktis membaca satu atribut ringkasan; atribut teknis lain '
            f'hanya jadi penghalus.</div>', unsafe_allow_html=True)
    with col2:
        ui.section("Akurasi model", "diuji pada pemain yang belum pernah dilihat")
        if METRICS:
            honest = METRICS["holdout_fresh_model"]
            insample = METRICS["shipped_model_on_all_rows"]
            base = METRICS["baseline_median"]
            st.markdown(ui.tiles([
                ("R² holdout", f"{honest['r2']:.3f}", f"{ui.count(honest['n'])} baris uji"),
                ("Median error", f"{honest['median_ape_pct']:.1f}%", "out-of-sample"),
                ("Dalam ±20%", f"{honest['within_20pct']:.0f}%", "prediksi"),
                ("R² in-sample", f"{insample['r2']:.3f}", "angka yang menipu", True),
                ("Error in-sample", f"{insample['median_ape_pct']:.1f}%", "di data latih", True),
                ("Baseline", f"{base['median_ape_pct']:.0f}%", "median harga saja", True),
            ]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="fc-note" style="margin-top:10px">Split '
                f'<span class="fc-kbd">group by pemain</span> supaya snapshot umur '
                f'berbeda dari orang yang sama tidak bocor antara latih dan uji. '
                f'Model yang disertakan dilatih di seluruh baris. Median error '
                f'{insample["median_ape_pct"]:.0f}% pada data latih terlalu optimistis; '
                f'hasil holdout pembanding adalah {honest["median_ape_pct"]:.0f}%. '
                f'Ini median error, bukan jaminan rentang prediksi. Verdict BARGAIN/OVERPRICED '
                f'dibaca sebagai peringkat relatif (kuantil 15/85 residual), bukan '
                f'klaim absolut.</div>', unsafe_allow_html=True)
        else:
            st.info("Jalankan `python scripts/evaluate_model.py` untuk mengisi panel ini.")

    ui.section("Sebaran harga pasar", "skala log, snapshot terbaru")
    st.plotly_chart(ui.value_hist(insights.value_distribution(DF)),
                    width="stretch", key="value_hist", config=ui.CHART_CONFIG)

    ui.section("Papan atas atribut", "snapshot terbaru, mengikuti filter lini")
    stat_pick = st.selectbox("Atribut", CARD_STATS + ["ovr", "movement_reactions"],
                            format_func=lambda c: ATTR_LABEL.get(c, c), key="stat_pick")
    leaders = insights.stat_leaders(DF, stat_pick, 10, insight_groups)
    if leaders.empty:
        st.info("Pilih minimal satu lini untuk menampilkan papan atas atribut.")
        return
    st.dataframe(
        leaders.rename(columns={
            "short_name": "Pemain", "club_name": "Klub", "position_main": "Posisi",
            "age": "Umur", "ovr": "OVR", stat_pick: ATTR_LABEL.get(stat_pick, stat_pick),
            "value_m": "Nilai (jt €)"}),
        width="stretch", hide_index=True,
        column_config={"Nilai (jt €)": st.column_config.NumberColumn(format="%.1f")})


# ----------------------------------------------------------------- tab 5
def render_audit() -> None:
    ui.tab_intro("Laporan terbuka soal apa yang diperbaiki dari CSV mentah, dan apa "
                 "yang masih jadi keterbatasan.")
    ui.section("Ringkasan", "sebelum dan sesudah pembersihan")
    st.markdown(ui.tiles([
        ("Baris mentah", ui.count(DQ.rows_raw), "data_pemain_siap_pakai.csv"),
        ("Baris dipakai", ui.count(DQ.rows_used), f"{DQ.rows_used / DQ.rows_raw * 100:.1f}% dari mentah"),
        ("Pemain unik", ui.count(DQ.players), "nama + posisi utama"),
        ("Klub", ui.count(DQ.clubs), f"{DQ.leagues_after} kompetisi"),
        ("Klub dilabel ulang", f"{DQ.clubs_relabelled}", "liga salah di CSV"),
        ("Baris GK dibuang", f"{DQ.gk_rows_dropped}", "atribut kiper tidak ada"),
    ]), unsafe_allow_html=True)

    st.markdown(
        f'<div class="fc-note" style="margin-top:14px">'
        f'<b>1. Label liga rusak.</b> Nama liga di CSV sudah dipotong prefix negaranya, '
        f'sehingga <i>Brazilian Serie A</i> melebur ke <i>Serie A</i>, '
        f'<i>Russian Premier League</i> ke <i>Premier League</i>, dan '
        f'<i>Austrian Bundesliga</i> ke <i>Bundesliga</i>. Akibatnya Santos, Zenit, '
        f'dan Rapid Wien tampil sebagai klub liga top 5. {DQ.clubs_relabelled} klub '
        f'dipetakan ulang ke kompetisi dan negara aslinya: '
        + ", ".join(f"{v} klub → {k}" for k, v in sorted(DQ.relabel_detail.items())) +
        '.<br><br>'
        f'<b>2. Semua snapshot dipakai.</b> Setiap baris adalah satu edisi/musim pemain '
        f'(Mbappé muncul 8 kali, umur 17 di Monaco sampai 24 di PSG). Versi lama app '
        f'membuang duplikat nama dan hanya menyisakan 8.746 baris termahal — '
        f'{ui.count(DQ.rows_raw - 8746)} baris hilang beserta seluruh informasi perkembangan umur. '
        f'Sekarang {ui.count(DQ.rows_used)} baris terpakai untuk kurva umur, lini waktu, dan percentile.'
        f'<br><br>'
        f'<b>3. Identitas pemain.</b> {DQ.same_age_collisions} baris masih bertabrakan '
        f'(nama + posisi + umur sama) karena dataset tidak punya ID pemain; sisanya '
        f'terpisah rapi. Nama pendek saja tidak cukup — ada tiga "Danilo" berbeda.'
        f'<br><br>'
        f'<b>4. Rating kartu direkonstruksi.</b> CSV tidak punya kolom <i>overall</i>, '
        f'jadi OVR diperkirakan dari bobot posisi atas enam stat kartu digabung '
        f'<i>movement_reactions</i>. Korelasinya 0,905 terhadap log harga (reactions '
        f'sendiri 0,83). Angka ini estimasi, bukan OVR resmi EA.'
        f'</div>', unsafe_allow_html=True)

    with st.expander("Batasan yang perlu diingat"):
        st.markdown(
            '<div class="fc-note">'
            '• Dataset tidak menyimpan tahun edisi, jadi harga antar snapshot berasal dari '
            'ekonomi game yang berbeda dan tidak disesuaikan inflasi.<br>'
            '• Klub yang terdata hanya klub yang bertahan di daftar akhir tiap liga, '
            'sehingga pemain di klub terdegradasi bisa hilang dari sebagian musim.<br>'
            '• Tidak ada kolom kebangsaan, tinggi, kaki dominan, maupun foto — kartu '
            'memakai monogram inisial sebagai ganti portrait.<br>'
            '• Model hanya melihat 25 atribut. Faktor nyata seperti sisa kontrak, cedera, '
            'menit bermain, dan hype pasar tidak ada di data.'
            '</div>', unsafe_allow_html=True)

    with st.expander("Contoh baris hasil olahan"):
        cols = ["short_name", "club_name", "league_name", "country", "position_main",
                "position_group", "age", "ovr", "tier", "value_m", "pred_m", "gap_pct",
                "verdict", "snapshot_index", "snapshot_total"]
        st.dataframe(DF.loc[DF["player_key"] == player_key, cols], width="stretch",
                     hide_index=True)


# ----------------------------------------------------------------- render
TABS = st.tabs(["PLAYER HUB", "MARKET SCANNER", "HEAD TO HEAD", "INSIGHTS", "DATA AUDIT"])
with TABS[0]:
    picked_label, player_key, selected = render_player_picker()
    group_peers = LATEST.loc[LATEST["position_group"] == selected["position_group"]]
    render_player_hub()
with TABS[1]:
    render_scanner()
with TABS[2]:
    render_compare()
with TABS[3]:
    render_insights()
with TABS[4]:
    render_audit()

st.markdown(
    f'<div class="fc-foot">Football Valuation Centre · {ui.count(DQ.rows_used)} snapshot · '
    f'model Random Forest 100 pohon · verdict memakai kuantil residual '
    f'{ui.pct(math.expm1(LOW) * 100, 0)} / {ui.pct(math.expm1(HIGH) * 100, 0)}</div>',
    unsafe_allow_html=True)
