import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Football Player Valuation Tool",
    page_icon="⚽",
    layout="wide"
)

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('data_pemain_siap_pakai.csv')
    
    # --- PERBAIKAN ---
    # Karena kolom 'overall' tidak ada, kita urutkan berdasarkan 'value_eur' (Harga)
    # Pemain versi utama biasanya harganya paling mahal/valid.
    df = df.sort_values(by=['value_eur'], ascending=False)
    
    # Hapus duplikat berdasarkan nama. 
    # keep='first' artinya kita simpan yang paling atas (yang harganya paling mahal)
    df = df.drop_duplicates(subset=['short_name'], keep='first')
    
    return df

@st.cache_resource
def load_model():
    model = joblib.load('model_rf_market_value.pkl')
    return model

try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("File tidak ditemukan! Pastikan csv dan pkl ada di folder yang sama.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTER PINTAR)
# ==========================================
st.sidebar.header("🔍 Cari Pemain")

# Filter 1: Liga
list_liga = sorted(df['league_name'].unique())
liga_pilihan = st.sidebar.selectbox("Pilih Liga:", list_liga)
df_liga = df[df['league_name'] == liga_pilihan]

# Filter 2: Klub (Fitur Baru)
list_klub = sorted(df_liga['club_name'].unique())
klub_pilihan = st.sidebar.selectbox("Pilih Klub:", ["Semua Klub"] + list_klub)

if klub_pilihan != "Semua Klub":
    df_filtered = df_liga[df_liga['club_name'] == klub_pilihan]
else:
    df_filtered = df_liga

# Filter 3: Nama Pemain
list_pemain = sorted(df_filtered['short_name'].unique())
nama_pemain = st.sidebar.selectbox("Pilih Nama Pemain:", list_pemain)

# Ambil data pemain terpilih
player_data = df[df['short_name'] == nama_pemain].iloc[0]

# ==========================================
# 4. PROSES PREDIKSI & LOGIKA STATUS
# ==========================================
# Kolom fitur HARUS SAMA PERSIS urutannya dengan saat training di Colab
feature_cols = [
    'age', 'pace', 'movement_acceleration', 'movement_sprint_speed',
    'movement_agility', 'movement_reactions', 'movement_balance',
    'physic', 'power_stamina', 'power_strength', 'mentality_composure',
    'shooting', 'attacking_finishing', 'power_shot_power', 'power_long_shots',
    'passing', 'attacking_short_passing', 'mentality_vision', 'attacking_crossing',
    'dribbling', 'skill_dribbling', 'skill_ball_control',
    'defending', 'defending_standing_tackle', 'mentality_interceptions'
]

# Siapkan data input
input_data = pd.DataFrame([player_data[feature_cols]])

# Prediksi
predicted_value = model.predict(input_data)[0]
actual_value = player_data['value_eur']

# Hitung Selisih
diff = predicted_value - actual_value
persentase_diff = (diff / actual_value) * 100 if actual_value > 0 else 0

# Logika Status (Threshold 10%)
threshold = 10.0

if persentase_diff > threshold:
    status = "Undervalued (Layak Beli)"
    status_box = "success" # Warna Hijau
    pesan = "Statistik pemain ini jauh di atas harganya. Potensi keuntungan tinggi!"
elif persentase_diff < -threshold:
    status = "Overvalued (Terlalu Mahal)"
    status_box = "error" # Warna Merah
    pesan = "Performa statistik tidak sebanding dengan harganya yang tinggi."
else:
    status = "Fair Value (Harga Wajar)"
    status_box = "info" # Warna Biru
    pesan = "Harga pasar saat ini sudah sesuai dengan kualitas statistik pemain."

# ==========================================
# 5. TAMPILAN UTAMA (DASHBOARD)
# ==========================================
st.title(f"⚽ Analisis Valuasi: {player_data['short_name']}")
st.markdown(f"**Klub:** {player_data['club_name']} | **Posisi:** {player_data['player_positions']} | **Umur:** {player_data['age']} Tahun")

st.divider()

# TAB LAYOUT (Agar rapi)
tab1, tab2, tab3 = st.tabs(["📊 Analisis Harga", "🕸️ Radar Skill", "📝 Data Mentah"])

# --- TAB 1: ANALISIS HARGA ---
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Harga Pasar (Market)", f"€ {actual_value:,.0f}")
    
    with col2:
        st.metric("Harga Wajar (AI)", f"€ {predicted_value:,.0f}", 
                  delta=f"{diff:,.0f} ({persentase_diff:.1f}%)")
    
    with col3:
        st.markdown("### Status:")
        # Tampilkan kotak status berwarna
        if status_box == "success":
            st.success(f"**{status}**")
        elif status_box == "error":
            st.error(f"**{status}**")
        else:
            st.info(f"**{status}**")

    st.caption(f"💡 *Catatan AI: {pesan}*")
    
    st.subheader("Atribut Penentu Harga")
    # Tampilkan grafik batang atribut tertinggi pemain ini
    stats_to_show = input_data.T.reset_index()
    stats_to_show.columns = ['Atribut', 'Nilai']
    stats_to_show = stats_to_show.sort_values(by='Nilai', ascending=False).head(10)
    st.bar_chart(stats_to_show.set_index('Atribut'))

# --- TAB 2: RADAR CHART (VISUALISASI KEREN) ---
with tab2:
    st.subheader(f"Profil Skill: {player_data['short_name']}")
    
    # Pilih atribut utama untuk Radar Chart
    # Pastikan nama kolom ini ada di CSV Anda
    categories = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    
    # Ambil nilainya
    values = [player_data[cat] for cat in categories]
    
    # Buat DataFrame khusus grafik
    df_radar = pd.DataFrame(dict(
        r=values,
        theta=categories
    ))
    
    # Buat Grafik Radar dengan Plotly
    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True,
                        range_r=[0,100], 
                        title='Atribut Utama (Skala 0-100)')
    fig.update_traces(fill='toself', line_color='blue')
    
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: DATA MENTAH ---
with tab3:
    st.write("Data lengkap dari dataset:")
    st.dataframe(player_data)