"""Konfigurasi terpusat: path, kamus perbaikan liga, dan konstanta rating.

Semua "angka ajaib" dikumpulkan di sini supaya modul lain bebas dari hardcode.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = ROOT / "data_pemain_siap_pakai.csv"
MODEL_PKL = ROOT / "model_rf_market_value.pkl"
METRICS_JSON = ROOT / "model_metrics.json"
CSS_FILE = ROOT / "assets" / "style.css"

# --------------------------------------------------------------------------
# PERBAIKAN DATA: kolom `league_name` di CSV sumber rusak.
# Nama liga di dataset asli dipotong prefix negaranya, sehingga:
#   "Italian Serie A"      + "Brazilian Serie A"       -> "Serie A"
#   "English Premier League" + "Russian Premier League" -> "Premier League"
#   "German Bundesliga"    + "Austrian Bundesliga"      -> "Bundesliga"
# Akibatnya 73 klub non-top-5 ikut terlabeli sebagai liga top 5.
# Kamus di bawah mengembalikan liga dan negara yang benar per klub.
# --------------------------------------------------------------------------
BRAZILIAN_CLUBS = [
    "América Mineiro", "Athletico Paranaense", "Atlético GO", "Atlético Mineiro",
    "Avaí FC", "Bahia", "Botafogo", "Bragantino", "Ceará",
    "Centro Sportivo Alagoano", "Chapecoense", "Corinthians", "Coritiba",
    "Cruzeiro", "Cuiabá", "Figueirense", "Flamengo", "Fluminense", "Fortaleza",
    "Goiás", "Grêmio", "Internacional", "Juventude", "Palmeiras", "Paraná",
    "Ponte Preta", "Santa Cruz", "Santos", "Sport Club do Recife", "São Paulo",
    "Vasco da Gama", "Vitória",
]
RUSSIAN_CLUBS = [
    "Amkar Perm", "Arsenal Tula", "CSKA Moskva", "Dinamo Moscow",
    "FC Anzhi Makhachkala", "FC Krasnodar", "FC Orenburg", "FC Rostov",
    "FC Tom Tomsk", "FC Tosno", "FC Ufa", "FC Ural Yekaterinburg",
    "Krylya Sovetov Samara", "Lokomotiv Moskva", "Rubin Kazan",
    "SKA Khabarovsk", "Spartak Moskva", "Terek Grozny", "Zenit St. Petersburg",
]
UKRAINIAN_CLUBS = ["Dynamo Kyiv", "Shakhtar Donetsk"]
SOUTH_AFRICAN_CLUBS = ["Kaizer Chiefs", "Mamelodi Sundowns", "Orlando Pirates"]
AUSTRIAN_CLUBS = [
    "Admira", "Austria Klagenfurt", "Austria Lustenau", "Austria Wien",
    "Blau-Weiß Linz", "FC Wacker Innsbruck", "Hartberg", "LASK Linz",
    "Rapid Wien", "Rheindorf Altach", "Ried", "SV Mattersburg", "Salzburg",
    "St. Pölten", "Sturm Graz", "WSG Tirol", "Wolfsberger AC",
]

CLUB_LEAGUE_FIX: dict[str, tuple[str, str]] = {}
for _club in BRAZILIAN_CLUBS:
    CLUB_LEAGUE_FIX[_club] = ("Brasileirão Série A", "Brasil")
for _club in RUSSIAN_CLUBS:
    CLUB_LEAGUE_FIX[_club] = ("Russian Premier League", "Rusia")
for _club in UKRAINIAN_CLUBS:
    CLUB_LEAGUE_FIX[_club] = ("Ukrainian Premier League", "Ukraina")
for _club in SOUTH_AFRICAN_CLUBS:
    CLUB_LEAGUE_FIX[_club] = ("South African PSL", "Afrika Selatan")
for _club in AUSTRIAN_CLUBS:
    CLUB_LEAGUE_FIX[_club] = ("Austrian Bundesliga", "Austria")

# Liga top 5 yang labelnya memang sudah benar -> hanya perlu negara.
TOP5_COUNTRY = {
    "Premier League": "Inggris",
    "La Liga": "Spanyol",
    "Serie A": "Italia",
    "Bundesliga": "Jerman",
    "Ligue 1": "Prancis",
}

LEAGUE_SHORT = {
    "Premier League": "EPL",
    "La Liga": "LAL",
    "Serie A": "SEA",
    "Bundesliga": "BUN",
    "Ligue 1": "LI1",
    "Brasileirão Série A": "BRA",
    "Russian Premier League": "RPL",
    "Ukrainian Premier League": "UPL",
    "South African PSL": "PSL",
    "Austrian Bundesliga": "AUT",
}

# --------------------------------------------------------------------------
# Posisi
# --------------------------------------------------------------------------
POSITION_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "ATT", "RW": "ATT", "ST": "ATT", "CF": "ATT",
}
GROUP_LABEL = {"GK": "Kiper", "DEF": "Belakang", "MID": "Tengah", "ATT": "Depan"}

# Enam stat kartu ala EA FC.
CARD_STATS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
CARD_STAT_LABEL = {
    "pace": "PAC", "shooting": "SHO", "passing": "PAS",
    "dribbling": "DRI", "defending": "DEF", "physic": "PHY",
}

# Fitur detail per stat utama, untuk panel "rincian atribut".
STAT_DETAIL = {
    "pace": ["movement_acceleration", "movement_sprint_speed"],
    "shooting": ["attacking_finishing", "power_shot_power", "power_long_shots"],
    "passing": ["attacking_short_passing", "mentality_vision", "attacking_crossing"],
    "dribbling": ["movement_agility", "movement_balance", "skill_dribbling", "skill_ball_control"],
    "defending": ["defending_standing_tackle", "mentality_interceptions"],
    "physic": ["power_stamina", "power_strength", "mentality_composure"],
}

ATTR_LABEL = {
    "movement_acceleration": "Acceleration", "movement_sprint_speed": "Sprint Speed",
    "movement_agility": "Agility", "movement_reactions": "Reactions",
    "movement_balance": "Balance", "power_stamina": "Stamina",
    "power_strength": "Strength", "mentality_composure": "Composure",
    "attacking_finishing": "Finishing", "power_shot_power": "Shot Power",
    "power_long_shots": "Long Shots", "attacking_short_passing": "Short Passing",
    "mentality_vision": "Vision", "attacking_crossing": "Crossing",
    "skill_dribbling": "Dribbling", "skill_ball_control": "Ball Control",
    "defending_standing_tackle": "Standing Tackle",
    "mentality_interceptions": "Interceptions",
    "age": "Umur", "pace": "Pace", "shooting": "Shooting", "passing": "Passing",
    "dribbling": "Dribbling", "defending": "Defending", "physic": "Physicality",
}

# Ambang tier kartu berdasarkan OVR estimasi.
TIER_BINS = [(85, "icon"), (75, "gold"), (65, "silver"), (0, "bronze")]
TIER_LABEL = {"icon": "ELITE", "gold": "GOLD", "silver": "SILVER", "bronze": "BRONZE"}

# Kuantil residual untuk verdict (0.15 / 0.85 -> 15% teratas & terbawah).
VERDICT_LOW_Q = 0.15
VERDICT_HIGH_Q = 0.85
