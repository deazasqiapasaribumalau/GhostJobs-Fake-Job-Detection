import streamlit as st
import joblib
import json
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GhostJobs Detector",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    model      = joblib.load(os.path.join(MODEL_DIR, "random_forest_tuned_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    df_demo    = pd.read_csv(os.path.join(MODEL_DIR, "sample_demo.csv"))
    for res in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        nltk.download(res, quiet=True)
    stemmer    = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    return config, model, vectorizer, scaler, stemmer, stop_words, df_demo

config, model, vectorizer, scaler, stemmer, STOP_WORDS, df_demo = load_assets()

THRESHOLD     = config["threshold_optimal"]
KOLOM_NUMERIK = config["kolom_numerik"]
KOLOM_TEKS    = config["kolom_teks"]

# ── Helpers (tidak diubah) ────────────────────────────────────────────────────
def preprocess_teks(teks: str) -> str:
    if not isinstance(teks, str) or teks.strip() == "":
        return ""
    teks = teks.lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"\S+@\S+", " ", teks)
    teks = re.sub(r"\d+", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    tokens = word_tokenize(teks)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

def extract_numerik(d: dict) -> list:
    desc = d.get("description", "")
    return [
        int(d.get("telecommuting", 0)),
        int(d.get("has_company_logo", 0)),
        int(d.get("has_questions", 0)),
        len(d.get("description", "")),
        len(d.get("requirements", "")),
        len(d.get("benefits", "")),
        len(d.get("company_profile", "")),
        int(bool(re.search(r'\$|salary|compensation|pay|usd|eur|per hour|per annum', desc, re.I))),
        int(bool(re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', desc))),
        desc.count("!"),
        int(bool(re.search(r'urgent|immediately|apply now|limited slots|hurry|fast cash', desc, re.I))),
    ]

def predict(d: dict):
    teks_gabung = " ".join([preprocess_teks(d.get(col, "")) for col in KOLOM_TEKS])
    X_tfidf  = vectorizer.transform([teks_gabung])
    num_vals = np.array(extract_numerik(d)).reshape(1, -1)
    X_num    = scaler.transform(num_vals)
    X_final  = hstack([X_tfidf, csr_matrix(X_num)])
    prob     = model.predict_proba(X_final)[0][1]
    label    = int(prob >= THRESHOLD)
    return prob, label

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
            
/* sembunyiin nav button duplikat di sidebar */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    position: absolute !important;
    opacity: 0 !important;
    height: 100% !important;
    width: 100% !important;
    top: 0 !important;
    left: 0 !important;
    cursor: pointer !important;
    z-index: 10 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] {
    position: relative !important;
    margin-top: -44px !important;
    height: 44px !important;
}

/* ── Layout background ── */
.stApp { background: #eff4fb; }
.block-container { max-width: 740px !important; padding: 2rem 1.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #d7e3ef;
}
[data-testid="stSidebar"] * { color: #6b87a4 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #214968 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
            
            /* ─────────────────────────────
   PAGE TRANSITION ANIMATION
───────────────────────────── */

.fade-page{
    animation: fadePage 1.2s cubic-bezier(0.22, 1, 0.36, 1);
}

@keyframes fadePage{
    from{
        opacity:0;
        transform:translateY(60px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

/* efek hover card */
.input-card,
.about-card,
.result-wrap{
    transition: all 0.25s ease;
}

.input-card:hover,
.about-card:hover{
    transform: translateY(-2px);
    border-color:#3e7bb7;
}

/* efek muncul bertahap */
.gj-header{
    animation: fadePage 0.8s ease;
}

/* smooth sidebar */
[data-testid="stSidebar"]{
    transition: all 0.3s ease;
}

/* ── Hero header ── */
.gj-header {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-bottom: 0.5rem;
}
.gj-logo-wrap {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.75rem;
}
.gj-logo-box {
    width: 44px; height: 44px;
    background: #eaf1f9;
    border: 1px solid #c0d2e5;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.gj-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #214968;
    letter-spacing: -0.5px;
}
.gj-title span { color: #2d6faa; }
.gj-sub {
    font-size: 14px;
    color: #6f87a1;
    margin-top: 0;
}

/* ── Step label ── */
.step-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #2d6faa;
    margin: 1.5rem 0 0.6rem;
}
.step-num {
    width: 20px; height: 20px;
    background: #eaf1f9;
    border: 1px solid #c0d2e5;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    color: #2d6faa;
}

/* ── Input cards ── */
.input-card {
    background: #ffffff;
    border: 1px solid #d7e3ef;
    border-radius: 12px;
    padding: 1.25rem 1.25rem 0.75rem;
    margin-bottom: 0.75rem;
}

/* ── Streamlit inputs override ── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
.stSelectbox > div > div > div {
    background: #ffffff !important;
    border: 1px solid #d7e3ef !important;
    border-radius: 8px !important;
    color: #214968 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: #3e7bb7 !important;
    box-shadow: 0 0 0 2px rgba(62,123,183,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #6f87a1 !important;
    font-size: 13px !important;
}

/* ── Checkbox ── */
.stCheckbox label p { color: #4f6f8b !important; font-size: 14px !important; }

/* ── Analyze button ── */
div[data-testid="stButton"] > button {
    background: #eef4fb !important;
    color: #214968 !important;
    border: 1px solid #c7d7e6 !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s, border-color 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #e4edf7 !important;
    border-color: #9ebada !important;
}

/* ── Result card ── */
.result-wrap {
    border-radius: 14px;
    overflow: hidden;
    margin: 1.5rem 0 0.75rem;
    border: 1px solid #d7e3ef;
}
.result-top-fake {
    background: #fff3f3;
    border-left: 4px solid #E24B4A;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.result-top-real {
    background: #f3fbf4;
    border-left: 4px solid #4a9922;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 6px;
}
.result-verdict-fake { color: #c94d4d; }
.result-verdict-real { color: #368d4d; }
.result-pct-fake { font-size: 13px; color: #9b4d4d; }
.result-pct-real { font-size: 13px; color: #4a8b57; }
.bar-outer {
    height: 5px;
    background: #d7e3ef;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 8px;
    max-width: 260px;
}
.bar-fake { height: 100%; background: #E24B4A; border-radius: 3px; }
.bar-real { height: 100%; background: #4a9922; border-radius: 3px; }
.result-body {
    background: #ffffff;
    padding: 1rem 1.5rem;
    border-top: 1px solid #d7e3ef;
}
.flag-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    font-size: 13px;
    color: #b35757;
    border-bottom: 1px solid #e6eef7;
}
.safe-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    font-size: 13px;
    color: #4e8f5a;
    border-bottom: 1px solid #e6eef7;
}
.flag-item i { color: #E24B4A; flex-shrink: 0; }
.safe-item i { color: #4a9922; flex-shrink: 0; }
.result-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding: 0.75rem 1.5rem;
    background: #ffffff;
    border-top: 1px solid #d7e3ef;
}
.meta-chip {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    background: #eef4fb;
    color: #6f87a1;
    border: 1px solid #d7e3ef;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #fff8e9;
    border: 1px solid #f0dfb0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 12px;
    color: #8a7638;
    margin-top: 0.75rem;
}

/* ── Sidebar nav indicator ── */
.sidebar-flag {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    font-size: 12px;
    color: #6f87a1 !important;
    border-bottom: 1px solid #e6eef7;
}
.dot-r { width:7px; height:7px; background:#d35d5d; border-radius:50%; flex-shrink:0; }
.dot-g { width:7px; height:7px; background:#5ca66a; border-radius:50%; flex-shrink:0; }

/* ── About page ── */
.about-card {
    background: #ffffff;
    border: 1px solid #d7e3ef;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.about-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid #e6eef7;
}
.about-label { color: #6f87a1; }
.about-val   { color: #214968; font-weight: 500; }
.member-row  {
    padding: 8px 0;
    border-bottom: 1px solid #e6eef7;
}
.member-name { font-size: 13px; font-weight: 600; color: #214968; }
.member-sub  { font-size: 11px; color: #6f87a1; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
        padding: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        display: flex !important;
        align-items: center !important;
        padding: 10px 12px !important;
        width: 100%;
        transition: background 0.15s;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #eef4fb !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: rgba(62,123,183,0.14) !important;
    }
    [data-testid="stSidebar"] .stRadio p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #214968 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Brand ──────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 14px 16px 10px;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="38" height="38" rx="10" fill="#eef4fb"/>
                <rect x="0.5" y="0.5" width="37" height="37" rx="9.5" stroke="#c0d2e5" stroke-opacity="0.8"/>
                <path d="M19 7L9 11V19C9 24.5 13.5 29.5 19 31C24.5 29.5 29 24.5 29 19V11L19 7Z"
                      fill="#dfeaf5" stroke="#5f90bf" stroke-width="1.2"/>
                <circle cx="17.5" cy="18" r="4" stroke="#4f86ba" stroke-width="1.5" fill="none"/>
                <line x1="20.5" y1="21" x2="23" y2="23.5" stroke="#4f86ba" stroke-width="1.8" stroke-linecap="round"/>
                <circle cx="16" cy="16.5" r="1" fill="#9ec0de" opacity="0.8"/>
            </svg>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:800; color:#214968; letter-spacing:-0.5px; line-height:1.0; margin-top:-1px;">
                    Ghost<span style="color:#2d6faa;">Jobs</span>
                </div>
                <div style="font-size:10px; color:#6f87a1; letter-spacing:0.08em; text-transform:uppercase; margin-top:1px; font-weight:800;">
                    Fake Job Detector
                </div>
            </div>
        </div>
        <div style="display:inline-flex; align-items:center; gap:6px; background:#eef4fb; border:1px solid #d7e3ef; border-radius:20px; padding:4px 10px; margin-top:2px;">
            <div style="width:6px;height:6px;background:#22c55e;border-radius:50%;"></div>
            <span style="font-size:10px;color:#4f8b63;letter-spacing:0.05em;">Model aktif</span>
        </div>
    </div>
    <div style="height:1px;background:#d7e3ef;margin:0 16px 16px;"></div>
    """, unsafe_allow_html=True)

    # ── Nav label ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:10px;font-weight:800;color:#6f87a1;letter-spacing:0.12em;text-transform:uppercase;padding:0 16px;margin-bottom:4px;">
        Navigasi
    </div>
    """, unsafe_allow_html=True)

    # ── Nav items dengan icon manual ───────────────────────────
    NAV_ITEMS = ["Deteksi Lowongan", "Panduan", "Tentang Sistem"]
    NAV_ICONS = {
        "Deteksi Lowongan": """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>""",
        "Panduan":          """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>""",
        "Tentang Sistem":   """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>""",
    }

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Deteksi Lowongan"

    for item in NAV_ITEMS:
        is_active = st.session_state.current_page == item
        active_bg    = "background:rgba(62,123,183,0.12); border:1px solid #bfd1e3;" if is_active else "background:transparent; border:1px solid transparent;"
        icon_bg      = "background:#dfeaf5;" if is_active else "background:#eef4fb;"
        icon_color   = "#2d6faa" if is_active else "#6f87a1"
        label_color  = "#214968" if is_active else "#6f87a1"
        badge        = '<span style="margin-left:auto;background:#edf4fb;color:#2d6faa;font-size:10px;padding:2px 8px;border-radius:10px;font-weight:500;border:1px solid #d7e3ef;">Aktif</span>' if is_active else ""

        icon_svg = NAV_ICONS[item].replace('stroke="currentColor"', f'stroke="{icon_color}"')

        st.markdown(f"""
        <div onclick="void(0)" style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;{active_bg};margin:0 4px 3px;cursor:pointer;">
            <div style="width:30px;height:30px;border-radius:8px;{icon_bg};display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                {icon_svg}
            </div>
            <span style="font-size:13px;font-weight:{'700' if is_active else '600'};color:{label_color};">{item}</span>
            {badge}
        </div>
        """, unsafe_allow_html=True)

        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state.current_page = item
            st.rerun()

    page = st.session_state.current_page

    # ── Divider + Mode Demo ────────────────────────────────────
    st.markdown("""
    <div style="height:1px;background:#d7e3ef;margin:12px 16px;"></div>
    <div style="font-size:10px;font-weight:800;color:#6f87a1;letter-spacing:0.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;">
        Mode Demo
    </div>
    """, unsafe_allow_html=True)

    use_demo = st.toggle("Gunakan contoh dari database", value=False)

    # ── Divider + Indikator ────────────────────────────────────
    st.markdown("""
    <div style="height:1px;background:#d7e3ef;margin:12px 16px;"></div>
    <div style="font-size:10px;font-weight:800;color:#6f87a1;letter-spacing:0.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;">
        Indikator Lowongan Palsu
    </div>
    """, unsafe_allow_html=True)

    flags = [
        ("Gaji tidak realistis / tidak jelas", True),
        ("Bahasa terlalu memaksa / urgent",    True),
        ("Ada email di deskripsi",             True),
        ("Tidak ada logo perusahaan",          True),
        ("Banyak tanda seru !!!",              True),
        ("Ada profil perusahaan lengkap",      False),
        ("Ada pertanyaan skrining",            False),
        ("Benefit tercantum jelas",            False),
    ]
    for text, is_red in flags:
        dot_color = "#c04040" if is_red else "#2ecc71"
        text_color = "#b06060" if is_red else "#4a9a6a"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 16px;font-size:12px;color:{text_color};border-bottom:1px solid #0d1828;">
            <div style="width:6px;height:6px;background:{dot_color};border-radius:50%;flex-shrink:0;"></div>
            {text}
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div style="height:1px;background:#d7e3ef;margin:12px 16px;"></div>
    <div style="text-align:center;padding:0 16px 20px;">
        <div style="font-size:10px;color:#6f87a1;font-weight:600;">Tim PJK-GM088</div>
        <div style="font-size:10px;color:#91a6bd;margin-top:2px;font-weight:600;">Capstone Pijak × IBM SkillsBuild 2026</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DETEKSI
# ══════════════════════════════════════════════════════════════
if page == "Deteksi Lowongan":

    st.markdown('<div class="fade-page">', unsafe_allow_html=True)
 
    st.markdown("""
    <div class="gj-header">
        <div class="gj-logo-wrap">
            <svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="46" height="46" rx="12" fill="#eef4fb"/>
                <rect x="0.5" y="0.5" width="45" height="45" rx="11.5" stroke="#c0d2e5" stroke-opacity="0.95"/>
                <path d="M23 8L10 13V22C10 29 15.5 35.5 23 37.5C30.5 35.5 36 29 36 22V13L23 8Z"
                        fill="#dfeaf5" stroke="#5f90bf" stroke-width="1.3"/>
                <circle cx="21" cy="22" r="5" stroke="#4f86ba" stroke-width="1.8" fill="none"/>
                <line x1="24.8" y1="25.8" x2="28" y2="29" stroke="#4f86ba" stroke-width="2.2" stroke-linecap="round"/>
                <circle cx="19.5" cy="20" r="1.2" fill="#9ec0de" opacity="0.8"/>
            </svg>
            <div class="gj-title">Ghost<span>Jobs</span></div>
        </div>
        <div class="gj-sub">Deteksi lowongan kerja palsu berbasis AI — akurat &amp; transparan</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Mode Demo Database ─────────────────────────────────────
    selected_row = None
    if use_demo:
        st.markdown('<div class="step-label" style="margin-top:0;"><span class="step-num" style="background:#0c3020;border-color:#1a6a40;color:#2ecc71;">▶</span> Pilih contoh lowongan dari database</div>', unsafe_allow_html=True)

        df_demo['_label_text'] = df_demo.apply(
            lambda r: f"[{'PALSU' if r['fraudulent']==1 else 'ASLI '}]  {str(r.get('title','(no title)'))[:55]}  —  {str(r.get('location',''))[:25]}",
            axis=1
        )
        pilihan = st.selectbox(
            "Pilih lowongan",
            options=df_demo.index.tolist(),
            format_func=lambda i: df_demo.loc[i, '_label_text'],
            label_visibility="collapsed"
        )
        selected_row = df_demo.loc[pilihan]

        label_asli = "PALSU" if selected_row['fraudulent'] == 1 else "ASLI"
        warna_label = "#e24b4a" if selected_row['fraudulent'] == 1 else "#2ecc71"
        st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:8px;
                    padding:8px 14px;margin-bottom:0.75rem;font-size:12px;color:#6f87a1;
                    display:flex;gap:16px;flex-wrap:wrap;">
            <span>Label asli: <strong style="color:{warna_label};">{label_asli}</strong></span>
                <span>Industri: <strong style="color:#214968;">{str(selected_row.get('industry','—'))}</strong></span>
                <span>Lokasi: <strong style="color:#214968;">{str(selected_row.get('location','—'))}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        st.info("Form di bawah sudah terisi otomatis dari dataset. Klik **Analisis** untuk melihat prediksi model.")

    def sv(col):
        """Helper: ambil value dari selected_row, return '' kalau None/NaN."""
        if selected_row is None:
            return ""
        val = selected_row.get(col, "")
        return "" if pd.isna(val) else str(val)

    def sb(col):
        """Helper: ambil boolean dari selected_row."""
        if selected_row is None:
            return False
        try:
            return bool(int(selected_row.get(col, 0)))
        except Exception:
            return False

    # ── Bagian 1: Informasi Lowongan (Wajib) ──────────────────
    st.markdown('<div class="step-label"><span class="step-num">1</span> Informasi Lowongan <span style="color:#1a4a6a;font-size:10px;">(wajib diisi)</span></div>', unsafe_allow_html=True)
 
    title = st.text_input("Judul posisi", value=sv("title"), placeholder="e.g. Data Analyst, Software Engineer")
    description = st.text_area(
        "Deskripsi pekerjaan",
        value=sv("description"),
        height=150,
        placeholder="Salin deskripsi lengkap lowongan di sini — tanggung jawab, tugas, ekspektasi..."
    )
    requirements = st.text_area(
        "Persyaratan & kualifikasi",
        value=sv("requirements"),
        height=100,
        placeholder="Skill, pengalaman, sertifikasi, pendidikan yang dibutuhkan..."
    )
 
    # ── Bagian 2: Informasi Perusahaan (Disarankan) ────────────
    st.markdown('<div class="step-label"><span class="step-num">2</span> Informasi Perusahaan <span style="color:#1a4a6a;font-size:10px;">(disarankan)</span></div>', unsafe_allow_html=True)
 
    with st.expander("Isi informasi perusahaan", expanded=use_demo):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            company_name = st.text_input("Nama perusahaan", value=sv("company_name"), placeholder="e.g. PT Maju Bersama")
            industry     = st.text_input("Industri",        value=sv("industry"),     placeholder="e.g. Technology, Finance")
        with col_p2:
            location     = st.text_input("Lokasi",          value=sv("location"),     placeholder="e.g. Jakarta / Remote")
        company_profile  = st.text_area("Profil perusahaan", value=sv("company_profile"), height=80,
                                        placeholder="Visi misi, budaya kerja, deskripsi singkat perusahaan...")
 
    # ── Bagian 3: Informasi Tambahan (Opsional) ────────────────
    st.markdown('<div class="step-label"><span class="step-num">3</span> Informasi Tambahan <span style="color:#1a4a6a;font-size:10px;">(opsional)</span></div>', unsafe_allow_html=True)
 
    with st.expander("Isi informasi tambahan", expanded=use_demo):
        col_a, col_b = st.columns(2)
        with col_a:
            benefits            = st.text_area("Benefit & tunjangan", value=sv("benefits"), height=70,
                                               placeholder="Gaji, asuransi, THR, fasilitas lainnya...")
            required_experience = st.text_input("Level pengalaman",   value=sv("required_experience"), placeholder="e.g. Entry Level, Mid-Senior")
            department          = st.text_input("Departemen",         value=sv("department"),          placeholder="e.g. Engineering, Marketing")
        with col_b:
            required_education  = st.text_input("Pendidikan",         value=sv("required_education"),  placeholder="e.g. S1, Bachelor's Degree")
            function_field      = st.text_input("Fungsi pekerjaan",   value=sv("function"),            placeholder="e.g. Engineering, Sales")
            employment_type     = st.text_input("Tipe pekerjaan",     value=sv("employment_type"),     placeholder="e.g. Full-time, Part-time")
 
    # ── Bagian 4: Atribut ──────────────────────────────────────
    st.markdown('<div class="step-label"><span class="step-num">4</span> Atribut Lowongan</div>', unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        has_company_logo = st.checkbox("Ada logo perusahaan",   value=sb("has_company_logo"))
    with col2:
        telecommuting    = st.checkbox("Posisi Remote / WFH",   value=sb("telecommuting"))
    with col3:
        has_questions    = st.checkbox("Ada pertanyaan skrining", value=sb("has_questions"))
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Tombol Deteksi ─────────────────────────────────────────
    if st.button("Analisis Lowongan Sekarang"):
        if not description.strip():
            st.warning("Isi minimal kolom **Deskripsi Pekerjaan** untuk memulai analisis.")
        else:
            with st.spinner("Sedang menganalisis lowongan..."):
                data = {
                    "title": title,
                    "location": location,
                    "department": department,
                    "company_profile": company_profile,
                    "description": description,
                    "requirements": requirements,
                    "benefits": benefits,
                    "employment_type": employment_type,
                    "required_experience": required_experience,
                    "required_education": required_education,
                    "industry": industry,
                    "function": function_field,
                    "telecommuting": int(telecommuting),
                    "has_company_logo": int(has_company_logo),
                    "has_questions": int(has_questions),
                }
                prob, label = predict(data)
 
            prob_pct = prob * 100

            if prob >= 0.8:
                risk_level = "🔴 High Risk"
            elif prob >= 0.5:
                risk_level = "🟡 Medium Risk"
            else:
                risk_level = "🟢 Low Risk"

            # ── Perbandingan label asli vs prediksi (mode demo) ─
            if use_demo and selected_row is not None:
                label_asli_val = int(selected_row['fraudulent'])
                benar = label == label_asli_val
                status_txt   = "✔ Prediksi BENAR" if benar else "✘ Prediksi SALAH"
                status_warna = "#2ecc71" if benar else "#e24b4a"
                la_txt  = "PALSU" if label_asli_val == 1 else "ASLI"
                pr_txt  = "PALSU" if label == 1 else "ASLI"
                la_warn = "#e24b4a" if label_asli_val == 1 else "#2ecc71"
                pr_warn = "#e24b4a" if label == 1 else "#2ecc71"
                st.markdown("""
                <hr>

                <div style="
                text-align:center;
                font-size:13px;
                color:#4a7fa8;
                padding:15px;
                ">
                GhostJobs Detector © 2026 <br>
                Capstone Project - Fake Job Detection using Machine Learning
                </div>
                """, unsafe_allow_html=True)

            if label == 1:
                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-top-fake">
                        <div style="font-size:32px;">🚨</div>
                        <div style="flex:1;">
                            <div class="result-verdict result-verdict-fake">Lowongan terindikasi PALSU</div>
                            <div class="result-pct-fake">Probabilitas penipuan: <strong>{prob_pct:.1f}%</strong> &nbsp;·&nbsp; Threshold: {THRESHOLD*100:.1f}%</div>
                            <div class="bar-outer"><div class="bar-fake" style="width:{min(prob_pct,100):.1f}%"></div></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
 
                red_flags = []
                if not has_company_logo:
                    red_flags.append("Tidak ada logo perusahaan")
                if not has_questions:
                    red_flags.append("Tidak ada pertanyaan skrining")
                if not benefits.strip():
                    red_flags.append("Informasi benefit tidak diisi")
                if not company_profile.strip():
                    red_flags.append("Tidak ada profil perusahaan")
                desc_lower = description.lower()
                if re.search(r'urgent|immediately|apply now|limited slots|hurry|fast cash', desc_lower):
                    red_flags.append("Deskripsi mengandung kata-kata memaksa / urgent")
                if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', description):
                    red_flags.append("Deskripsi mengandung alamat email langsung")
                if description.count("!") > 3:
                    red_flags.append(f"Terlalu banyak tanda seru ({description.count('!')} kali)")
 
                if red_flags:
                    flags_html = "".join([
                        f'<div class="flag-item">⚠ {f}</div>' for f in red_flags
                    ])
                    st.markdown(f"""
                    <div class="result-body">
                        <div style="font-size:11px;color:#5a3030;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Indikator yang terdeteksi</div>
                        {flags_html}
                    </div>
                    """, unsafe_allow_html=True)
 
                st.markdown(f"""
                    <div class="result-meta">
                        <span class="meta-chip">Random Forest</span>
                        <span class="meta-chip">TF-IDF + Numerik</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            # ── Hasil REAL ─────────────────────────────────────
            else:
                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-top-real">
                        <div style="font-size:32px;">✅</div>
                        <div style="flex:1;">
                            <div class="result-verdict result-verdict-real">Lowongan terindikasi ASLI</div>
                            <div class="result-pct-real">Probabilitas penipuan: <strong>{prob_pct:.1f}%</strong> &nbsp;·&nbsp; Threshold: {THRESHOLD*100:.1f}%</div>
                            <div class="bar-outer"><div class="bar-real" style="width:{100-prob_pct:.1f}%"></div></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
 
                safe_signals = []
                if has_company_logo:        safe_signals.append("Memiliki logo perusahaan")
                if has_questions:           safe_signals.append("Ada pertanyaan skrining yang terstruktur")
                if benefits.strip():        safe_signals.append("Informasi benefit tersedia")
                if company_profile.strip(): safe_signals.append("Profil perusahaan dicantumkan")
 
                if safe_signals:
                    sigs_html = "".join([
                        f'<div class="safe-item">✔ {s}</div>' for s in safe_signals
                    ])
                    st.markdown(f"""
                    <div class="result-body">
                        <div style="font-size:11px;color:#2a5a2a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Sinyal positif ditemukan</div>
                        {sigs_html}
                    </div>
                    """, unsafe_allow_html=True)
 
                st.markdown(f"""
                    <div class="result-meta">
                        <span class="meta-chip">Random Forest</span>
                        <span class="meta-chip">TF-IDF + Numerik</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
            # ── Detail teknis (compact, inline HTML) ───────────
            with st.expander("Detail teknis analisis"):
                st.markdown(f"""
                <div style="display:flex;gap:12px;flex-wrap:wrap;padding:4px 0;">
                    <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:8px;padding:8px 14px;min-width:100px;">
                        <div style="font-size:10px;color:#6f87a1;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Model</div>
                        <div style="font-size:13px;font-weight:600;color:#214968;">Random Forest</div>
                    </div>
                        <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:8px;padding:8px 14px;min-width:100px;">
                            <div style="font-size:10px;color:#6f87a1;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Probabilitas</div>
                            <div style="font-size:13px;font-weight:600;color:#214968;">{prob_pct:.2f}%</div>
                    </div>
                        <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:8px;padding:8px 14px;min-width:100px;">
                            <div style="font-size:10px;color:#6f87a1;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">Threshold</div>
                            <div style="font-size:13px;font-weight:600;color:#214968;">{THRESHOLD*100:.2f}%</div>
                    </div>
                </div>
                    <div style="font-size:11px;color:#6f87a1;margin-top:8px;">
                    Fitur teks: TF-IDF ({len(KOLOM_TEKS)} kolom) · Fitur numerik: {len(KOLOM_NUMERIK)} fitur
                </div>
                """, unsafe_allow_html=True)
 
            st.markdown("""
            <div class="disclaimer">
                ⚠ Hasil ini dihasilkan oleh model ML dan tidak menjamin kebenaran 100%.
                Selalu verifikasi mandiri melalui situs resmi perusahaan sebelum melamar.
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PANDUAN
# ══════════════════════════════════════════════════════════════
elif page == "Panduan":

    st.markdown('<div class="fade-page">', unsafe_allow_html=True)

    st.markdown("""
    <div class="gj-header">
        <div class="gj-title">Panduan <span>Penggunaan</span></div>
        <div class="gj-sub">Cara menggunakan GhostJobs Detector dengan efektif</div>
    </div>
    """, unsafe_allow_html=True)
 

    steps = [
        ("Buka halaman Deteksi", "Klik menu 'Deteksi Lowongan' di sidebar."),
        ("Tempel deskripsi lowongan", "Salin teks deskripsi dan persyaratan dari lowongan yang ingin diperiksa, lalu tempel ke kolom yang tersedia."),
        ("Tandai atribut", "Centang apakah lowongan remote, ada logo perusahaan, dan ada pertanyaan skrining."),
        ("Isi info opsional (jika ada)", "Makin lengkap datanya, makin akurat hasilnya. Tapi tidak wajib."),
        ("Klik Analisis", "Tekan tombol dan tunggu hasil — biasanya kurang dari 1 detik."),
        ("Baca hasil dengan bijak", "Gunakan hasil sebagai acuan, bukan keputusan final. Tetap verifikasi mandiri."),
    ]

    for i, (t, d) in enumerate(steps, 1):
        st.markdown(f"""
        <div style="display:flex;gap:12px;align-items:flex-start;background:#ffffff;border:1px solid #d7e3ef;border-radius:10px;padding:1rem 1.25rem;margin-bottom:0.6rem;">
            <div style="width:24px;height:24px;background:#eef4fb;border:1px solid #d7e3ef;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;color:#2d6faa;flex-shrink:0;margin-top:2px;">{i}</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;color:#214968;margin-bottom:3px;">{t}</div>
                <div style="font-size:13px;color:#6f87a1;line-height:1.6;">{d}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top:1.25rem;">
        ⚠ GhostJobs Detector adalah alat bantu, bukan pengganti penilaian manusia.
        Jangan pernah membayar biaya atau memberikan data sensitif sebelum memverifikasi keaslian perusahaan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TENTANG
# ══════════════════════════════════════════════════════════════
elif page == "Tentang Sistem":

    st.markdown('<div class="fade-page">', unsafe_allow_html=True)

    st.markdown("""
    <div class="gj-header">
        <div class="gj-title">Tentang <span>Sistem</span></div>
        <div class="gj-sub">Informasi teknis &amp; tim pengembang GhostJobs</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Baris 1: Pipeline + Spesifikasi ───────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
        <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:12px;padding:0.65rem;">
            <div style="font-size:13px;font-weight:600;color:#2d6faa;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;margin-top:8px;margin-left:12px;">Pipeline Model</div>
    """, unsafe_allow_html=True)

    pipeline = [
        ("01", "Text Preprocessing",     "Bersihkan HTML/URL/email → tokenisasi → stopword removal → stemming"),
        ("02", "Feature Engineering",    "TF-IDF (12 kolom teks) + 11 fitur numerik custom"),
        ("03", "Class Imbalance",        "SMOTE (Synthetic Minority Oversampling)"),
        ("04", "Model Training",         "Logistic Regression & Random Forest + RandomizedSearchCV"),
        ("05", "Threshold Optimization", f"Optimal threshold: {THRESHOLD:.4f} (F1-Score)"),
    ]
    for num, pt, pd_ in pipeline:
        st.markdown(f"""
        <div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #0d1828;align-items:flex-start;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:10px;font-weight:700;color:#2d6faa;
                        background:#eef4fb;border:1px solid #d7e3ef;border-radius:4px;
                        padding:2px 5px;flex-shrink:0;margin-top:1px;letter-spacing:0.04em;">{num}</div>
            <div>
                <div style="font-weight:600;color:#214968;font-size:13px;line-height:1.3;">{pt}</div>
                <div style="color:#6f87a1;font-size:11px;margin-top:2px;line-height:1.5;">{pd_}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Spesifikasi ────────────────────────────────────────────
    specs = [
        ("Model",         "Random Forest Tuned"),
        ("Tuning",        "RandomizedSearchCV"),
        ("Fitur teks",    "12 kolom TF-IDF"),
        ("Fitur numerik", "11 fitur custom"),
        ("Imbalance",     "SMOTE"),
        ("Threshold",     f"{THRESHOLD:.4f}"),
        ("Dataset",       "fake_job_postings.csv"),
        ("Sumber",        "Kaggle"),
    ]

    st.markdown("""
        <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:12px;padding:0.65rem;">
            <div style="font-size:13px;font-weight:600;color:#2d6faa;letter-spacing:0.12em;
                        text-transform:uppercase;margin-bottom:12px;margin-top:8px;margin-left:12px;">Spesifikasi</div>
    """, unsafe_allow_html=True)

    for lbl, val in specs:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:7px 0;border-bottom:1px solid #0d1828;font-size:12px;">
            <span style="color:#6f87a1;">{lbl}</span>
            <span style="color:#214968;font-weight:500;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Tim Pengembang ─────────────────────────────────────────
    members = [
        ("Dea Zasqia P. Malau", "APC322D6X0291", "Streamlit & Integrasi",       "DZ"),
        ("Dian Nazira",         "APC322D6X0292", "EDA & Data Quality",           "DN"),
        ("Adinda Muarriva",     "APC322D6X0294", "Preprocessing & Feature Eng.", "AM"),
        ("Khairun Nisa",        "APC322D6X0409", "Model Training & Evaluasi",    "KN"),
    ]

    st.markdown("""
    <div style="background:#ffffff;border:1px solid #d7e3ef;border-radius:12px;padding:0.75rem;margin-bottom:12px;">
        <div style="font-size:13px;font-weight:600;color:#2d6faa;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:4px;margin-top:8px;margin-left:12px;">Tim Pengembang</div>
        <div style="font-size:11px;color:#6f87a1;margin-bottom:12px;margin-left:12px;">
            PJK-GM088 · Capstone Pijak × IBM SkillsBuild 2026
        </div>
    """, unsafe_allow_html=True)

    for name, cid, role, initials in members:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #e6eef7;">
            <div style="width:36px;height:36px;border-radius:8px;
                        background:#eef4fb;border:1px solid #d7e3ef;
                        display:flex;align-items:center;justify-content:center;
                        font-family:'Space Grotesk',sans-serif;font-size:11px;
                        font-weight:700;color:#2d6faa;flex-shrink:0;">{initials}</div>
            <div style="flex:1;min-width:0;">
                <div style="font-weight:600;color:#214968;font-size:13px;">{name}</div>
                <div style="font-size:11px;color:#6f87a1;margin-top:1px;">{cid}</div>
            </div>
            <div style="font-size:11px;color:#2d6faa;background:#eef4fb;
                        border:1px solid #d7e3ef;border-radius:20px;
                        padding:3px 10px;white-space:nowrap;">{role}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)