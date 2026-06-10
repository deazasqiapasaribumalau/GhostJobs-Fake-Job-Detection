import streamlit as st
import joblib
import json
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GhostJobs Detector",
    page_icon="👻",
    layout="centered"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_assets():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    model     = joblib.load(os.path.join(MODEL_DIR, "random_forest_tuned_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    for res in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        nltk.download(res, quiet=True)
    return config, model, vectorizer, scaler

config, model, vectorizer, scaler = load_assets()

THRESHOLD     = config["threshold_optimal"]
KOLOM_NUMERIK = config["kolom_numerik"]
KOLOM_TEKS    = config["kolom_teks"]

stemmer    = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    teks_gabung = " ".join([
        preprocess_teks(d.get(col, "")) for col in KOLOM_TEKS
    ])
    X_tfidf  = vectorizer.transform([teks_gabung])
    num_vals = np.array(extract_numerik(d)).reshape(1, -1)
    X_num    = scaler.transform(num_vals)
    X_final  = hstack([X_tfidf, csr_matrix(X_num)])
    prob     = model.predict_proba(X_final)[0][1]
    label    = int(prob >= THRESHOLD)
    return prob, label

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
}
.ghost-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.ghost-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #0f172a;
    margin: 0;
}
.ghost-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 0.3rem;
}
.result-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    text-align: center;
}
.result-fake {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
}
.result-real {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
}
.result-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-prob {
    font-size: 0.9rem;
    color: #64748b;
}
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}
.stButton > button {
    background: #0f172a;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1e293b;
}
.divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ghost-header">
    <p class="ghost-title">👻 GhostJobs Detector</p>
    <p class="ghost-sub">Deteksi lowongan kerja palsu menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Informasi Utama
st.markdown('<p class="section-label">Informasi Lowongan</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    title       = st.text_input("Judul Posisi", placeholder="e.g. Senior Data Analyst")
    location    = st.text_input("Lokasi", placeholder="e.g. Jakarta, Indonesia")
    department  = st.text_input("Departemen", placeholder="e.g. Data & Analytics")
    industry    = st.text_input("Industri", placeholder="e.g. Technology")
with col2:
    employment_type      = st.text_input("Tipe Pekerjaan", placeholder="e.g. Full-time")
    required_experience  = st.text_input("Pengalaman", placeholder="e.g. Mid-Senior level")
    required_education   = st.text_input("Pendidikan", placeholder="e.g. Bachelor's Degree")
    function_field       = st.text_input("Fungsi", placeholder="e.g. Engineering")

st.markdown('<p class="section-label">Deskripsi & Konten</p>', unsafe_allow_html=True)

company_profile = st.text_area("Profil Perusahaan", height=100, placeholder="Deskripsikan perusahaan...")
description     = st.text_area("Deskripsi Pekerjaan *", height=150, placeholder="Tulis deskripsi lengkap lowongan...")
requirements    = st.text_area("Persyaratan", height=100, placeholder="Kualifikasi dan persyaratan...")
benefits        = st.text_area("Benefit", height=80, placeholder="Tunjangan, asuransi, dll...")

st.markdown('<p class="section-label">Informasi Tambahan</p>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    telecommuting    = st.checkbox("Remote / WFH")
with col4:
    has_company_logo = st.checkbox("Ada Logo Perusahaan")
with col5:
    has_questions    = st.checkbox("Ada Pertanyaan Skrining")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🔍 Analisis Lowongan"):
    if not description.strip():
        st.warning("Mohon isi minimal kolom **Deskripsi Pekerjaan**.")
    else:
        with st.spinner("Menganalisis..."):
            data = {
                "title": title, "location": location, "department": department,
                "company_profile": company_profile, "description": description,
                "requirements": requirements, "benefits": benefits,
                "employment_type": employment_type,
                "required_experience": required_experience,
                "required_education": required_education,
                "industry": industry, "function": function_field,
                "telecommuting": int(telecommuting),
                "has_company_logo": int(has_company_logo),
                "has_questions": int(has_questions),
            }
            prob, label = predict(data)

        if label == 1:
            st.markdown(f"""
            <div class="result-box result-fake">
                <div class="result-label">⚠️ Lowongan Terindikasi PALSU</div>
                <div class="result-prob">Probabilitas penipuan: <strong>{prob*100:.1f}%</strong> (threshold: {THRESHOLD*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            st.error("**Hati-hati!** Model mendeteksi beberapa ciri lowongan tidak asli. Lakukan verifikasi lebih lanjut sebelum melamar.")
        else:
            st.markdown(f"""
            <div class="result-box result-real">
                <div class="result-label">✅ Lowongan Terindikasi ASLI</div>
                <div class="result-prob">Probabilitas penipuan: <strong>{prob*100:.1f}%</strong> (threshold: {THRESHOLD*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            st.success("Lowongan ini tampak **legitim** berdasarkan analisis model. Tetap waspada dan lakukan verifikasi mandiri.")

        with st.expander("ℹ️ Tentang hasil ini"):
            st.markdown(f"""
            - Model yang digunakan: **Random Forest (Tuned)**
            - Threshold optimal: **{THRESHOLD:.4f}**
            - Fitur yang dianalisis: TF-IDF teks + {len(KOLOM_NUMERIK)} fitur numerik
            - Hasil ini bersifat **prediktif**, bukan keputusan final. Selalu verifikasi secara mandiri.
            """)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#94a3b8; font-size:0.8rem;">GhostJobs · Tim PJK-GM088 · Capstone Pijak 2026</p>', unsafe_allow_html=True)
