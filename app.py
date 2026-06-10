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
    for res in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        nltk.download(res, quiet=True)
    stemmer    = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    return config, model, vectorizer, scaler, stemmer, stop_words

config, model, vectorizer, scaler, stemmer, STOP_WORDS = load_assets()

THRESHOLD     = config["threshold_optimal"]
KOLOM_NUMERIK = config["kolom_numerik"]
KOLOM_TEKS    = config["kolom_teks"]

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

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default streamlit header/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1B2A !important;
    border-right: 1px solid #1B2B3B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio label { color: #94A3B8 !important; }
[data-testid="stSidebar"] hr { border-color: #1B2B3B !important; }

/* Main background */
.stApp { background: #F8FAFC; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: "👻";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
}
.page-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.page-header p {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(45,156,219,0.2);
    border: 1px solid rgba(45,156,219,0.4);
    color: #2D9CDB;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* Section cards */
.section-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title span {
    width: 20px; height: 20px;
    background: #0D1B2A;
    color: white;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
}

/* Result boxes */
.result-fake {
    background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
    border: 2px solid #FC8181;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-real {
    background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
    border: 2px solid #68D391;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-fake .result-title { color: #C53030; }
.result-real .result-title { color: #276749; }
.result-sub { font-size: 0.9rem; color: #4A5568; }

/* Probability meter */
.prob-meter {
    background: #E2E8F0;
    border-radius: 100px;
    height: 8px;
    margin: 1rem auto;
    max-width: 300px;
    overflow: hidden;
}
.prob-fill-fake {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #F6AD55, #FC8181, #E53E3E);
    transition: width 0.6s ease;
}
.prob-fill-real {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #68D391, #38A169);
    transition: width 0.6s ease;
}

/* Red flags */
.flag-box {
    background: #FFF5F5;
    border-left: 3px solid #FC8181;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.875rem;
    color: #742A2A;
}
.safe-box {
    background: #F0FFF4;
    border-left: 3px solid #68D391;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.875rem;
    color: #22543D;
}

/* Analyze button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0D1B2A, #1B3A5C) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
}

/* Sidebar guide steps */
.guide-step {
    background: rgba(45,156,219,0.08);
    border: 1px solid rgba(45,156,219,0.2);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}
.guide-step-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #2D9CDB !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.guide-step-text {
    font-size: 0.85rem;
    color: #CBD5E1 !important;
    margin-top: 0.2rem;
}

/* Redflag indicators */
.indicator-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0;
    font-size: 0.85rem;
    color: #4A5568;
    border-bottom: 1px solid #F1F5F9;
}
.dot-red { width:8px; height:8px; background:#FC8181; border-radius:50%; flex-shrink:0; }
.dot-green { width:8px; height:8px; background:#68D391; border-radius:50%; flex-shrink:0; }

/* Input styling */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2D9CDB !important;
    box-shadow: 0 0 0 3px rgba(45,156,219,0.1) !important;
}

/* Disclaimer */
.disclaimer {
    background: #FFFBEB;
    border: 1px solid #F6E05E;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #744210;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:700; color:#F1F5F9;">
            👻 GhostJobs
        </div>
        <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">Fake Job Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["🔍  Deteksi Lowongan", "📖  Panduan Penggunaan", "ℹ️  Tentang Sistem"], label_visibility="collapsed")

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem; font-weight:600; color:#475569; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.75rem;">
        Indikator Lowongan Palsu
    </div>
    """, unsafe_allow_html=True)

    flags = [
        ("Tidak ada info gaji yang jelas", True),
        ("Bahasa terlalu memaksa / urgent", True),
        ("Alamat email di deskripsi kerja", True),
        ("Tidak ada logo perusahaan", True),
        ("Banyak tanda seru (!!!)", True),
        ("Ada profil perusahaan lengkap", False),
        ("Ada pertanyaan skrining", False),
        ("Informasi benefit jelas", False),
    ]
    for text, is_red in flags:
        dot = "dot-red" if is_red else "dot-green"
        icon = "🚨" if is_red else "✅"
        st.markdown(f'<div class="indicator-row"><span class="{dot}"></span>{icon} {text}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; text-align:center; padding-top:0.5rem;">
        Tim PJK-GM088 · Capstone Pijak 2026<br>
        <span style="color:#334155;">Random Forest · TF-IDF + Numerik</span>
    </div>
    """, unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# PAGE 1: DETEKSI
# ══════════════════════════════════════════════════════════════
if "Deteksi" in page:

    st.markdown("""
    <div class="page-header">
        <div class="badge">Machine Learning · Random Forest</div>
        <h1>Deteksi Lowongan Kerja Palsu</h1>
        <p>Masukkan detail lowongan kerja yang ingin kamu periksa. Semakin lengkap data yang diisi, semakin akurat hasil analisisnya.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Info Dasar ──────────────────────────────────
    st.markdown("""
    <div class="section-card">
        <div class="section-title"><span>1</span> Informasi Dasar Lowongan</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            title      = st.text_input("Judul Posisi", placeholder="e.g. Data Analyst, Software Engineer")
            location   = st.text_input("Lokasi", placeholder="e.g. Jakarta, Indonesia / Remote")
            department = st.text_input("Departemen", placeholder="e.g. Engineering, Marketing")
            industry   = st.text_input("Industri", placeholder="e.g. Technology, Finance")
        with col2:
            employment_type     = st.text_input("Tipe Pekerjaan", placeholder="e.g. Full-time, Part-time, Contract")
            required_experience = st.text_input("Level Pengalaman", placeholder="e.g. Entry Level, Mid-Senior")
            required_education  = st.text_input("Pendidikan", placeholder="e.g. Bachelor's Degree, S1")
            function_field      = st.text_input("Fungsi Pekerjaan", placeholder="e.g. Engineering, Sales")

    # ── Section 2: Konten ──────────────────────────────────────
    st.markdown("""
    <div class="section-card" style="margin-top:0.5rem;">
        <div class="section-title"><span>2</span> Konten Lowongan</div>
    </div>
    """, unsafe_allow_html=True)

    company_profile = st.text_area("Profil Perusahaan",     height=100, placeholder="Ceritakan tentang perusahaan, visi misi, budaya kerja, dsb...")
    description     = st.text_area("Deskripsi Pekerjaan ✱", height=160, placeholder="Tulis deskripsi lengkap pekerjaan, tanggung jawab, dan ekspektasi...")
    requirements    = st.text_area("Persyaratan & Kualifikasi", height=120, placeholder="Skill, pengalaman, sertifikasi yang dibutuhkan...")
    benefits        = st.text_area("Benefit & Tunjangan",    height=80,  placeholder="Gaji, asuransi, THR, bonus, fasilitas lainnya...")

    # ── Section 3: Atribut ─────────────────────────────────────
    st.markdown("""
    <div class="section-card" style="margin-top:0.5rem;">
        <div class="section-title"><span>3</span> Atribut Tambahan</div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        telecommuting    = st.checkbox("🏠 Posisi Remote / WFH")
    with col4:
        has_company_logo = st.checkbox("🏢 Memiliki Logo Perusahaan")
    with col5:
        has_questions    = st.checkbox("📋 Ada Pertanyaan Skrining")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyze Button ─────────────────────────────────────────
    if st.button("🔍 Analisis Lowongan Sekarang"):
        if not description.strip():
            st.warning("⚠️ Mohon isi minimal kolom **Deskripsi Pekerjaan** untuk memulai analisis.")
        else:
            with st.spinner("Sedang menganalisis lowongan..."):
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

            prob_pct = prob * 100

            if label == 1:
                st.markdown(f"""
                <div class="result-fake">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">Lowongan Terindikasi PALSU</div>
                    <div class="result-sub">Probabilitas penipuan: <strong>{prob_pct:.1f}%</strong></div>
                    <div class="prob-meter"><div class="prob-fill-fake" style="width:{min(prob_pct,100):.1f}%"></div></div>
                    <div style="font-size:0.8rem; color:#9B2C2C; margin-top:0.5rem;">Threshold deteksi: {THRESHOLD*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.error("**Hati-hati!** Model mendeteksi pola yang sering ditemukan pada lowongan tidak asli. Lakukan verifikasi lebih lanjut sebelum memberikan data pribadi atau membayar biaya apapun.")

                # Analisis red flags
                red_flags = []
                if not has_company_logo: red_flags.append("Tidak ada logo perusahaan")
                if not has_questions:    red_flags.append("Tidak ada pertanyaan skrining")
                if not benefits.strip(): red_flags.append("Informasi benefit tidak diisi")
                if not company_profile.strip(): red_flags.append("Tidak ada profil perusahaan")
                desc_lower = description.lower()
                if re.search(r'urgent|immediately|apply now|limited slots|hurry|fast cash', desc_lower):
                    red_flags.append("Deskripsi mengandung kata-kata memaksa/urgent")
                if re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', description):
                    red_flags.append("Deskripsi mengandung alamat email langsung")
                if description.count("!") > 3:
                    red_flags.append(f"Terlalu banyak tanda seru ({description.count('!')} kali)")

                if red_flags:
                    st.markdown("**Faktor yang mempengaruhi hasil:**")
                    for flag in red_flags:
                        st.markdown(f'<div class="flag-box">🚩 {flag}</div>', unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-real">
                    <div class="result-icon">✅</div>
                    <div class="result-title">Lowongan Terindikasi ASLI</div>
                    <div class="result-sub">Probabilitas penipuan: <strong>{prob_pct:.1f}%</strong></div>
                    <div class="prob-meter"><div class="prob-fill-real" style="width:{100-prob_pct:.1f}%"></div></div>
                    <div style="font-size:0.8rem; color:#276749; margin-top:0.5rem;">Threshold deteksi: {THRESHOLD*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.success("Lowongan ini tampak **legitim** berdasarkan analisis model. Tetap lakukan verifikasi mandiri sebelum memberikan data sensitif.")

                safe_signals = []
                if has_company_logo: safe_signals.append("Memiliki logo perusahaan")
                if has_questions:    safe_signals.append("Ada pertanyaan skrining yang terstruktur")
                if benefits.strip(): safe_signals.append("Informasi benefit tersedia")
                if company_profile.strip(): safe_signals.append("Profil perusahaan dicantumkan")

                if safe_signals:
                    st.markdown("**Sinyal positif yang ditemukan:**")
                    for signal in safe_signals:
                        st.markdown(f'<div class="safe-box">✅ {signal}</div>', unsafe_allow_html=True)

            with st.expander("📊 Detail Teknis Analisis"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", "Random Forest Tuned")
                c2.metric("Probabilitas Penipuan", f"{prob_pct:.2f}%")
                c3.metric("Threshold Optimal", f"{THRESHOLD*100:.2f}%")
                st.markdown(f"""
                - Fitur teks diproses menggunakan **TF-IDF Vectorizer** ({len(KOLOM_TEKS)} kolom teks)
                - Fitur numerik: {len(KOLOM_NUMERIK)} fitur (panjang teks, deteksi email, kata urgent, dsb)
                - Hasil ini bersifat **prediktif** — bukan keputusan final
                """)

            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> Hasil analisis ini dihasilkan oleh model machine learning dan tidak menjamin kebenaran 100%. 
                Selalu lakukan verifikasi mandiri melalui situs resmi perusahaan sebelum melamar.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2: PANDUAN
# ══════════════════════════════════════════════════════════════
elif "Panduan" in page:

    st.markdown("""
    <div class="page-header">
        <div class="badge">Dokumentasi</div>
        <h1>Panduan Penggunaan</h1>
        <p>Pelajari cara menggunakan GhostJobs Detector dengan efektif untuk melindungi diri dari lowongan kerja palsu.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Apa itu GhostJobs Detector?</div>
        <p style="color:#4A5568; font-size:0.95rem; line-height:1.7;">
        GhostJobs Detector adalah sistem berbasis <strong>Machine Learning</strong> yang dirancang untuk membantu pencari kerja 
        mengidentifikasi apakah sebuah lowongan kerja berpotensi palsu atau tidak. Sistem ini menganalisis teks dan atribut 
        lowongan menggunakan model <strong>Random Forest</strong> yang telah dilatih pada ribuan data lowongan nyata dan palsu.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Langkah-langkah Penggunaan")

    steps = [
        ("1", "Buka halaman Deteksi", "Klik menu '🔍 Deteksi Lowongan' di sidebar kiri untuk membuka form analisis."),
        ("2", "Isi Informasi Dasar", "Masukkan judul posisi, lokasi, departemen, industri, tipe pekerjaan, level pengalaman, dan pendidikan yang dibutuhkan."),
        ("3", "Salin Konten Lowongan", "Salin dan tempel teks dari lowongan yang ingin diperiksa ke kolom Deskripsi Pekerjaan, Persyaratan, Benefit, dan Profil Perusahaan."),
        ("4", "Centang Atribut", "Tandai apakah lowongan tersebut remote, memiliki logo perusahaan, atau memiliki pertanyaan skrining."),
        ("5", "Klik Analisis", "Tekan tombol 'Analisis Lowongan Sekarang' dan tunggu beberapa detik untuk melihat hasil."),
        ("6", "Baca Hasil dengan Bijak", "Perhatikan probabilitas dan faktor-faktor yang ditemukan. Gunakan sebagai acuan, bukan keputusan final."),
    ]

    for num, title_step, desc_step in steps:
        st.markdown(f"""
        <div class="guide-step" style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:1rem 1.25rem; margin-bottom:0.75rem; display:flex; gap:1rem; align-items:flex-start;">
            <div style="background:#0D1B2A; color:white; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:0.85rem; flex-shrink:0; margin-top:2px;">{num}</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; color:#1A202C; margin-bottom:0.2rem;">{title_step}</div>
                <div style="color:#64748B; font-size:0.875rem; line-height:1.6;">{desc_step}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💡 Tips Mendapatkan Hasil Terbaik")

    tips = [
        ("✏️", "Isi selengkap mungkin", "Semakin banyak informasi yang dimasukkan, semakin akurat analisis model."),
        ("📋", "Salin teks asli", "Salin langsung dari postingan lowongan tanpa mengubah teks agar analisis lebih akurat."),
        ("🔄", "Gunakan sebagai referensi", "Gunakan hasil analisis sebagai salah satu pertimbangan, bukan satu-satunya acuan."),
        ("🌐", "Verifikasi mandiri", "Cek situs resmi perusahaan dan LinkedIn mereka untuk konfirmasi lebih lanjut."),
    ]

    col_a, col_b = st.columns(2)
    for i, (icon, tip_title, tip_desc) in enumerate(tips):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div style="background:white; border:1px solid #E2E8F0; border-radius:10px; padding:1.25rem; margin-bottom:0.75rem;">
                <div style="font-size:1.5rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; color:#1A202C; margin-bottom:0.3rem;">{tip_title}</div>
                <div style="color:#64748B; font-size:0.85rem; line-height:1.5;">{tip_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top:1rem;">
        ⚠️ <strong>Penting:</strong> GhostJobs Detector adalah alat bantu, bukan pengganti penilaian manusia. 
        Jangan pernah memberikan data pribadi, membayar biaya apapun, atau memberikan akses akun sebelum 
        memverifikasi keaslian perusahaan secara mandiri.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3: TENTANG
# ══════════════════════════════════════════════════════════════
elif "Tentang" in page:

    st.markdown("""
    <div class="page-header">
        <div class="badge">Capstone Pijak 2026</div>
        <h1>Tentang Sistem</h1>
        <p>Informasi teknis dan latar belakang pengembangan GhostJobs Detector.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Latar Belakang</div>
            <p style="color:#4A5568; font-size:0.9rem; line-height:1.8;">
            Lowongan kerja palsu merupakan masalah yang semakin marak di era digital. 
            Berdasarkan dataset <strong>fake_job_postings.csv</strong> dari Kaggle, sekitar 4.8% dari total 
            lowongan yang beredar terindikasi palsu — sebuah angka yang kecil namun berdampak besar 
            bagi para pencari kerja yang tidak waspada.
            </p>
            <p style="color:#4A5568; font-size:0.9rem; line-height:1.8;">
            GhostJobs Detector dikembangkan sebagai solusi berbasis ML untuk membantu pengguna 
            mengidentifikasi potensi penipuan sebelum melamar.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
            <div class="section-title">Pipeline Model</div>
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("Text Preprocessing", "Pembersihan HTML, URL, email, angka → tokenisasi → stopword removal → stemming"),
            ("Feature Engineering", "TF-IDF Vectorizer (12 kolom teks) + 11 fitur numerik custom"),
            ("Class Imbalance", "SMOTE (Synthetic Minority Oversampling Technique)"),
            ("Model Training", "Logistic Regression & Random Forest + RandomizedSearchCV tuning"),
            ("Threshold Optimization", f"Optimal threshold: {THRESHOLD:.4f} (berdasarkan F1-Score)"),
        ]

        for step_title, step_desc in pipeline_steps:
            st.markdown(f"""
            <div style="display:flex; gap:0.75rem; padding:0.6rem 0; border-bottom:1px solid #F1F5F9; align-items:flex-start;">
                <div style="width:6px; height:6px; background:#2D9CDB; border-radius:50%; margin-top:7px; flex-shrink:0;"></div>
                <div>
                    <div style="font-weight:600; color:#1A202C; font-size:0.875rem;">{step_title}</div>
                    <div style="color:#64748B; font-size:0.8rem;">{step_desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Spesifikasi Model</div>
        """, unsafe_allow_html=True)

        specs = [
            ("Model Terpilih", "Random Forest Tuned"),
            ("Metode Tuning", "RandomizedSearchCV"),
            ("Fitur Teks", "12 kolom (TF-IDF)"),
            ("Fitur Numerik", "11 fitur custom"),
            ("Penanganan Imbalance", "SMOTE"),
            ("Threshold Optimal", f"{THRESHOLD:.4f}"),
            ("Dataset", "fake_job_postings.csv"),
            ("Sumber Data", "Kaggle"),
        ]

        for label_s, value_s in specs:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #F1F5F9; font-size:0.85rem;">
                <span style="color:#64748B;">{label_s}</span>
                <span style="font-weight:600; color:#1A202C;">{value_s}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card" style="margin-top:0;">
            <div class="section-title">Tim Pengembang</div>
        """, unsafe_allow_html=True)

        members = [
            ("Dea Zasqia P. Malau", "APC322D6X0291", "Streamlit & Integrasi Model"),
            ("Dian Nazira", "APC322D6X0292", "EDA & Data Quality"),
            ("Adinda Muarriva", "APC322D6X0294", "Preprocessing & Feature Eng."),
            ("Khairun Nisa", "APC322D6X0409", "Model Training & Evaluasi"),
        ]

        for name, cohort_id, role in members:
            st.markdown(f"""
            <div style="padding:0.6rem 0; border-bottom:1px solid #F1F5F9;">
                <div style="font-weight:600; color:#1A202C; font-size:0.875rem;">{name}</div>
                <div style="color:#64748B; font-size:0.75rem;">{cohort_id} · {role}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)