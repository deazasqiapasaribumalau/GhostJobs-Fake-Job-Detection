# 👻 GhostJobs - Fake Job Detector

> **Deteksi lowongan kerja palsu berbasis AI** mengklasifikasikan job posting sebagai asli atau scam menggunakan machine learning, dilengkapi penjelasan hasil prediksi yang mudah dipahami.

🔗 **Live App:** [ghostjobs-fake-job-detection.streamlit.app](https://ghostjobs-fake-job-detection.streamlit.app/) 
📁 **Dataset:** [Kaggle — Real or Fake? Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) 
🏫 **Program:** Capstone Project Pijak × IBM SkillsBuild 2026 · Tim PJK-GM088 

---

## 📖 Tentang Proyek

Pertumbuhan platform rekrutmen digital di Indonesia tidak diimbangi dengan mekanisme verifikasi lowongan yang memadai. Jutaan lowongan palsu beredar setiap tahunnya, dengan korban terbanyak adalah fresh graduate dan individu dalam kondisi mendesak.

**GhostJobs** hadir sebagai solusi — sistem deteksi otomatis yang menganalisis teks deskripsi pekerjaan, profil perusahaan, persyaratan kandidat, dan pola bahasa untuk mendeteksi indikasi penipuan, lalu memberikan hasil klasifikasi beserta penjelasannya kepada pengguna.

---

## ✨ Fitur Utama

- 🔍 **Deteksi Otomatis** — Klasifikasi lowongan sebagai *asli* atau *palsu* dalam hitungan detik
- 📝 **Mode Input Manual** — Tempel deskripsi lowongan sendiri untuk dianalisis
- 🎯 **Mode Demo** — Pilih contoh lowongan langsung dari dataset Kaggle, lengkap dengan label asli untuk perbandingan
- 💡 **Penjelasan Prediksi** — Tampilkan indikator mencurigakan yang ditemukan model
- 📊 **Detail Teknis** — Tampilkan probabilitas, threshold, dan fitur yang digunakan

---

## 🗂️ Struktur Folder

```
GhostJobs/
├── app.py                              # Aplikasi utama Streamlit
├── models/
│   ├── config.json                     # Konfigurasi threshold & nama kolom
│   ├── random_forest_tuned_model.pkl   # Model Random Forest terlatih
│   ├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
│   ├── scaler.pkl                      # Scaler fitur numerik
│   └── sample_demo.csv                 # Dataset sampel untuk Mode Demo
├── requirements.txt                    # Daftar dependencies
└── README.md
```

---

## ⚙️ Instalasi & Menjalankan Lokal

### Prasyarat
- Python 3.8+
- pip

### Langkah-langkah

**1. Clone repository**
```bash
git clone https://github.com/deazasqiapasaribumalau/GhostJobs-Fake-Job-Detection.git
cd GhostJobs-Fake-Job-Detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
streamlit run app.py
```

**4. Buka browser**
```
http://localhost:8501
```

> **Catatan:** Pastikan folder `models/` beserta semua file `.pkl`, `config.json`, dan `sample_demo.csv` tersedia sebelum menjalankan aplikasi.

---

## 📦 Dependencies

```
streamlit
scikit-learn
joblib
pandas
numpy
scipy
nltk
```

Install sekaligus:
```bash
pip install streamlit scikit-learn joblib pandas numpy scipy nltk
```

---

## 🔧 Pipeline Model

```
Raw Job Posting
      │
      ▼
┌─────────────────────────────┐
│  01. Text Preprocessing     │  lowercase → strip HTML/URL/email → tokenize
│                             │  → remove stopwords → stemming (PorterStemmer)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  02. Feature Engineering    │  TF-IDF (12 kolom teks)
│                             │  + 11 fitur numerik custom
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  03. Class Imbalance        │  SMOTE — Synthetic Minority Oversampling
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  04. Model Training         │  Logistic Regression vs Random Forest
│                             │  + RandomizedSearchCV hyperparameter tuning
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  05. Threshold Optimization │  Optimal threshold berdasarkan F1-Score & Recall
│                             │  (false negative lebih berbahaya dari false positive)
└────────────┬────────────────┘
             │
             ▼
      Prediksi Final
   ✅ Asli  /  ❌ Palsu
```

### Fitur Numerik Custom (11 fitur)

| Fitur | Deskripsi |
|-------|-----------|
| `telecommuting` | Apakah lowongan remote |
| `has_company_logo` | Ada/tidaknya logo perusahaan |
| `has_questions` | Ada/tidaknya pertanyaan skrining |
| `len_description` | Panjang teks deskripsi |
| `len_requirements` | Panjang teks persyaratan |
| `len_benefits` | Panjang teks benefit |
| `len_company_profile` | Panjang profil perusahaan |
| `has_salary` | Ada/tidaknya informasi gaji |
| `has_email_in_desc` | Ada/tidaknya email di deskripsi |
| `exc_mark_count` | Jumlah tanda seru (`!`) dalam deskripsi |
| `has_urgency_words` | Ada kata seperti *"URGENT"*, *"Apply Now"*, *"Fast Cash"* |

---

## 🖥️ Cara Menggunakan Aplikasi

### Mode Input Manual
1. Buka halaman **Deteksi Lowongan** di sidebar
2. Tempel teks deskripsi dan persyaratan lowongan yang ingin diperiksa
3. Centang atribut yang sesuai (remote, ada logo, ada pertanyaan skrining)
4. Isi informasi opsional jika tersedia (judul, industri, lokasi, gaji)
5. Klik tombol **Analisis**
6. Baca hasil prediksi dan indikator yang ditemukan

### Mode Demo
1. Aktifkan toggle **Mode Demo** di sidebar
2. Pilih contoh lowongan dari dropdown dataset Kaggle
3. Form akan terisi otomatis
4. Bandingkan prediksi model dengan label asli dari dataset

---

## 📊 Dataset

| Atribut | Detail |
|---------|--------|
| Sumber | Kaggle — *shivamb/real-or-fake-fake-jobposting-prediction* |
| Jumlah data | 17.880 job posting |
| Distribusi | ±95,2% asli · ±4,8% palsu |
| Fitur utama | title, location, company_profile, description, requirements, benefits, industry, dll. |

---

## 📈 Performa Model

Model terbaik yang digunakan adalah **Random Forest (Tuned)**, dipilih karena secara konsisten unggul di tiga metrik evaluasi dibanding Logistic Regression dan Random Forest Base.

| Metrik | Logistic Regression | Random Forest Base | Random Forest Tuned |
|--------|--------------------|--------------------|---------------------|
| F1-Score | ~0.79 | ~0.82 | **~0.85** |
| Recall | ~0.91 | ~0.72 | **~0.74** |
| AUC-ROC | ~0.97 | ~0.97 | **~0.98** |

> Threshold dioptimasi untuk meminimalkan *false negative* (lowongan palsu yang lolos terdeteksi sebagai asli), karena risikonya lebih besar dibanding *false positive*.

---

## 👥 Tim Pengembang

| Nama | ID | Peran |
|------|----|-------|
| Dea Zasqia P. Malau | APC322D6X0291 | Streamlit & Integrasi |
| Dian Nazira | APC322D6X0292 | EDA & Data Quality |
| Adinda Muarriva | APC322D6X0294 | Preprocessing & Feature Engineering |
| Khairun Nisa | APC322D6X0409 | Model Training & Evaluasi |

---

## ⚠️ Disclaimer

Hasil prediksi GhostJobs dihasilkan oleh model machine learning dan **tidak menjamin kebenaran 100%**. Gunakan hasil sebagai acuan awal, bukan keputusan final. Selalu verifikasi secara mandiri melalui situs resmi perusahaan sebelum melamar atau memberikan data pribadi apapun.

---

<div align="center">
  <sub>👻 GhostJobs · PJK-GM088 · Pijak × IBM SkillsBuild 2026</sub>
</div>
