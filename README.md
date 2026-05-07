# 🧠 Klasifikasi MRI Tumor Otak dengan Pendekatan Explainable AI (Score-CAM)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.13.2-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Repositori ini berisi kode sumber lengkap untuk proyek Tugas Akhir/Skripsi yang berfokus pada analisis citra medis (Medical Image Analysis). Proyek ini membandingkan kinerja dua arsitektur Deep Learning (*DenseNet201* dan *ResNet-50*) dalam mengklasifikasikan citra MRI tumor otak ke dalam 4 kelas: **Glioma, Meningioma, Pituitary, dan No Tumor**.

Nilai kebaruan (*novelty*) dari penelitian ini terletak pada implementasi mandiri (*custom build*) dari **Score-CAM (Score-Weighted Class Activation Mapping)** menggunakan optimasi *Raw Tensor Forward Pass*, serta analisis komputasi inferensi lintas perangkat (Cloud Server vs Edge Device/Apple M3).

---

## ✨ Fitur Utama Penelitian

1. **Arsitektur Fine-Tuning Teroptimasi:** Modifikasi dan *fine-tuning* pada DenseNet201 (Akurasi Uji: **96.60%**) dan ResNet-50 dengan teknik *Weight Decay* terkontrol dan *Global Average Pooling*.
2. **Custom Score-CAM (From Scratch):** Algoritma Explainable AI (XAI) dibangun murni dari operasi matematika dasar TensorFlow tanpa bergantung pada pustaka pihak ketiga (*third-party libraries*) seperti `tf-keras-vis`. Dioptimasi dengan *Iterative Chunking* dan interpolasi `Lanczos4` untuk mencegah *Out-Of-Memory* (OOM) dan menjaga presisi area tumor.
3. **Hardware Benchmarking:** Analisis empiris kecepatan inferensi dan visualisasi XAI antara lingkungan *Entry-Level Cloud* (1-Core Intel Xeon Google Colab) melawan *Edge Device Consumer* (Apple Silicon M3 CPU & GPU).
4. **Cross-Platform Workflow:** Pemisahan *pipeline* secara modular antara Kaggle (Training) dan Google Colab (Evaluation) untuk efisiensi *resource cloud*.
5. **Web Dashboard Real-Time:** Aplikasi purwarupa medis berbasis antarmuka web (Streamlit) untuk demonstrasi prediksi dan ekstraksi *heatmap* Score-CAM secara *real-time*.

---

## 🛠️ Lingkungan Pengembangan & Spesifikasi (Tech Stack)

Eksperimen ini direproduksi secara ketat menggunakan lingkungan perangkat lunak berikut:

| Kategori | Library / Framework | Versi Target | Fungsi Utama |
| :--- | :--- | :--- | :--- |
| **Framework Utama** | `tensorflow` | 2.19.0 | Mesin komputasi Deep Learning (Backend) |
| **Arsitektur** | `keras` | 3.13.2 | API perancangan lapisan model |
| **Numerik & Data** | `numpy`, `pandas` | 2.0.2, 2.2.2 | Manipulasi matriks dan pengelolaan dataset |
| **Visualisasi** | `matplotlib`, `seaborn`| 3.10.0, 0.13.2| Render grafik, Heatmap, dan Confusion Matrix |
| **Evaluasi Metrik** | `scikit-learn` | 1.6.1 | Perhitungan *Precision, Recall, F1-Score* |
| **Pemrosesan Citra** | `pillow`, `opencv-python`| 11.3.0, 4.x | Resize (Lanczos4) dan Color Mapping (Jet) |
| **Deployment** | `streamlit` | 1.55.0 | Pembuatan purwarupa aplikasi web medis |

---

## 📂 Struktur Repositori

```text
📦 TugasAkhir
 ┣ 📂 datasets/                  # Direktori data MRI (Tidak diunggah karena ukuran/lisensi)
 ┣ 📂 local_models/              # Tempat penyimpanan file model final (.keras)
 ┣ 📂 Visualizations/            # Output otomatis grafik, Score-CAM, dan Matrix
 ┣ 📜 01_Preprocessing_Eksplorasi.ipynb   # Notebook: Analisis Data dan Praproses
 ┣ 📜 02_Training_FineTuning.ipynb        # Notebook: Pelatihan Kaggle (Defensive Logging)
 ┣ 📜 03_Analisis_Evaluasi.ipynb          # Notebook: Colab Evaluation (Universal Hybrid Code)
 ┣ 📜 app.py                     # Source code utama untuk Streamlit Web Dashboard
 ┣ 📜 requirements.txt           # Daftar dependensi library
 ┗ 📜 README.md                  # Dokumentasi repositori
```

---

## 🚀 Panduan Instalasi dan Penggunaan Lokal

Untuk menjalankan antarmuka web Streamlit (*app.py*) di perangkat lokal (laptop/PC) Anda, ikuti langkah-langkah berikut:

**1. Kloning Repositori**
```bash
git clone [https://github.com/](https://github.com/)[USERNAME_GITHUB_ANDA]/[NAMA_REPO].git
cd [NAMA_REPO]
```

**2. Siapkan Virtual Environment (Direkomendasikan)**
```bash
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# ATAU
venv\Scripts\activate     # Untuk Windows
```

**3. Instalasi Dependensi**
```bash
pip install -r requirements.txt
```

**4. Persiapan Model**
Pastikan Anda telah meletakkan file model hasil pelatihan (`Final_DenseNet201.keras` dan `Final_ResNet50.keras`) ke dalam *root* direktori atau folder `local_models/`.

**5. Jalankan Aplikasi Web**
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di *browser* Anda pada alamat `http://localhost:8501`.

---

## 📊 Alur Reproduksi Eksperimen (Cloud)

Proyek ini dirancang secara modular. Jika Anda ingin melatih ulang dari awal:
1. **Eksekusi di Kaggle:** Jalankan `02_Training_FineTuning.ipynb` menggunakan GPU T4 x2 di Kaggle Notebook. Skrip ini sudah dilengkapi dengan sistem *Defensive Logging* (CSVLogger & output gambar) untuk mencegah kehilangan data akibat *Notebook Failed/Timeout*.
2. **Evaluasi di Colab:** Unduh hasil bobot model, lalu muat ke dalam `03_Analisis_Evaluasi.ipynb` di Google Colab. Sel ini sudah menggunakan logika *Universal Path* yang adaptif terhadap lingkungan eksekusinya.

---

## ✒️ Sitasi dan Hak Cipta

Kode dan arsitektur dalam repositori ini dikembangkan murni untuk keperluan riset akademis penyusunan Tugas Akhir/Skripsi. Seluruh visualisasi Explainable AI tidak dimaksudkan sebagai pengganti diagnosis medis profesional dari dokter spesialis radiologi/onkologi.

**Peneliti Utama:** M. Ihsan Rizqullah Adfa

**Institusi:** Prodi S1 Informatika, Departemen Informatika, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Syiah Kuala.

**Tahun:** 2026
```
