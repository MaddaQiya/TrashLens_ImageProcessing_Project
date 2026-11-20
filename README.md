# ♻️ Sistem Klasifikasi Sampah Otomatis Berbasis Pengolahan Citra Digital

Proyek Akhir Mata Kuliah **Pengolahan Citra Digital (PCD)** — Klasifikasi Sampah Otomatis menggunakan dataset **TrashNet** (6 kelas: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) dengan pendekatan pengolahan citra klasik (*white-box pipeline*: Prapemrosesan CLAHE, Segmentasi Otsu + Morfologi Closing, Ekstraksi 18 Fitur HSV + GLCM + Hu Moments, serta Klasifikasi Random Forest & SVM).

---

## 📁 Struktur Folder Proyek

```text
klasifikasi-sampah-pcd/
├── dataset-resized/            # Dataset TrashNet (6 folder kelas: cardboard, glass, metal, paper, plastic, trash)
│   ├── cardboard/              # 403 citra
│   ├── glass/                  # 501 citra
│   ├── metal/                  # 410 citra
│   ├── paper/                  # 594 citra
│   ├── plastic/                # 482 citra
│   └── trash/                  # 137 citra
├── models/                     # Artefak hasil pelatihan & evaluasi model
│   ├── best_model.joblib       # Model klasifikasi terbaik (akurasi tertinggi)
│   ├── rf_model.joblib         # Model Random Forest (n_estimators=200, class_weight='balanced')
│   ├── svm_model.joblib        # Model SVM (RBF kernel, class_weight='balanced')
│   ├── scaler.joblib           # StandardScaler yang di-fit pada data latih
│   ├── metrics.json            # Metrik evaluasi lengkap per kelas & perbandingan RF vs SVM
│   ├── confusion_matrix.png    # Visualisasi Confusion Matrix data uji
│   └── sample_pipeline.png     # Contoh visualisasi citra 4 tahap (Asli, CLAHE, Grayscale, Binary Mask)
├── image_processor.py          # Modul pipeline prapemrosesan, segmentasi, dan ekstraksi 18 fitur
├── train.py                    # Skrip pelatihan, evaluasi, perbandingan RF vs SVM, dan penyimpanan model
├── app.py                      # Aplikasi web interaktif Streamlit (3 Tab)
├── requirements.txt            # Daftar dependensi pustaka Python
└── README.md                   # Dokumentasi proyek
```

---

## 🔬 Metodologi & Pipeline Pengolahan Citra

1. **Prapemrosesan Citra**:
   - **Resize**: Penyeragaman ukuran citra menjadi `256 x 256` piksel.
   - **Gaussian Blur (`5x5`)**: Peredaman derau (*noise filtering*).
   - **CLAHE (*Contrast Limited Adaptive Histogram Equalization*)**: Peningkatan kontras lokal pada kanal luminansi **L** di ruang warna **LAB** (`clipLimit=2.0`, `tileGridSize=(8,8)`).
2. **Segmentasi Citra**:
   - Konversi citra hasil CLAHE ke **Grayscale**.
   - **Otsu Thresholding** (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) untuk memisahkan objek dari latar belakang secara otomatis.
   - **Operasi Morfologi Closing** (kernel `5x5`) untuk menutup lubang kecil pada *binary mask*.
3. **Ekstraksi Fitur (Total 18 Fitur)**:
   - **Ciri Warna (6 Fitur)**: *Mean* dan *Standard Deviation* pada masing-masing kanal ruang warna **HSV** (H, S, V).
   - **Ciri Tekstur (5 Fitur)**: Rata-rata 5 fitur **GLCM** (*Contrast, Dissimilarity, Homogeneity, Energy, Correlation*) pada jarak $d=1$ dan 4 sudut ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
   - **Ciri Bentuk (7 Fitur)**: **7 Hu Moments** dari *binary mask* dengan transformasi skala logaritmik $-\text{sign}(h) \cdot \log_{10}(|h| + 10^{-10})$.
4. **Klasifikasi & Evaluasi**:
   - Pembagian dataset **80% Train** dan **20% Test** secara *stratified* (`random_state=42`).
   - Normalisasi fitur menggunakan `StandardScaler` (*fit* hanya pada data latih untuk mencegah *data leakage*).
   - Perbandingan algoritma **Random Forest** (`n_estimators=200`, `class_weight='balanced'`) dan **SVM** (`kernel='rbf'`, `class_weight='balanced'`).

---

## ⚙️ Cara Instalasi

1. Pastikan **Python 3.10+** (direkomendasikan Python 3.13) telah terpasang.
2. Download dataset dan atur sesuai struktur Folder (https://github.com/garythung/trashnet) data -> dataset-resized
3. Buka terminal / PowerShell pada direktori proyek ini, lalu instal seluruh dependensi:

```powershell
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Melatih Model & Mengekstraksi Fitur (`train.py`)
Jalankan skrip pelatihan untuk memproses seluruh citra di `dataset-resized/`, melatih Random Forest dan SVM, serta menyimpan hasil evaluasi ke folder `models/`:

```powershell
python train.py
```

> **Catatan**: Skrip `train.py` sudah dilengkapi deteksi otomatis struktur direktori dataset, baik yang berada langsung di `dataset-resized/<kelas>` maupun yang terekstrak ganda di `dataset-resized/dataset-resized/<kelas>`.

### 2. Menjalankan Aplikasi Web Interaktif (`app.py`)
Setelah proses pelatihan selesai, jalankan antarmuka Streamlit:

```powershell
streamlit run app.py
```

Aplikasi akan terbuka di peramban (*browser*) pada alamat `http://localhost:8501` dan menyediakan **3 Tab**:
- **Tab 1 (Uji & Visualisasi Citra)**: Unggah gambar atau ambil foto via kamera (`st.camera_input`), lihat 4 tahapan pemrosesan citra secara berdampingan, tabel 18 fitur (HSV, GLCM, Hu Moments), hasil prediksi kelas sampah, probabilitas keyakinan, serta waktu komputasi dalam milidetik (`ms`).
- **Tab 2 (Evaluasi Model)**: Lihat metrik Accuracy, Precision, Recall, F1-Score, tabel perbandingan Random Forest vs SVM, metrik per kelas, serta grafik Confusion Matrix.
- **Tab 3 (Alur Metode & Pustaka)**: Penjelasan lengkap tahapan pengolahan citra digital dan daftar pustaka Python yang digunakan.
