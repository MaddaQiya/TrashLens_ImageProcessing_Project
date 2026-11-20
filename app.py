import os
import json
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from image_processor import process_image, FEATURE_NAMES

st.set_page_config(
    page_title="Klasifikasi Sampah Otomatis - PCD",
    page_icon="♻️",
    layout="wide"
)

MODEL_DIR = "models"

CLASS_INFO = {
    "cardboard": {"label": "Cardboard (Kardus)", "category": "Anorganik - Dapat Didaur Ulang", "icon": "📦"},
    "glass": {"label": "Glass (Kaca)", "category": "Anorganik - Dapat Didaur Ulang", "icon": "🍾"},
    "metal": {"label": "Metal (Logam/Kaleng)", "category": "Anorganik - Dapat Didaur Ulang", "icon": "🥫"},
    "paper": {"label": "Paper (Kertas)", "category": "Anorganik - Dapat Didaur Ulang", "icon": "📄"},
    "plastic": {"label": "Plastic (Plastik)", "category": "Anorganik - Dapat Didaur Ulang", "icon": "🥤"},
    "trash": {"label": "Trash (Residu/Lainnya)", "category": "Residu / Non-Daur Ulang", "icon": "🗑️"}
}


@st.cache_resource
def load_artifacts():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

    if not os.path.exists(metrics_path) or not os.path.exists(scaler_path):
        return None, None, None

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_model_file = metrics.get("best_model_file", "best_model.joblib")
    model_path = os.path.join(MODEL_DIR, best_model_file)
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "rf_model.joblib")

    if not os.path.exists(model_path):
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, metrics


def decode_uploaded_image(file_buffer):
    file_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr


def main():
    st.title("♻️ Sistem Klasifikasi Sampah Otomatis Berbasis Pengolahan Citra Digital")
    st.markdown(
        "**Proyek Akhir Mata Kuliah Pengolahan Citra Digital** — Klasifikasi 6 kelas sampah dataset **TrashNet** "
        "(*cardboard, glass, metal, paper, plastic, trash*) menggunakan ekstraksi fitur warna (HSV), tekstur (GLCM), dan bentuk (Hu Moments)."
    )

    model, scaler, metrics = load_artifacts()

    if model is None or scaler is None or metrics is None:
        st.warning(
            "⚠️ Model belum ditemukan di folder `models/`. Silakan jalankan `python train.py` terlebih dahulu."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Tab 1: Uji & Visualisasi Citra",
        "📊 Tab 2: Evaluasi Model",
        "📖 Tab 3: Alur Metode & Pustaka"
    ])

    # =========================================================================
    # TAB 1: UJI & VISUALISASI CITRA
    # =========================================================================
    with tab1:
        st.subheader("Uji Klasifikasi & Visualisasi Tahapan Pengolahan Citra")

        input_mode = st.radio(
            "Pilih Sumber Citra:",
            ["📁 Upload File Gambar", "📷 Kamera Langsung (Webcam)"],
            horizontal=True
        )

        img_buffer = None
        if input_mode == "📁 Upload File Gambar":
            img_buffer = st.file_uploader(
                "Unggah citra sampah (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"]
            )
        else:
            img_buffer = st.camera_input("Ambil foto sampah menggunakan kamera")

        if img_buffer is not None:
            image_bgr = decode_uploaded_image(img_buffer)
            if image_bgr is None:
                st.error("Gagal membaca file citra. Pastikan format gambar valid.")
            else:
                # Hitung waktu pemrosesan (prapemrosesan + ekstraksi fitur + prediksi)
                start_time = time.perf_counter()
                features, stages = process_image(image_bgr)
                features_scaled = scaler.transform([features])
                pred_class = model.predict(features_scaled)[0]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features_scaled)[0]
                    classes = list(model.classes_)
                else:
                    classes = metrics.get("classes", list(CLASS_INFO.keys()))
                    probs = [1.0 if c == pred_class else 0.0 for c in classes]
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                max_prob = float(np.max(probs)) * 100.0
                c_info = CLASS_INFO.get(
                    pred_class,
                    {"label": pred_class.capitalize(), "category": "-", "icon": "♻️"}
                )

                st.markdown("---")
                # Ringkasan Hasil Prediksi
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric(
                        label="Hasil Prediksi Kelas Sampah",
                        value=f"{c_info['icon']} {c_info['label']}"
                    )
                    st.caption(f"Kategori: **{c_info['category']}**")
                with res_col2:
                    st.metric(
                        label="Tingkat Keyakinan (Probabilitas)",
                        value=f"{max_prob:.2f}%"
                    )
                with res_col3:
                    st.metric(
                        label="Waktu Pemrosesan",
                        value=f"{elapsed_ms:.2f} ms"
                    )

                # Visualisasi 4 Tahap Pengolahan Citra Secara Berdampingan
                st.markdown("### 🖼️ Visualisasi 4 Tahap Pengolahan Citra Digital")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.image(stages["original"], caption="1. Citra Asli (Resized 256×256)", use_container_width=True)
                with col2:
                    st.image(stages["clahe"], caption="2. Hasil Gaussian Blur + CLAHE", use_container_width=True)
                with col3:
                    st.image(stages["gray"], caption="3. Citra Grayscale", use_container_width=True, clamp=True)
                with col4:
                    st.image(stages["mask"], caption="4. Mask Segmentasi (Otsu + Closing)", use_container_width=True, clamp=True)

                st.markdown("---")
                detail_col1, detail_col2 = st.columns([1.2, 1])

                with detail_col1:
                    st.markdown("### 📋 Tabel Nilai Ekstraksi Fitur (18 Fitur)")
                    hsv_df = pd.DataFrame({
                        "Fitur Warna (HSV)": FEATURE_NAMES[0:6],
                        "Nilai": [round(float(v), 4) for v in features[0:6]]
                    })
                    glcm_df = pd.DataFrame({
                        "Fitur Tekstur (GLCM)": FEATURE_NAMES[6:11],
                        "Nilai": [round(float(v), 4) for v in features[6:11]]
                    })
                    hu_df = pd.DataFrame({
                        "Fitur Bentuk (Hu Moments)": FEATURE_NAMES[11:18],
                        "Nilai": [round(float(v), 4) for v in features[11:18]]
                    })

                    f_tab1, f_tab2, f_tab3, f_tab_all = st.tabs([
                        "🎨 Warna (HSV - 6)",
                        "🧱 Tekstur (GLCM - 5)",
                        "📐 Bentuk (Hu Moments - 7)",
                        "📑 Semua Fitur (18)"
                    ])
                    with f_tab1:
                        st.dataframe(hsv_df, use_container_width=True, hide_index=True)
                    with f_tab2:
                        st.dataframe(glcm_df, use_container_width=True, hide_index=True)
                    with f_tab3:
                        st.dataframe(hu_df, use_container_width=True, hide_index=True)
                    with f_tab_all:
                        all_features_df = pd.DataFrame({
                            "No": list(range(1, 19)),
                            "Kelompok": (["Warna (HSV)"] * 6) + (["Tekstur (GLCM)"] * 5) + (["Bentuk (Hu Moments)"] * 7),
                            "Nama Fitur": FEATURE_NAMES,
                            "Nilai Asli": [round(float(v), 6) for v in features],
                            "Nilai Terstandarisasi (Z-Score)": [round(float(v), 4) for v in features_scaled[0]]
                        })
                        st.dataframe(all_features_df, use_container_width=True, hide_index=True)

                with detail_col2:
                    st.markdown("### 📈 Distribusi Probabilitas per Kelas")
                    prob_df = pd.DataFrame({
                        "Kelas": [CLASS_INFO.get(c, {"label": c})["label"] for c in classes],
                        "Probabilitas (%)": [round(float(p) * 100.0, 2) for p in probs]
                    }).sort_values(by="Probabilitas (%)", ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    st.bar_chart(prob_df.set_index("Kelas")["Probabilitas (%)"])

        else:
            st.info("👆 Silakan unggah gambar sampah atau gunakan kamera untuk melihat proses tahapan pengolahan citra dan hasil prediksi.")
            sample_pipe_path = os.path.join(MODEL_DIR, "sample_pipeline.png")
            if os.path.exists(sample_pipe_path):
                with st.expander("Lihat Contoh Hasil Pipeline 4 Tahap pada Sampel Dataset TrashNet"):
                    st.image(sample_pipe_path, caption="Contoh Visualisasi 4 Tahap Pengolahan Citra Digital", use_container_width=True)

    # =========================================================================
    # TAB 2: EVALUASI MODEL
    # =========================================================================
    with tab2:
        st.subheader("Evaluasi Kinerja Model & Perbandingan Algoritma")

        best_name = metrics.get("best_model", "Random Forest")
        acc = metrics.get("accuracy", 0.0) * 100.0
        prec_w = metrics.get("precision_weighted", metrics.get("report", {}).get("weighted avg", {}).get("precision", 0.0)) * 100.0
        rec_w = metrics.get("recall_weighted", metrics.get("report", {}).get("weighted avg", {}).get("recall", 0.0)) * 100.0
        f1_w = metrics.get("f1_weighted", metrics.get("report", {}).get("weighted avg", {}).get("f1-score", 0.0)) * 100.0

        st.markdown(f"**Model Terbaik yang Digunakan:** `{best_name}` (Total Dataset: **{metrics.get('dataset_total', 2527)}** citra | Train: **{metrics.get('train_size', 2021)}** | Test: **{metrics.get('test_size', 506)}**)")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{acc:.2f}%")
        m_col2.metric("Precision (Weighted)", f"{prec_w:.2f}%")
        m_col3.metric("Recall (Weighted)", f"{rec_w:.2f}%")
        m_col4.metric("F1-Score (Weighted)", f"{f1_w:.2f}%")

        st.markdown("---")
        st.markdown("### ⚖️ Tabel Perbandingan Model: Random Forest vs SVM (RBF Kernel)")
        comp = metrics.get("model_comparison", {})
        if comp:
            comp_rows = []
            for m_name, m_data in comp.items():
                comp_rows.append({
                    "Model Klasifikasi": m_name,
                    "Accuracy (%)": round(m_data.get("accuracy", 0.0) * 100, 2),
                    "Precision Macro (%)": round(m_data.get("precision_macro", 0.0) * 100, 2),
                    "Recall Macro (%)": round(m_data.get("recall_macro", 0.0) * 100, 2),
                    "F1-Score Macro (%)": round(m_data.get("f1_macro", 0.0) * 100, 2),
                    "Precision Weighted (%)": round(m_data.get("precision_weighted", 0.0) * 100, 2),
                    "Recall Weighted (%)": round(m_data.get("recall_weighted", 0.0) * 100, 2),
                    "F1-Score Weighted (%)": round(m_data.get("f1_weighted", 0.0) * 100, 2),
                    "Status": "✅ Model Terbaik" if m_name == best_name else "Pembanding"
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        eval_col1, eval_col2 = st.columns([1.15, 1])

        with eval_col1:
            st.markdown(f"### 📋 Metrik Evaluasi per Kelas ({best_name})")
            per_class = metrics.get("per_class", {})
            if not per_class and "report" in metrics:
                for c in metrics.get("classes", []):
                    if c in metrics["report"]:
                        per_class[c] = metrics["report"][c]

            class_rows = []
            for cls_name, cls_m in per_class.items():
                class_rows.append({
                    "Kelas Sampah": CLASS_INFO.get(cls_name, {"label": cls_name})["label"],
                    "Precision (%)": round(cls_m.get("precision", 0.0) * 100, 2),
                    "Recall (%)": round(cls_m.get("recall", 0.0) * 100, 2),
                    "F1-Score (%)": round(cls_m.get("f1-score", 0.0) * 100, 2),
                    "Jumlah Sampel Uji (Support)": int(cls_m.get("support", 0))
                })
            class_df = pd.DataFrame(class_rows)
            st.dataframe(class_df, use_container_width=True, hide_index=True)

        with eval_col2:
            st.markdown("### 🟦 Confusion Matrix")
            cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"Confusion Matrix ({best_name})", use_container_width=True)
            else:
                st.warning("Gambar `confusion_matrix.png` belum tersedia.")

    # =========================================================================
    # TAB 3: ALUR METODE & PUSTAKA
    # =========================================================================
    with tab3:
        st.subheader("Alur Metode Pengolahan Citra Digital & Pustaka Python")

        st.markdown("""
        ### 1. Ringkasan Alur Metode Pengolahan Citra Digital (Pipeline)
        Sistem ini dirancang dengan pendekatan **Pengolahan Citra Digital Klasik (Non-Deep Learning / White-Box)** yang terdiri dari 4 tahapan utama:

        1. **Prapemrosesan Citra (*Image Preprocessing*)**:
           - **Resize (`256×256` piksel)**: Menyeragamkan dimensi spasial seluruh citra input agar komputasi konsisten.
           - **Gaussian Blur (`5×5`)**: Mereduksi derau frekuensi tinggi (*high-frequency noise*) sebelum ekstraksi kontras dan segmentasi.
           - **CLAHE (*Contrast Limited Adaptive Histogram Equalization*)**: Diterapkan pada kanal luminansi **L** di ruang warna **CIE-LAB** (`clipLimit=2.0`, `tileGridSize=(8,8)`) untuk meningkatkan kontras lokal tanpa memperkuat derau secara berlebihan.

        2. **Segmentasi Objek (*Image Segmentation*)**:
           - **Konversi Grayscale**: Mengubah citra hasil peningkatan kontras BGR menjadi citra derajat keabuan 1 kanal.
           - **Otsu Thresholding (`THRESH_BINARY_INV + THRESH_OTSU`)**: Menentukan nilai ambang (*threshold*) secara otomatis berdasarkan histogram bimodal untuk memisahkan objek sampah dari latar belakang terang.
           - **Morfologi Closing (`5×5` Rectangular Kernel)**: Menutup celah atau lubang hitam kecil di dalam objek hasil binerisasi sehingga terbentuk *binary mask* yang utuh.

        3. **Ekstraksi Fitur (*Feature Extraction* — Total 18 Fitur Numerik)**:
           - **Ciri Warna — HSV Color Moments (6 Fitur)**: Menghitung nilai *Mean* (rata-rata) dan *Standard Deviation* (simpangan baku) pada masing-masing kanal **Hue (H)**, **Saturation (S)**, dan **Value (V)**.
           - **Ciri Tekstur — GLCM / *Gray-Level Co-occurrence Matrix* (5 Fitur)**: Menghitung matriks ko-okurensi pada jarak $d=1$ dan 4 arah sudut ($0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ$), lalu merata-ratakan 5 properti tekstur Haralick: *Contrast*, *Dissimilarity*, *Homogeneity*, *Energy*, dan *Correlation*.
           - **Ciri Bentuk — Hu Moments (7 Fitur)**: Menghitung 7 momen invarian Hu (*translation, scale, and rotation invariant*) dari *binary mask* hasil segmentasi, kemudian ditransformasikan ke skala logaritmik:
             $$H'_i = -\\text{sign}(h_i) \\cdot \\log_{10}(|h_i| + 10^{-10})$$

        4. **Normalisasi & Klasifikasi (*Machine Learning Classification*)**:
           - **Pembagian Data (*Stratified Split*)**: Dataset TrashNet (2.527 citra, 6 kelas) dibagi menjadi **80% Data Latih** dan **20% Data Uji** secara *stratified* (`random_state=42`).
           - **StandardScaler**: Menstandarisasi ke-18 fitur menjadi distribusi dengan $\\mu=0$ dan $\\sigma=1$ (*fit* hanya dilakukan pada data latih untuk mencegah *data leakage*).
           - **Pemodelan & Pemilihan Model Terbaik**: Membandingkan algoritma **Random Forest** (`n_estimators=200`, `class_weight='balanced'`) dan **Support Vector Machine / SVM** (`kernel='rbf'`, `class_weight='balanced'`). Model dengan akurasi uji tertinggi disimpan sebagai model utama.
        """)

        st.markdown("---")
        st.markdown("### 2. Daftar Pustaka Python yang Digunakan")
        lib_df = pd.DataFrame([
            {"Pustaka": "opencv-python-headless (cv2)", "Fungsi Utama dalam Proyek": "Pembacaan citra, Resize, Gaussian Blur, konversi ruang warna (BGR/LAB/HSV/Gray), CLAHE, Otsu Thresholding, Morfologi Closing, dan ekstraksi Hu Moments."},
            {"Pustaka": "scikit-image (skimage)", "Fungsi Utama dalam Proyek": "Ekstraksi fitur tekstur Gray-Level Co-occurrence Matrix (GLCM) menggunakan modul graycomatrix dan graycoprops."},
            {"Pustaka": "scikit-learn (sklearn)", "Fungsi Utama dalam Proyek": "Stratified train-test split, normalisasi fitur (StandardScaler), pelatihan model Random Forest & SVM (RBF), serta perhitungan metrik evaluasi & Confusion Matrix."},
            {"Pustaka": "numpy", "Fungsi Utama dalam Proyek": "Operasi matriks/array multidimensi citra, perhitungan statistik momen warna (Mean & Std Dev), dan transformasi logaritmik Hu Moments."},
            {"Pustaka": "pandas", "Fungsi Utama dalam Proyek": "Penyajian tabel nilai ekstraksi fitur, distribusi probabilitas, dan tabel perbandingan kinerja model."},
            {"Pustaka": "matplotlib & seaborn", "Fungsi Utama dalam Proyek": "Pembuatan visualisasi Confusion Matrix dan dokumentasi citra sebelum-sesudah 4 tahap (sample_pipeline.png)."},
            {"Pustaka": "joblib", "Fungsi Utama dalam Proyek": "Serialisasi dan penyimpanan model klasifikasi terbaik (.joblib) beserta objek StandardScaler."},
            {"Pustaka": "streamlit", "Fungsi Utama dalam Proyek": "Pembuatan antarmuka aplikasi web interaktif 3 Tab untuk pengujian citra, kamera langsung, dan pelaporan evaluasi model."}
        ])
        st.dataframe(lib_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
