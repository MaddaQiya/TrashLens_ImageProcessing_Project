import os
import json
import time
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from image_processor import process_image, FEATURE_NAMES

st.set_page_config(
    page_title="CLEANERS // WASTE_SCANNER v1.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_DIR = "models"

CLASS_INFO = {
    "cardboard": {
        "code": "CLS-01 // CARDBOard",
        "label": "CARDBOARD [KARDUS]",
        "category": "RECYCLABLE // ANORGANIK DAUR ULANG",
        "recyclable": True
    },
    "glass": {
        "code": "CLS-02 // GLASS",
        "label": "GLASS [KACA]",
        "category": "RECYCLABLE // ANORGANIK DAUR ULANG",
        "recyclable": True
    },
    "metal": {
        "code": "CLS-03 // METAL",
        "label": "METAL [LOGAM/KALENG]",
        "category": "RECYCLABLE // ANORGANIK DAUR ULANG",
        "recyclable": True
    },
    "paper": {
        "code": "CLS-04 // PAPER",
        "label": "PAPER [KERTAS]",
        "category": "RECYCLABLE // ANORGANIK DAUR ULANG",
        "recyclable": True
    },
    "plastic": {
        "code": "CLS-05 // PLASTIC",
        "label": "PLASTIC [PLASTIK]",
        "category": "RECYCLABLE // ANORGANIK DAUR ULANG",
        "recyclable": True
    },
    "trash": {
        "code": "CLS-06 // TRASH_RESIDUE",
        "label": "TRASH [RESIDU / NON-RECYCLABLE]",
        "category": "HAZARD / RESIDUE // NON-DAUR ULANG",
        "recyclable": False
    }
}


def inject_industrial_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

        /* 1. HIDE DEFAULT STREAMLIT CHROME */
        #MainMenu {visibility: hidden !important; display: none !important;}
        footer {visibility: hidden !important; display: none !important;}
        header {visibility: hidden !important; display: none !important;}
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        [data-testid="stStatusWidget"] {display: none !important;}
        .stDeployButton {display: none !important;}

        /* 2. GLOBAL RESET & NEO-BRUTALIST ANTI-ROUNDED */
        *, *::before, *::after {
            border-radius: 0px !important;
        }

        /* 3. GRITTY INDUSTRIAL BACKGROUND */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0D0D0F !important;
            background-image:
                linear-gradient(rgba(39, 39, 42, 0.28) 1px, transparent 1px),
                linear-gradient(90deg, rgba(39, 39, 42, 0.28) 1px, transparent 1px);
            background-size: 28px 28px;
            color: #F4F4F5 !important;
            font-family: 'Chakra Petch', sans-serif !important;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px !important;
        }

        /* 4. TYPOGRAPHY */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Chakra Petch', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: #F4F4F5 !important;
            font-weight: 700 !important;
        }

        p, label, span, div {
            color: #F4F4F5;
        }

        code, pre, .mono-val {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* 5. TERMINAL HEADER & HAZARD STRIPE */
        .terminal-header-box {
            background-color: #18181C;
            border: 2px solid #FACC15;
            box-shadow: 4px 4px 0px #FACC15;
            padding: 0;
            margin-bottom: 1.5rem;
        }

        .hazard-stripe {
            height: 12px;
            width: 100%;
            background: repeating-linear-gradient(
                -45deg,
                #FACC15,
                #FACC15 14px,
                #0D0D0F 14px,
                #0D0D0F 28px
            );
            border-bottom: 2px solid #FACC15;
        }

        .hazard-stripe-red {
            height: 10px;
            width: 100%;
            background: repeating-linear-gradient(
                -45deg,
                #FF2A2A,
                #FF2A2A 14px,
                #0D0D0F 14px,
                #0D0D0F 28px
            );
            border-bottom: 2px solid #FF2A2A;
        }

        .terminal-header-inner {
            padding: 1.1rem 1.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .terminal-title {
            font-family: 'Chakra Petch', sans-serif !important;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            color: #FACC15 !important;
            margin: 0;
            line-height: 1.1;
            text-transform: uppercase;
        }

        .terminal-subtitle {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.82rem;
            color: #A1A1AA !important;
            letter-spacing: 0.14em;
            margin-top: 0.35rem;
            text-transform: uppercase;
        }

        .terminal-badge {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem;
            background-color: #0D0D0F;
            color: #FACC15 !important;
            border: 2px solid #FACC15;
            padding: 0.4rem 0.75rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        /* 6. TABS STYLING */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 10px !important;
            background-color: transparent !important;
            border-bottom: 2px solid #27272A !important;
            padding-bottom: 8px !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"] {
            font-family: 'Chakra Petch', sans-serif !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.14em !important;
            font-size: 0.95rem !important;
            background-color: #18181C !important;
            color: #A1A1AA !important;
            border: 2px solid #27272A !important;
            padding: 0.65rem 1.35rem !important;
            transition: none !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            border-color: #FACC15 !important;
            color: #F4F4F5 !important;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #18181C !important;
            color: #FACC15 !important;
            border: 2px solid #FACC15 !important;
            box-shadow: 4px 4px 0px #FACC15 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* 7. RADIO & BUTTONS & UPLOADER */
        [data-testid="stRadio"] > div {
            background-color: #18181C !important;
            border: 2px solid #27272A !important;
            padding: 0.6rem 1rem !important;
        }

        [data-testid="stRadio"] label {
            font-family: 'JetBrains Mono', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-size: 0.85rem !important;
        }

        [data-testid="stFileUploader"] {
            background-color: #18181C !important;
            border: 2px solid #27272A !important;
            padding: 1rem !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #0D0D0F !important;
            border: 2px dashed #FACC15 !important;
        }

        [data-testid="stCameraInput"] {
            background-color: #18181C !important;
            border: 2px solid #27272A !important;
            padding: 1rem !important;
        }

        .stButton > button, [data-testid="stFileUploader"] button, [data-testid="stCameraInput"] button {
            font-family: 'Chakra Petch', sans-serif !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            background-color: #18181C !important;
            color: #FACC15 !important;
            border: 2px solid #FACC15 !important;
            box-shadow: 4px 4px 0px #FACC15 !important;
            padding: 0.45rem 1rem !important;
        }

        .stButton > button:hover, [data-testid="stFileUploader"] button:hover {
            background-color: #FACC15 !important;
            color: #0D0D0F !important;
        }

        /* 8. INDUSTRIAL PANELS & RESULT CARDS */
        .brutal-panel {
            background-color: #18181C;
            border: 2px solid #27272A;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .section-tag {
            font-family: 'Chakra Petch', sans-serif !important;
            font-size: 1.05rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #F4F4F5 !important;
            border-left: 4px solid #FACC15;
            padding-left: 0.65rem;
            margin: 1.2rem 0 0.85rem 0;
        }

        .result-card-yellow {
            background-color: #18181C;
            border: 2px solid #FACC15;
            box-shadow: 4px 4px 0px #FACC15;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }

        .result-card-red {
            background-color: #18181C;
            border: 2px solid #FF2A2A;
            box-shadow: 4px 4px 0px #FF2A2A;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }

        .result-card-neutral {
            background-color: #18181C;
            border: 2px solid #27272A;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }

        .card-tag-label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.76rem;
            color: #A1A1AA !important;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .card-main-value-yellow {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.55rem;
            font-weight: 700;
            color: #FACC15 !important;
            letter-spacing: 0.06em;
            line-height: 1.2;
        }

        .card-main-value-red {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.55rem;
            font-weight: 700;
            color: #FF2A2A !important;
            letter-spacing: 0.06em;
            line-height: 1.2;
        }

        .card-main-value-white {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F4F4F5 !important;
            letter-spacing: 0.06em;
            line-height: 1.2;
        }

        .card-sub-meta {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem;
            color: #A1A1AA !important;
            margin-top: 0.45rem;
            letter-spacing: 0.08em;
        }

        /* 9. IMAGE STAGE FRAMES */
        [data-testid="stImage"] {
            background-color: #18181C !important;
            border: 2px solid #27272A !important;
            padding: 6px !important;
        }

        [data-testid="stImage"] img {
            border: 1px solid #27272A !important;
        }

        .stage-header-bar {
            background-color: #18181C;
            border: 2px solid #27272A;
            border-bottom: 2px solid #FACC15;
            padding: 0.4rem 0.65rem;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem;
            font-weight: 700;
            color: #FACC15 !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        /* 10. CUSTOM INDUSTRIAL TABLES & PROBABILITY BARS */
        table.brutal-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #18181C;
            border: 2px solid #27272A;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }

        table.brutal-table th {
            background-color: #0D0D0F;
            color: #FACC15 !important;
            border: 1px solid #27272A;
            border-bottom: 2px solid #FACC15;
            padding: 0.55rem 0.75rem;
            text-align: left;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-family: 'Chakra Petch', sans-serif !important;
            font-size: 0.85rem;
        }

        table.brutal-table td {
            border: 1px solid #27272A;
            padding: 0.5rem 0.75rem;
            color: #F4F4F5;
            font-family: 'JetBrains Mono', monospace !important;
        }

        table.brutal-table tr:nth-child(even) {
            background-color: #141418;
        }

        .prob-row {
            background-color: #18181C;
            border: 2px solid #27272A;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
        }

        .prob-row-top-yellow {
            background-color: #18181C;
            border: 2px solid #FACC15;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
        }

        .prob-row-top-red {
            background-color: #18181C;
            border: 2px solid #FF2A2A;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
        }

        .prob-track {
            width: 100%;
            height: 10px;
            background-color: #0D0D0F;
            border: 1px solid #27272A;
        }

        .prob-fill-yellow {
            height: 100%;
            background-color: #FACC15;
        }

        .prob-fill-red {
            height: 100%;
            background-color: #FF2A2A;
        }

        .prob-fill-muted {
            height: 100%;
            background-color: #52525B;
        }

        /* 11. DATAFRAME & EXPANDER OVERRIDES */
        [data-testid="stDataFrame"] {
            border: 2px solid #27272A !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        [data-testid="stExpander"] {
            background-color: #18181C !important;
            border: 2px solid #27272A !important;
        }

        [data-testid="stExpander"] summary {
            font-family: 'Chakra Petch', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            color: #FACC15 !important;
        }

        hr {
            border: 0;
            border-top: 2px solid #27272A !important;
            margin: 1.5rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


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


def decode_uploaded_image_to_bgr(file_buffer):
    """
    Membaca buffer gambar dari st.file_uploader atau st.camera_input
    dalam format RGB (via PIL) kemudian mengonversinya secara eksplisit
    ke BGR dengan cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR).
    """
    try:
        file_buffer.seek(0)
        pil_img = Image.open(file_buffer).convert("RGB")
        img_array = np.array(pil_img, dtype=np.uint8)
        image_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception:
        return None


def render_html_table(df):
    headers = "".join([f"<th>{col}</th>" for col in df.columns])
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join([f"<td>{row[col]}</td>" for col in df.columns])
        rows_html.append(f"<tr>{cells}</tr>")
    body = "".join(rows_html)
    return f"<table class='brutal-table'><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


def main():
    inject_industrial_css()

    # TERMINAL HEADER WITH HAZARD STRIPE
    st.markdown(
        """
        <div class="terminal-header-box">
            <div class="hazard-stripe"></div>
            <div class="terminal-header-inner">
                <div>
                    <div class="terminal-title">CLEANERS // WASTE_SCANNER v1.0</div>
                    <div class="terminal-subtitle">DIGITAL IMAGE PROCESSING CLASSIFICATION TERMINAL // TRASHNET-6 PIPELINE</div>
                </div>
                <div class="terminal-badge">SYS_STATUS: ONLINE // 18-FEAT_VECTOR</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    model, scaler, metrics = load_artifacts()

    if model is None or scaler is None or metrics is None:
        st.markdown(
            """
            <div class="result-card-red">
                <div class="hazard-stripe-red" style="margin-bottom: 12px;"></div>
                <div class="card-tag-label">[SYS_ERROR // ARTIFACT_MISSING]</div>
                <div class="card-main-value-red">MODEL ARTIFACTS NOT FOUND IN 'models/'</div>
                <div class="card-sub-meta">EXECUTE 'python train.py' IN TERMINAL TO INITIALIZE CLASSIFIER WEIGHTS.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "[01] SCANNER_UNIT",
        "[02] EVAL_METRICS",
        "[03] SYSTEM_SPEC"
    ])

    # =========================================================================
    # TAB 1: [01] SCANNER_UNIT
    # =========================================================================
    with tab1:
        st.markdown("<div class='section-tag'>// INPUT_ACQUISITION_MODULE</div>", unsafe_allow_html=True)

        input_mode = st.radio(
            "SELECT_INPUT_CHANNEL:",
            ["[MODE_A] UPLOAD_IMAGE_FILE", "[MODE_B] OPTICAL_CAMERA_CAPTURE"],
            horizontal=True
        )

        img_buffer = None
        if input_mode == "[MODE_A] UPLOAD_IMAGE_FILE":
            img_buffer = st.file_uploader(
                "DROP_TARGET_IMAGE_FILE [SUPPORTED: JPG / JPEG / PNG]",
                type=["jpg", "jpeg", "png"]
            )
        else:
            img_buffer = st.camera_input("OPTICAL_SENSOR_FEED // CAPTURE_FRAME")

        if img_buffer is not None:
            image_bgr = decode_uploaded_image_to_bgr(img_buffer)
            if image_bgr is None:
                st.markdown(
                    """
                    <div class="result-card-red">
                        <div class="card-tag-label">[SCAN_ERROR // INVALID_BUFFER]</div>
                        <div class="card-main-value-red">DECODE_FAILURE: CORRUPTED IMAGE STREAM</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
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
                    {
                        "code": f"CLS // {pred_class.upper()}",
                        "label": pred_class.upper(),
                        "category": "UNCLASSIFIED",
                        "recyclable": pred_class != "trash"
                    }
                )
                is_trash = (pred_class == "trash") or (not c_info["recyclable"])

                card_class = "result-card-red" if is_trash else "result-card-yellow"
                val_class = "card-main-value-red" if is_trash else "card-main-value-yellow"
                stripe_class = "hazard-stripe-red" if is_trash else "hazard-stripe"
                status_text = "NON-RECYCLABLE // RESIDUE_HAZARD" if is_trash else "RECYCLABLE // RECOVERY_APPROVED"

                st.markdown("<div class='section-tag'>// TELEMETRY_READOUT</div>", unsafe_allow_html=True)

                res_col1, res_col2, res_col3, res_col4 = st.columns([1.4, 1, 1, 1.3])
                with res_col1:
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="{stripe_class}" style="margin-bottom: 10px;"></div>
                            <div class="card-tag-label">[DETECTED_CLASS]</div>
                            <div class="{val_class}">{c_info['label']}</div>
                            <div class="card-sub-meta">ID: {c_info['code']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with res_col2:
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="card-tag-label">[CONFIDENCE_LVL]</div>
                            <div class="{val_class}">{max_prob:.2f}%</div>
                            <div class="card-sub-meta">MODEL: {metrics.get('best_model', 'RF').upper()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with res_col3:
                    st.markdown(
                        f"""
                        <div class="result-card-neutral">
                            <div class="card-tag-label">[LATENCY_MS]</div>
                            <div class="card-main-value-white">{elapsed_ms:.2f} ms</div>
                            <div class="card-sub-meta">PIPELINE: 4-STAGE + 18-FEAT</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with res_col4:
                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="card-tag-label">[RECYCLABILITY_STATUS]</div>
                            <div class="{val_class}" style="font-size: 1.15rem;">{status_text}</div>
                            <div class="card-sub-meta">{c_info['category']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Visualisasi 4 Tahap Pengolahan Citra Secara Berdampingan
                st.markdown("<div class='section-tag'>// 4-STAGE_OPTICAL_PROCESSING_PIPELINE</div>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("<div class='stage-header-bar'>STAGE_01 // RESIZED_RGB [256x256]</div>", unsafe_allow_html=True)
                    st.image(stages["original"], use_container_width=True)
                with col2:
                    st.markdown("<div class='stage-header-bar'>STAGE_02 // BLUR_5x5 + LAB_CLAHE</div>", unsafe_allow_html=True)
                    st.image(stages["clahe"], use_container_width=True)
                with col3:
                    st.markdown("<div class='stage-header-bar'>STAGE_03 // GRAYSCALE_LUMINANCE</div>", unsafe_allow_html=True)
                    st.image(stages["gray"], use_container_width=True, clamp=True)
                with col4:
                    st.markdown("<div class='stage-header-bar'>STAGE_04 // OTSU_INV + MORPH_CLOSE</div>", unsafe_allow_html=True)
                    st.image(stages["mask"], use_container_width=True, clamp=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                detail_col1, detail_col2 = st.columns([1.25, 1])

                with detail_col1:
                    st.markdown("<div class='section-tag'>// EXTRACTED_FEATURE_VECTOR [18_DIMS]</div>", unsafe_allow_html=True)

                    hsv_df = pd.DataFrame({
                        "IDX": [f"F{i:02d}" for i in range(1, 7)],
                        "FEATURE_ID [HSV_COLOR]": FEATURE_NAMES[0:6],
                        "RAW_VALUE": [f"{float(v):.4f}" for v in features[0:6]],
                        "Z_SCORE": [f"{float(v):+.4f}" for v in features_scaled[0][0:6]]
                    })
                    glcm_df = pd.DataFrame({
                        "IDX": [f"F{i:02d}" for i in range(7, 12)],
                        "FEATURE_ID [GLCM_TEXTURE]": FEATURE_NAMES[6:11],
                        "RAW_VALUE": [f"{float(v):.4f}" for v in features[6:11]],
                        "Z_SCORE": [f"{float(v):+.4f}" for v in features_scaled[0][6:11]]
                    })
                    hu_df = pd.DataFrame({
                        "IDX": [f"F{i:02d}" for i in range(12, 19)],
                        "FEATURE_ID [HU_MOMENTS_LOG]": FEATURE_NAMES[11:18],
                        "RAW_VALUE": [f"{float(v):.4f}" for v in features[11:18]],
                        "Z_SCORE": [f"{float(v):+.4f}" for v in features_scaled[0][11:18]]
                    })

                    f_tab1, f_tab2, f_tab3, f_tab_all = st.tabs([
                        "[A] HSV_COLOR (06)",
                        "[B] GLCM_TEXTURE (05)",
                        "[C] HU_MOMENTS (07)",
                        "[D] FULL_VECTOR (18)"
                    ])
                    with f_tab1:
                        st.markdown(render_html_table(hsv_df), unsafe_allow_html=True)
                    with f_tab2:
                        st.markdown(render_html_table(glcm_df), unsafe_allow_html=True)
                    with f_tab3:
                        st.markdown(render_html_table(hu_df), unsafe_allow_html=True)
                    with f_tab_all:
                        all_features_df = pd.DataFrame({
                            "IDX": [f"F{i:02d}" for i in range(1, 19)],
                            "DOMAIN": (["HSV_COLOR"] * 6) + (["GLCM_TEXTURE"] * 5) + (["HU_SHAPE"] * 7),
                            "FEATURE_NAME": FEATURE_NAMES,
                            "RAW_VALUE": [f"{float(v):.6f}" for v in features],
                            "Z_SCORE": [f"{float(v):+.4f}" for v in features_scaled[0]]
                        })
                        st.markdown(render_html_table(all_features_df), unsafe_allow_html=True)

                with detail_col2:
                    st.markdown("<div class='section-tag'>// CLASS_PROBABILITY_DISTRIBUTION</div>", unsafe_allow_html=True)
                    prob_pairs = sorted(
                        zip(classes, probs),
                        key=lambda x: float(x[1]),
                        reverse=True
                    )
                    prob_bars_html = []
                    for rank, (cls_key, p_val) in enumerate(prob_pairs):
                        pct = float(p_val) * 100.0
                        cls_meta = CLASS_INFO.get(cls_key, {"label": cls_key.upper(), "recyclable": cls_key != "trash"})
                        if rank == 0:
                            if cls_key == "trash":
                                row_cls = "prob-row-top-red"
                                fill_cls = "prob-fill-red"
                                txt_color = "#FF2A2A"
                            else:
                                row_cls = "prob-row-top-yellow"
                                fill_cls = "prob-fill-yellow"
                                txt_color = "#FACC15"
                        else:
                            row_cls = "prob-row"
                            fill_cls = "prob-fill-muted"
                            txt_color = "#F4F4F5"

                        prob_bars_html.append(
                            f"""
                            <div class="{row_cls}">
                                <div class="prob-header">
                                    <span style="color: {txt_color}; font-weight: 700;">{cls_meta['label']}</span>
                                    <span style="color: {txt_color}; font-weight: 700;">{pct:.2f}%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="{fill_cls}" style="width: {max(pct, 1.0):.2f}%;"></div>
                                </div>
                            </div>
                            """
                        )
                    st.markdown("".join(prob_bars_html), unsafe_allow_html=True)

        else:
            st.markdown(
                """
                <div class="brutal-panel" style="border-left: 4px solid #FACC15;">
                    <div class="card-tag-label">[SCANNER_STANDBY // AWAITING_OPTICAL_INPUT]</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #F4F4F5;">
                        UPLOAD TARGET WASTE IMAGE OR ACTIVATE CAMERA FEED TO EXECUTE 4-STAGE PCD PIPELINE &amp; 18-FEATURE EXTRACTION.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            sample_pipe_path = os.path.join(MODEL_DIR, "sample_pipeline.png")
            if os.path.exists(sample_pipe_path):
                with st.expander("[REF_LOG] VIEW_SAMPLE_4-STAGE_PROCESSING_OUTPUT"):
                    st.image(sample_pipe_path, caption="SAMPLE_PIPELINE // TRASHNET_REFERENCE_LOG", use_container_width=True)

    # =========================================================================
    # TAB 2: [02] EVAL_METRICS
    # =========================================================================
    with tab2:
        st.markdown("<div class='section-tag'>// CLASSIFIER_BENCHMARK_TELEMETRY</div>", unsafe_allow_html=True)

        best_name = metrics.get("best_model", "Random Forest")
        acc = metrics.get("accuracy", 0.0) * 100.0
        prec_w = metrics.get("precision_weighted", metrics.get("report", {}).get("weighted avg", {}).get("precision", 0.0)) * 100.0
        rec_w = metrics.get("recall_weighted", metrics.get("report", {}).get("weighted avg", {}).get("recall", 0.0)) * 100.0
        f1_w = metrics.get("f1_weighted", metrics.get("report", {}).get("weighted avg", {}).get("f1-score", 0.0)) * 100.0

        st.markdown(
            f"""
            <div class="brutal-panel" style="border-left: 4px solid #FACC15;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #A1A1AA;">
                    ACTIVE_MODEL: <strong style="color: #FACC15;">{best_name.upper()}</strong> &nbsp;//&nbsp;
                    DATASET_TOTAL: <strong style="color: #F4F4F5;">{metrics.get('dataset_total', 2527)}</strong> &nbsp;//&nbsp;
                    TRAIN_SPLIT (80%): <strong style="color: #F4F4F5;">{metrics.get('train_size', 2021)}</strong> &nbsp;//&nbsp;
                    TEST_SPLIT (20%): <strong style="color: #F4F4F5;">{metrics.get('test_size', 506)}</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"""
                <div class="result-card-yellow">
                    <div class="card-tag-label">[TEST_ACCURACY]</div>
                    <div class="card-main-value-yellow">{acc:.2f}%</div>
                    <div class="card-sub-meta">STRATIFIED 20% SPLIT</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="result-card-neutral">
                    <div class="card-tag-label">[PRECISION_WEIGHTED]</div>
                    <div class="card-main-value-white">{prec_w:.2f}%</div>
                    <div class="card-sub-meta">6-CLASS WEIGHTED AVG</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="result-card-neutral">
                    <div class="card-tag-label">[RECALL_WEIGHTED]</div>
                    <div class="card-main-value-white">{rec_w:.2f}%</div>
                    <div class="card-sub-meta">6-CLASS WEIGHTED AVG</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="result-card-neutral">
                    <div class="card-tag-label">[F1_SCORE_WEIGHTED]</div>
                    <div class="card-main-value-white">{f1_w:.2f}%</div>
                    <div class="card-sub-meta">HARMONIC MEAN</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-tag'>// MODEL_COMPARISON_MATRIX [RANDOM_FOREST vs SVM_RBF]</div>", unsafe_allow_html=True)
        comp = metrics.get("model_comparison", {})
        if comp:
            comp_rows = []
            for m_name, m_data in comp.items():
                comp_rows.append({
                    "CLASSIFIER_MODEL": m_name.upper(),
                    "ACCURACY": f"{m_data.get('accuracy', 0.0) * 100:.2f}%",
                    "PREC_MACRO": f"{m_data.get('precision_macro', 0.0) * 100:.2f}%",
                    "REC_MACRO": f"{m_data.get('recall_macro', 0.0) * 100:.2f}%",
                    "F1_MACRO": f"{m_data.get('f1_macro', 0.0) * 100:.2f}%",
                    "PREC_WEIGHTED": f"{m_data.get('precision_weighted', 0.0) * 100:.2f}%",
                    "REC_WEIGHTED": f"{m_data.get('recall_weighted', 0.0) * 100:.2f}%",
                    "F1_WEIGHTED": f"{m_data.get('f1_weighted', 0.0) * 100:.2f}%",
                    "DEPLOY_STATUS": "[ACTIVE // BEST]" if m_name == best_name else "[STANDBY]"
                })
            comp_df = pd.DataFrame(comp_rows)
            st.markdown(render_html_table(comp_df), unsafe_allow_html=True)

        eval_col1, eval_col2 = st.columns([1.15, 1])

        with eval_col1:
            st.markdown(f"<div class='section-tag'>// PER_CLASS_METRICS [{best_name.upper()}]</div>", unsafe_allow_html=True)
            per_class = metrics.get("per_class", {})
            if not per_class and "report" in metrics:
                for c in metrics.get("classes", []):
                    if c in metrics["report"]:
                        per_class[c] = metrics["report"][c]

            class_rows = []
            for cls_name, cls_m in per_class.items():
                class_rows.append({
                    "WASTE_CLASS": CLASS_INFO.get(cls_name, {"label": cls_name.upper()})["label"],
                    "PRECISION": f"{cls_m.get('precision', 0.0) * 100:.2f}%",
                    "RECALL": f"{cls_m.get('recall', 0.0) * 100:.2f}%",
                    "F1_SCORE": f"{cls_m.get('f1-score', 0.0) * 100:.2f}%",
                    "TEST_SUPPORT": int(cls_m.get("support", 0))
                })
            class_df = pd.DataFrame(class_rows)
            st.markdown(render_html_table(class_df), unsafe_allow_html=True)

        with eval_col2:
            st.markdown("<div class='section-tag'>// CONFUSION_MATRIX_PLOT</div>", unsafe_allow_html=True)
            cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption=f"CONFUSION_MATRIX // {best_name.upper()}", use_container_width=True)
            else:
                st.markdown(
                    "<div class='result-card-red'><div class='card-main-value-red'>CONFUSION_MATRIX.PNG MISSING</div></div>",
                    unsafe_allow_html=True
                )

    # =========================================================================
    # TAB 3: [03] SYSTEM_SPEC
    # =========================================================================
    with tab3:
        st.markdown("<div class='section-tag'>// DIGITAL_IMAGE_PROCESSING_PIPELINE_ARCHITECTURE</div>", unsafe_allow_html=True)

        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.markdown(
                """
                <div class="brutal-panel">
                    <div class="card-tag-label">[STAGE_01 // IMAGE_PREPROCESSING]</div>
                    <ul style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; line-height: 1.65; color: #F4F4F5;">
                        <li><strong>SPATIAL_RESIZE:</strong> Bicubic/Linear interpolation to fixed <code>256 x 256</code> px grid.</li>
                        <li><strong>NOISE_SUPPRESSION:</strong> Low-pass <code>GaussianBlur</code> with kernel <code>5 x 5</code>.</li>
                        <li><strong>CONTRAST_ENHANCEMENT:</strong> RGB/BGR to <code>CIE-LAB</code> conversion; <code>CLAHE</code> (<code>clipLimit=2.0</code>, <code>tileGridSize=(8,8)</code>) applied strictly on Luminance channel <code>L</code>.</li>
                    </ul>
                </div>

                <div class="brutal-panel">
                    <div class="card-tag-label">[STAGE_02 // OBJECT_SEGMENTATION]</div>
                    <ul style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; line-height: 1.65; color: #F4F4F5;">
                        <li><strong>GRAYSCALE_PROJECTION:</strong> Single-channel intensity map extracted from CLAHE-enhanced BGR frame.</li>
                        <li><strong>OTSU_BINARIZATION:</strong> Automatic bimodal histogram thresholding via <code>THRESH_BINARY_INV + THRESH_OTSU</code>.</li>
                        <li><strong>MORPHOLOGICAL_CLOSING:</strong> Dilation followed by Erosion using a <code>5 x 5</code> rectangular structuring element to seal internal voids.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with spec_col2:
            st.markdown(
                """
                <div class="brutal-panel">
                    <div class="card-tag-label">[STAGE_03 // 18-DIMENSIONAL_FEATURE_EXTRACTION]</div>
                    <ul style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; line-height: 1.65; color: #F4F4F5;">
                        <li><strong>COLOR_DESCRIPTORS [06_FEATS]:</strong> First &amp; second-order statistical moments (Mean &amp; Std Dev) across <code>Hue (H)</code>, <code>Saturation (S)</code>, and <code>Value (V)</code> channels.</li>
                        <li><strong>TEXTURE_DESCRIPTORS [05_FEATS]:</strong> Haralick <code>GLCM</code> co-occurrence matrices at distance <code>d=1</code> averaged across 4 orientations (<code>0 deg, 45 deg, 90 deg, 135 deg</code>): <em>Contrast, Dissimilarity, Homogeneity, Energy, Correlation</em>.</li>
                        <li><strong>SHAPE_DESCRIPTORS [07_FEATS]:</strong> 7 orthogonal <code>Hu Moments</code> derived from central moments of the binary segmentation mask, normalized via logarithmic scale transform: <code>-sign(h) * log10(|h| + 1e-10)</code>.</li>
                    </ul>
                </div>

                <div class="brutal-panel">
                    <div class="card-tag-label">[STAGE_04 // NORMALIZATION_&_CLASSIFICATION]</div>
                    <ul style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; line-height: 1.65; color: #F4F4F5;">
                        <li><strong>STRATIFIED_PARTITION:</strong> 80% Train (2,021 frames) / 20% Test (506 frames) with <code>random_state=42</code>.</li>
                        <li><strong>Z-SCORE_SCALING:</strong> <code>StandardScaler</code> fitted strictly on training partition to prevent data leakage.</li>
                        <li><strong>CLASSIFIER_ARENA:</strong> Benchmarking <code>RandomForestClassifier(n_estimators=200, class_weight='balanced')</code> against <code>SVC(kernel='rbf', class_weight='balanced')</code>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div class='section-tag'>// CORE_DEPENDENCY_MANIFEST</div>", unsafe_allow_html=True)
        lib_df = pd.DataFrame([
            {"MODULE_NAME": "OPENCV-PYTHON-HEADLESS (cv2)", "SYSTEM_ROLE": "Spatial resizing, Gaussian filtering, LAB/HSV/Gray color space conversions, CLAHE, Otsu thresholding, Morphological Closing, Hu Moments."},
            {"MODULE_NAME": "SCIKIT-IMAGE (skimage.feature)", "SYSTEM_ROLE": "Gray-Level Co-occurrence Matrix (graycomatrix & graycoprops) multi-angle texture extraction."},
            {"MODULE_NAME": "SCIKIT-LEARN (sklearn)", "SYSTEM_ROLE": "Stratified train/test partitioning, StandardScaler normalization, Random Forest & SVM (RBF) training, and evaluation telemetry."},
            {"MODULE_NAME": "NUMPY", "SYSTEM_ROLE": "High-speed multidimensional array operations, HSV statistical moments, and logarithmic Hu Moment transformations."},
            {"MODULE_NAME": "PANDAS", "SYSTEM_ROLE": "Structured tabular formatting for 18-dimensional feature readouts and classifier comparison matrices."},
            {"MODULE_NAME": "MATPLOTLIB & SEABORN", "SYSTEM_ROLE": "Generation of Confusion Matrix heatmaps and 4-stage pipeline documentation figures."},
            {"MODULE_NAME": "JOBLIB", "SYSTEM_ROLE": "Binary serialization of trained classifier weights (.joblib) and StandardScaler parameters."},
            {"MODULE_NAME": "STREAMLIT", "SYSTEM_ROLE": "Industrial web terminal interface for live waste scanning and telemetry visualization."}
        ])
        st.markdown(render_html_table(lib_df), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
