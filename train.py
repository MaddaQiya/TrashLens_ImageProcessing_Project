import os
import json
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from image_processor import process_image, FEATURE_NAMES

EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def resolve_dataset_dir(base_dir="dataset-resized"):
    """
    Deteksi otomatis path dataset TrashNet apabila terekstrak ganda
    di `dataset-resized/dataset-resized/` atau langsung di `dataset-resized/`.
    """
    candidates = [
        base_dir,
        os.path.join(base_dir, "dataset-resized"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            subdirs = [
                d for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d)) and not d.startswith("_") and not d.startswith(".")
            ]
            if set(EXPECTED_CLASSES).issubset(set(subdirs)):
                return path
    raise FileNotFoundError(
        f"Folder dataset dengan 6 kelas {EXPECTED_CLASSES} tidak ditemukan di '{base_dir}'."
    )


def save_sample_pipeline(sample_img_bgr, sample_label, output_path):
    """
    Menyimpan 1 contoh visualisasi citra sebelum-sesudah (4 tahap:
    Asli/Resized, CLAHE, Grayscale, Binary Mask) ke models/sample_pipeline.png.
    """
    _, stages = process_image(sample_img_bgr)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(
        f"Tahapan Pengolahan Citra Digital - Sampel Kelas: {sample_label.capitalize()}",
        fontsize=14,
        fontweight="bold"
    )

    axes[0].imshow(stages["original"])
    axes[0].set_title("1. Citra Asli (Resized 256x256)")
    axes[0].axis("off")

    axes[1].imshow(stages["clahe"])
    axes[1].set_title("2. Blur + Peningkatan Kontras (CLAHE)")
    axes[1].axis("off")

    axes[2].imshow(stages["gray"], cmap="gray")
    axes[2].set_title("3. Konversi Grayscale")
    axes[2].axis("off")

    axes[3].imshow(stages["mask"], cmap="gray")
    axes[3].set_title("4. Segmentasi (Otsu + Closing 5x5)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(model, X_test_scaled, y_test, classes):
    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    per_class = {}
    for cls in classes:
        cls_metrics = report_dict.get(cls, {})
        per_class[cls] = {
            "precision": float(cls_metrics.get("precision", 0.0)),
            "recall": float(cls_metrics.get("recall", 0.0)),
            "f1-score": float(cls_metrics.get("f1-score", 0.0)),
            "support": int(cls_metrics.get("support", 0))
        }

    return {
        "accuracy": acc,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "report": report_dict,
        "report_text": report_text,
        "cm": cm,
        "y_pred": y_pred
    }


def main():
    dataset_dir = resolve_dataset_dir("dataset-resized")
    print(f"Dataset terdeteksi di: {os.path.abspath(dataset_dir)}")

    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d in EXPECTED_CLASSES
    ])

    print(f"Memulai ekstraksi 18 fitur untuk {len(classes)} kelas: {classes}...")

    X, y = [], []
    class_counts = {}
    sample_saved = False

    for label in classes:
        folder_path = os.path.join(dataset_dir, label)
        count = 0
        for filename in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is not None:
                features, _ = process_image(img)
                X.append(features)
                y.append(label)
                count += 1

                if not sample_saved:
                    sample_pipeline_path = os.path.join(MODEL_DIR, "sample_pipeline.png")
                    save_sample_pipeline(img, label, sample_pipeline_path)
                    sample_saved = True
                    print(f"Contoh visualisasi 4 tahap disimpan ke: {sample_pipeline_path}")

        class_counts[label] = count
        print(f"  - Kelas '{label}': {count} citra")

    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    print(f"Selesai ekstraksi fitur! Total citra diproses: {len(X)} (Dimensi fitur: {X.shape[1]})")

    # Bagi dataset 80% Train dan 20% Test secara stratified (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split Dataset: Train = {len(X_train)} citra, Test = {len(X_test)} citra")

    # Normalisasi StandardScaler (fit hanya di data latih untuk mencegah data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Latih Model Random Forest
    print("\n[1/2] Melatih model Random Forest (n_estimators=200, class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_eval = evaluate_model(rf_model, X_test_scaled, y_test, classes)
    print(f"Akurasi Random Forest: {rf_eval['accuracy'] * 100:.2f}%")
    print(rf_eval["report_text"])

    # 2. Latih Model SVM (RBF kernel)
    print("[2/2] Melatih model SVM (kernel='rbf', class_weight='balanced', probability=True)...")
    svm_model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    svm_eval = evaluate_model(svm_model, X_test_scaled, y_test, classes)
    print(f"Akurasi SVM (RBF): {svm_eval['accuracy'] * 100:.2f}%")
    print(svm_eval["report_text"])

    # Pilih model dengan akurasi terbaik
    if rf_eval["accuracy"] >= svm_eval["accuracy"]:
        best_model_name = "Random Forest"
        best_model = rf_model
        best_eval = rf_eval
        best_model_filename = "best_model.joblib"
    else:
        best_model_name = "SVM (RBF Kernel)"
        best_model = svm_model
        best_eval = svm_eval
        best_model_filename = "best_model.joblib"

    print(f"==> Model Terbaik: {best_model_name} (Akurasi: {best_eval['accuracy'] * 100:.2f}%)")

    # Simpan Confusion Matrix dari model terbaik
    cm = best_eval["cm"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(f"Confusion Matrix Klasifikasi Sampah ({best_model_name})")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Simpan model terbaik dan scaler
    joblib.dump(best_model, os.path.join(MODEL_DIR, best_model_filename))
    # Simpan juga dengan nama rf_model.joblib / svm_model.joblib agar fleksibel
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # Susun struktur metrics.json lengkap
    metrics_data = {
        "best_model": best_model_name,
        "best_model_file": best_model_filename,
        "dataset_total": int(len(X)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution": class_counts,
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "accuracy": best_eval["accuracy"],
        "precision_macro": best_eval["precision_macro"],
        "recall_macro": best_eval["recall_macro"],
        "f1_macro": best_eval["f1_macro"],
        "precision_weighted": best_eval["precision_weighted"],
        "recall_weighted": best_eval["recall_weighted"],
        "f1_weighted": best_eval["f1_weighted"],
        "per_class": best_eval["per_class"],
        "report": best_eval["report"],
        "model_comparison": {
            "Random Forest": {
                "accuracy": rf_eval["accuracy"],
                "precision_macro": rf_eval["precision_macro"],
                "recall_macro": rf_eval["recall_macro"],
                "f1_macro": rf_eval["f1_macro"],
                "precision_weighted": rf_eval["precision_weighted"],
                "recall_weighted": rf_eval["recall_weighted"],
                "f1_weighted": rf_eval["f1_weighted"],
                "per_class": rf_eval["per_class"],
                "report": rf_eval["report"]
            },
            "SVM (RBF Kernel)": {
                "accuracy": svm_eval["accuracy"],
                "precision_macro": svm_eval["precision_macro"],
                "recall_macro": svm_eval["recall_macro"],
                "f1_macro": svm_eval["f1_macro"],
                "precision_weighted": svm_eval["precision_weighted"],
                "recall_weighted": svm_eval["recall_weighted"],
                "f1_weighted": svm_eval["f1_weighted"],
                "per_class": svm_eval["per_class"],
                "report": svm_eval["report"]
            }
        }
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model terbaik ({best_model_filename}), scaler.joblib, metrics.json, confusion_matrix.png, dan sample_pipeline.png berhasil disimpan di folder '{MODEL_DIR}/'!")


if __name__ == "__main__":
    main()
