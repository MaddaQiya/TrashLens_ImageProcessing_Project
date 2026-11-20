import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES = [
    # Ciri Warna (HSV Color Moments: 6 fitur)
    "HSV_H_Mean", "HSV_H_Std",
    "HSV_S_Mean", "HSV_S_Std",
    "HSV_V_Mean", "HSV_V_Std",
    # Ciri Tekstur (GLCM: 5 fitur)
    "GLCM_Contrast",
    "GLCM_Dissimilarity",
    "GLCM_Homogeneity",
    "GLCM_Energy",
    "GLCM_Correlation",
    # Ciri Bentuk (Hu Moments: 7 fitur)
    "Hu_Moment_1",
    "Hu_Moment_2",
    "Hu_Moment_3",
    "Hu_Moment_4",
    "Hu_Moment_5",
    "Hu_Moment_6",
    "Hu_Moment_7"
]

def process_image(image_bgr, size=(256, 256)):
    """
    Melakukan prapemrosesan, segmentasi, dan ekstraksi fitur pada satu citra.
    
    Tahapan:
    a. Prapemrosesan: Resize (256x256), Gaussian Blur (5x5), dan CLAHE pada kanal L (ruang warna LAB).
    b. Segmentasi: Konversi ke Grayscale, Otsu Thresholding (THRESH_BINARY_INV + THRESH_OTSU),
       dan Operasi Morfologi Closing (kernel 5x5) untuk menghasilkan binary mask.
    c. Ekstraksi Fitur (Total 18 fitur):
       - Ciri Warna: Mean & Standard Deviation dari 3 kanal HSV (6 fitur).
       - Ciri Tekstur: GLCM (contrast, dissimilarity, homogeneity, energy, correlation) pada 4 sudut (5 fitur).
       - Ciri Bentuk: 7 Hu Moments dengan transformasi log dari binary mask (7 fitur).

    Mengembalikan:
    - feature_vector: numpy 1D array berisi 18 fitur
    - stages: dictionary berisi citra setiap tahapan (Asli/Resized, CLAHE, Grayscale, Binary Mask)
    """
    # 1. Prapemrosesan: Resize ke 256x256
    resized = cv2.resize(image_bgr, size)
    
    # 2. Prapemrosesan: Noise Filtering (Gaussian Blur 5x5)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    # 3. Prapemrosesan: Peningkatan Kontras (CLAHE pada kanal L di ruang warna LAB)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # 4. Segmentasi: Grayscale + Otsu Thresholding + Morfologi Closing (kernel 5x5)
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # --- EKSTRAKSI FITUR (Total 18 Fitur) ---
    # A. Ciri Warna (Mean & Std Dev dari 3 kanal HSV = 6 fitur)
    hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    color_features = []
    for i in range(3):
        channel = hsv[:, :, i]
        color_features.append(float(np.mean(channel)))
        color_features.append(float(np.std(channel)))
        
    # B. Ciri Tekstur (GLCM pada 4 sudut: 0, 45, 90, 135 derajat = 5 fitur)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    texture_features = [
        float(np.mean(graycoprops(glcm, 'contrast'))),
        float(np.mean(graycoprops(glcm, 'dissimilarity'))),
        float(np.mean(graycoprops(glcm, 'homogeneity'))),
        float(np.mean(graycoprops(glcm, 'energy'))),
        float(np.mean(graycoprops(glcm, 'correlation')))
    ]
    
    # C. Ciri Bentuk (7 Hu Moments dengan transformasi log dari binary mask = 7 fitur)
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    shape_features = [
        float(-np.sign(h) * np.log10(abs(h) + 1e-10)) if h != 0 else 0.0
        for h in hu_moments
    ]
    
    # Gabungkan semua fitur (Total: 18 fitur)
    feature_vector = np.array(color_features + texture_features + shape_features, dtype=np.float64)
    
    # Simpan citra setiap tahapan untuk keperluan visualisasi
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    clahe_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    stages = {
        "original": resized_rgb,
        "clahe": clahe_rgb,
        "gray": gray,
        "mask": mask,
        # Alias kompatibilitas
        "resized_rgb": resized_rgb,
        "enhanced_rgb": clahe_rgb
    }
    
    return feature_vector, stages
