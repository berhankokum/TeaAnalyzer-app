"""
YOLO ile Çay Alanı Tespiti
===========================
Eğitilmiş YOLO11-seg modeliyle drone görüntülerinde çay alanlarını
piksel düzeyinde maskeler.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class TeaDetector:
    """YOLO instance segmentation ile çay alanı tespiti."""

    def __init__(self, model_path, conf_threshold=0.35, iou_threshold=0.5):
        """
        model_path: tea_yolo.pt yolu
        conf_threshold: tespit güven eşiği (0-1)
        iou_threshold: NMS için IoU eşiği
        """
        self.model = YOLO(str(model_path))
        self.conf = conf_threshold
        self.iou = iou_threshold

    def detect_tea_mask(self, img_rgb):
        """
        Görüntüde çay alanlarını tespit et, birleşik bir maske döndür.

        Args:
            img_rgb: (H, W, 3) RGB numpy array

        Returns:
            tea_mask: (H, W) bool array, True = çay olan piksel
            n_detections: tespit edilen çay segmentinin sayısı
            avg_confidence: ortalama güven skoru (yoksa 0)
        """
        h, w = img_rgb.shape[:2]
        # YOLO BGR bekler
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Tahmin (verbose=False -> konsol gürültüsü olmasın)
        results = self.model.predict(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            retina_masks=True,  # tam çözünürlük maske
        )

        if not results or results[0].masks is None:
            return np.zeros((h, w), dtype=bool), 0, 0.0

        result = results[0]
        masks = result.masks.data.cpu().numpy()  # (N, h_mask, w_mask) float 0-1
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

        # Birleşik maske
        combined = np.zeros((h, w), dtype=bool)
        for m in masks:
            # Maske boyutu girdi boyutuyla aynı olmayabilir, hizala
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            combined |= (m > 0.5)

        avg_conf = float(np.mean(confidences)) if len(confidences) else 0.0
        return combined, len(masks), avg_conf


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    MODEL = r"C:\Users\berhan\Desktop\tez\TeaAnalyzer\models\tea_yolo.pt"
    TEST_IMG = r"C:\Users\berhan\Desktop\tez\TeaAnalyzer\4.png"

    print("YOLO detector test ediliyor...")
    detector = TeaDetector(MODEL)

    img_bgr = cv2.imread(TEST_IMG)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"Görüntü boyutu: {img_rgb.shape}")

    mask, n, conf = detector.detect_tea_mask(img_rgb)
    h, w = mask.shape
    pct = 100 * mask.sum() / (h * w)
    print(f"Tespit edilen çay segment sayısı: {n}")
    print(f"Ortalama güven: {conf:.3f}")
    print(f"Çay maskesi kapsama: %{pct:.1f}")

    # Görsel kayıt: maske overlay
    if n > 0:
        overlay = img_rgb.copy()
        green = np.zeros_like(img_rgb)
        green[..., 1] = 255
        overlay = np.where(mask[..., None],
                           cv2.addWeighted(img_rgb, 0.5, green, 0.5, 0),
                           img_rgb // 3)  # dışı koyulaştır
        cv2.imwrite("test_tea_mask.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("Test görseli kaydedildi: test_tea_mask.png")