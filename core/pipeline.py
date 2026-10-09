"""
Birleşik Analiz Pipeline'ı
==========================
YOLO çay tespiti + RF analizi zincirini tek yerden yönetir.
Hem tek hem toplu analiz bunu kullanır.
"""

import cv2
from pathlib import Path
from core.analyzer import TeaAnalyzer
from core.tea_detector import TeaDetector


class AnalysisPipeline:
    """YOLO + RF birleşik pipeline."""

    def __init__(self, rf_model_path, yolo_model_path=None,
                 diger_threshold=0.60, downscale=0.5,
                 yolo_conf=0.35):
        self.analyzer = TeaAnalyzer(
            str(rf_model_path),
            diger_threshold=diger_threshold,
            downscale=downscale,
        )
        # YOLO opsiyonel - dosya yoksa None
        self.detector = None
        if yolo_model_path and Path(yolo_model_path).exists():
            try:
                self.detector = TeaDetector(str(yolo_model_path),
                                            conf_threshold=yolo_conf)
            except Exception as e:
                print(f"YOLO yüklenemedi: {e}")
                self.detector = None

    @property
    def has_yolo(self):
        return self.detector is not None

    def analyze(self, image_path, generate_visuals=True, use_tea_detection=True):
        """
        Tam pipeline: (varsa) YOLO çay tespiti + RF analizi.

        use_tea_detection: False ise YOLO atlanır, tüm görüntü analiz edilir.
        """
        tea_mask = None
        tea_info = None

        if self.detector is not None and use_tea_detection:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask, n, conf = self.detector.detect_tea_mask(img_rgb)
                if n > 0:  # çay tespit edildiyse maskeyi kullan
                    tea_mask = mask
                    tea_info = {"n_segments": n, "confidence": conf}

        return self.analyzer.analyze(
            image_path,
            generate_visuals=generate_visuals,
            tea_mask=tea_mask,
            tea_info=tea_info,
        )