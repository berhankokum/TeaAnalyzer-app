"""
Çay Bahçesi Analiz Motoru
==========================
Tüm görüntü işleme mantığını içeren çekirdek sınıf.
Arayüzden bağımsız - ayrı test edilebilir.
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """Tek bir görüntünün analiz sonucu."""
    filename: str = ""
    image_width: int = 0
    image_height: int = 0

    # Yüzdeler
    vegetation_pct: float = 0.0
    shadow_pct: float = 0.0
    other_pct: float = 0.0

    # Vejetasyon indeksleri (sadece vejetasyon piksellerinde)
    exg_mean: float = 0.0
    gli_mean: float = 0.0
    vari_mean: float = 0.0

    # Heterojenlik (parsel içi tutarlılık göstergesi)
    exg_std: float = 0.0

    # Renk tonu dağılımı (çay yaprağı yeşil-sarı analizi)
    yellow_green_pct: float = 0.0   # sarımsı-açık yeşil (stres/genç)
    light_green_pct: float = 0.0    # açık yeşil
    mid_green_pct: float = 0.0      # orta yeşil (sağlıklı)
    dark_green_pct: float = 0.0     # koyu yeşil (olgun)
    dominant_color: str = ""        # baskın ton açıklaması

    # Görseller (numpy array olarak tutulur)
    original_rgb: np.ndarray = field(default=None, repr=False)
    segmentation_rgb: np.ndarray = field(default=None, repr=False)
    overlay_rgb: np.ndarray = field(default=None, repr=False)

    # YOLO çay maskesi bilgisi (varsa)
    tea_mask: np.ndarray = field(default=None, repr=False)
    tea_coverage_pct: float = 0.0  # görüntünün ne kadarı çay
    n_tea_segments: int = 0
    tea_confidence: float = 0.0
    analysis_mode: str = "full_image"  # "full_image" veya "tea_only"

    def to_dict(self):
        return {
            "Dosya": self.filename,
            "Mod": "Çay alanı" if self.analysis_mode == "tea_only" else "Tüm görüntü",
            "Çay Kapsamı (%)": round(self.tea_coverage_pct, 1),
            "Yeşil Örtü (%)": round(self.vegetation_pct, 1),
            "Gölge (%)": round(self.shadow_pct, 1),
            "Sarımsı Yeşil (%)": round(self.yellow_green_pct, 1),
            "Açık Yeşil (%)": round(self.light_green_pct, 1),
            "Orta Yeşil (%)": round(self.mid_green_pct, 1),
            "Koyu Yeşil (%)": round(self.dark_green_pct, 1),
            "Baskın Ton": self.dominant_color,
            "ExG": round(self.exg_mean, 3),
            "GLI": round(self.gli_mean, 3),
            "VARI": round(self.vari_mean, 3),
        }


class TeaAnalyzer:
    """Çay bahçesi görüntü analiz motoru."""

    # Görselleştirme renkleri (RGB)
    COLOR_VEGETATION = (0, 200, 0)
    COLOR_SHADOW = (50, 50, 50)
    COLOR_OTHER = (200, 200, 200)

    def __init__(self, model_path, diger_threshold=0.60, downscale=0.5):
        """
        model_path: rf_model_binary.pkl yolu
        diger_threshold: bu olasılığın altı "belirsiz" sayılır
        downscale: hız için görüntü küçültme oranı (0.5 = yarı boyut)
        """
        self.model = joblib.load(model_path)
        self.diger_threshold = diger_threshold
        self.downscale = downscale

    # --------------------------------------------------------
    # ÖZELLİK ÇIKARIM
    # --------------------------------------------------------
    def _extract_features(self, img_rgb):
        """RGB görüntüden 8 özellik haritası çıkar (RF girdisi)."""
        rgb = img_rgb.reshape(-1, 3).astype(np.float32)
        R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        bgr = img_rgb.reshape(-1, 1, 3)[:, :, [2, 1, 0]].astype(np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        ExG = 2 * G - R - B
        denom = 2 * G + R + B
        # Sıfır bölme uyarısını engelle: önce güvenli payda oluştur
        GLI = np.zeros_like(G)
        safe = denom > 1e-6
        GLI[safe] = (2 * G[safe] - R[safe] - B[safe]) / denom[safe] * 255

        return np.column_stack([R, G, B, H, S, V, ExG, GLI])

    # --------------------------------------------------------
    # İNDEKS HESAPLAMA (robust)
    # --------------------------------------------------------
    def _compute_indices(self, img_rgb, veg_mask):
        """Vejetasyon piksellerinde robust indeks hesabı."""
        if not veg_mask.any():
            return dict(exg_mean=0, exg_std=0, gli_mean=0, vari_mean=0)

        img = img_rgb.astype(np.float32)
        R = img[..., 0][veg_mask]
        G = img[..., 1][veg_mask]
        B = img[..., 2][veg_mask]

        ExG = 2 * G - R - B

        denom_gli = 2 * G + R + B
        GLI = np.where(denom_gli > 1e-3, (2 * G - R - B) / denom_gli, 0)

        denom_vari = G + R - B
        valid = np.abs(denom_vari) > 5.0
        VARI = np.full_like(R, np.nan)
        VARI[valid] = (G[valid] - R[valid]) / denom_vari[valid]
        VARI = np.clip(VARI, -1.0, 1.0)

        def robust_mean(arr):
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return 0.0
            lo, hi = np.percentile(arr, [1, 99])
            clean = arr[(arr >= lo) & (arr <= hi)]
            return float(clean.mean()) if len(clean) else 0.0

        def robust_std(arr):
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                return 0.0
            lo, hi = np.percentile(arr, [1, 99])
            clean = arr[(arr >= lo) & (arr <= hi)]
            return float(clean.std()) if len(clean) else 0.0

        return dict(
            exg_mean=robust_mean(ExG),
            exg_std=robust_std(ExG),
            gli_mean=robust_mean(GLI),
            vari_mean=robust_mean(VARI),
        )

    def _compute_hue_distribution(self, img_rgb, veg_mask):
        """
        Vejetasyon piksellerinde yeşil-sarı renk tonu dağılımını hesapla.
        HSV uzayındaki H (Hue) kanalı kullanılır.
        OpenCV H aralığı: 0-180
        """
        if not veg_mask.any():
            return dict(
                yellow_green_pct=0.0,
                light_green_pct=0.0,
                mid_green_pct=0.0,
                dark_green_pct=0.0,
                dominant_color="Yetersiz veri",
            )

        # Sadece vejetasyon piksellerini al
        veg_pixels_rgb = img_rgb[veg_mask].astype(np.uint8)

        # HSV'ye çevir
        bgr = veg_pixels_rgb[:, [2, 1, 0]].reshape(-1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        H = hsv[:, 0].astype(np.float32)
        V = hsv[:, 2].astype(np.float32)

        # Çok karanlık pikselleri dışla (gölge etkisi)
        bright = V > 40
        H = H[bright]
        if len(H) == 0:
            return dict(
                yellow_green_pct=0.0,
                light_green_pct=0.0,
                mid_green_pct=0.0,
                dark_green_pct=0.0,
                dominant_color="Yetersiz veri",
            )

        n = len(H)

        # 4 renk kategorisi (H değeri aralıkları)
        yellow_green = ((H >= 22) & (H < 38)).sum()  # sarımsı yeşil
        light_green  = ((H >= 38) & (H < 52)).sum()  # açık yeşil
        mid_green    = ((H >= 52) & (H < 68)).sum()  # orta yeşil
        dark_green   = ((H >= 68) & (H < 90)).sum()  # koyu yeşil

        yg_pct = 100 * yellow_green / n
        lg_pct = 100 * light_green  / n
        mg_pct = 100 * mid_green    / n
        dg_pct = 100 * dark_green   / n

        # Baskın ton
        values = {
            "Sarımsı yeşil": yg_pct,
            "Açık yeşil": lg_pct,
            "Orta yeşil": mg_pct,
            "Koyu yeşil": dg_pct,
        }
        dominant = max(values, key=values.get)

        return dict(
            yellow_green_pct=round(yg_pct, 1),
            light_green_pct=round(lg_pct, 1),
            mid_green_pct=round(mg_pct, 1),
            dark_green_pct=round(dg_pct, 1),
            dominant_color=dominant,
        )
    
    # --------------------------------------------------------
    # SAĞLIK SKORU (çiftçi dostu özet)
    # --------------------------------------------------------
    def _compute_health_score(self, veg_pct, exg_mean, exg_std):
        """
        0-100 arası çiftçi dostu sağlık skoru.
        Bileşenler:
          - Vejetasyon yüzdesi (ne kadar yeşil örtü var)
          - ExG ortalaması (yeşilliğin yoğunluğu)
          - Homojenlik (düşük std = tutarlı bahçe)
        """
        # Vejetasyon yüzdesi katkısı (0-50 puan)
        veg_component = min(veg_pct / 100 * 50, 50)

        # ExG yoğunluğu katkısı (0-30 puan), ExG ~ 0-150 aralığında
        exg_component = min(max(exg_mean, 0) / 150 * 30, 30)

        # Homojenlik katkısı (0-20 puan), düşük std = iyi
        # exg_std genelde 20-60 arası, düşük olması homojen demek
        homogeneity_component = max(0, 20 - (exg_std / 60 * 20))

        return veg_component + exg_component + homogeneity_component

    # --------------------------------------------------------
    # ANA ANALİZ FONKSİYONU
    # --------------------------------------------------------
    def analyze(self, image_path, generate_visuals=True, tea_mask=None,
                tea_info=None):
        """
        Bir görüntüyü analiz et.

        Args:
            image_path: görüntü yolu
            generate_visuals: görselleri de üret (toplu modda False yap)
            tea_mask: (H, W) bool array - varsa SADECE bu maske içinde analiz yapılır
                      (downscale öncesi orijinal boyutta olmalı)
            tea_info: dict - YOLO bilgileri {n_segments, confidence}
        """
        image_path = str(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Görüntü açılamadı: {image_path}")

        img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb_full.shape[:2]

        # Hız için küçült
        if self.downscale != 1.0:
            img_rgb = cv2.resize(img_rgb_full, None,
                                 fx=self.downscale, fy=self.downscale,
                                 interpolation=cv2.INTER_AREA)
        else:
            img_rgb = img_rgb_full

        h, w = img_rgb.shape[:2]
        total_px = h * w

        # Çay maskesini downscale boyutuna getir (varsa)
        tea_mask_scaled = None
        tea_coverage = 0.0
        if tea_mask is not None:
            if tea_mask.shape != (h, w):
                tea_mask_scaled = cv2.resize(
                    tea_mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                tea_mask_scaled = tea_mask.astype(bool)
            tea_coverage = 100 * tea_mask_scaled.sum() / total_px

        # RF segmentasyonu
        X = self._extract_features(img_rgb)
        preds = self.model.predict(X)
        proba = self.model.predict_proba(X).max(axis=1)

        diger_mask = (proba < self.diger_threshold).reshape(h, w)
        class_map = preds.reshape(h, w)
        veg_mask = (class_map == 0) & ~diger_mask
        shadow_mask = (class_map == 1) & ~diger_mask

        # Çay maskesi varsa: yüzdelerin paydasını çay alanı yap
        if tea_mask_scaled is not None and tea_mask_scaled.any():
            # Sadece çay alanı içindeki pikseller sayılır
            veg_in_tea = veg_mask & tea_mask_scaled
            shadow_in_tea = shadow_mask & tea_mask_scaled
            diger_in_tea = diger_mask & tea_mask_scaled
            tea_total = tea_mask_scaled.sum()

            veg_pct = 100 * veg_in_tea.sum() / tea_total
            shadow_pct = 100 * shadow_in_tea.sum() / tea_total
            other_pct = 100 * diger_in_tea.sum() / tea_total

            # Maskeleri tema göstermek için güncelle: çay dışı her şey "diğer" değil
            # ama görsel için ayrı tutalım
            analysis_veg_mask = veg_in_tea
            analysis_shadow_mask = shadow_in_tea
            analysis_other_mask = diger_in_tea
            indices_mask = veg_in_tea  # indeksler çay içindeki vejetasyondan
            mode = "tea_only"
        else:
            # Tüm görüntü
            veg_pct = 100 * veg_mask.sum() / total_px
            shadow_pct = 100 * shadow_mask.sum() / total_px
            other_pct = 100 * diger_mask.sum() / total_px
            analysis_veg_mask = veg_mask
            analysis_shadow_mask = shadow_mask
            analysis_other_mask = diger_mask
            indices_mask = veg_mask
            mode = "full_image"

        indices = self._compute_indices(img_rgb, indices_mask)
        hue_dist = self._compute_hue_distribution(img_rgb, indices_mask)

        result = AnalysisResult(
            filename=Path(image_path).name,
            image_width=orig_w,
            image_height=orig_h,
            vegetation_pct=veg_pct,
            shadow_pct=shadow_pct,
            other_pct=other_pct,
            exg_mean=indices["exg_mean"],
            gli_mean=indices["gli_mean"],
            vari_mean=indices["vari_mean"],
            exg_std=indices["exg_std"],
            yellow_green_pct=hue_dist["yellow_green_pct"],
            light_green_pct=hue_dist["light_green_pct"],
            mid_green_pct=hue_dist["mid_green_pct"],
            dark_green_pct=hue_dist["dark_green_pct"],
            dominant_color=hue_dist["dominant_color"],
            tea_coverage_pct=tea_coverage,
            analysis_mode=mode,
        )

        if tea_info:
            result.n_tea_segments = int(tea_info.get("n_segments", 0))
            result.tea_confidence = float(tea_info.get("confidence", 0.0))

        # Görseller
        if generate_visuals:
            seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)

            if tea_mask_scaled is not None:
                # Çay dışı bölgeyi soluk göster
                non_tea = ~tea_mask_scaled
                seg_rgb[non_tea] = (40, 40, 40)  # koyu gri

            seg_rgb[analysis_veg_mask] = self.COLOR_VEGETATION
            seg_rgb[analysis_shadow_mask] = self.COLOR_SHADOW
            seg_rgb[analysis_other_mask] = self.COLOR_OTHER

            overlay = cv2.addWeighted(img_rgb, 0.5, seg_rgb, 0.5, 0)

            result.original_rgb = img_rgb
            result.segmentation_rgb = seg_rgb
            result.overlay_rgb = overlay
            if tea_mask_scaled is not None:
                result.tea_mask = tea_mask_scaled

        return result


# ============================================================
# TEST (modülü doğrudan çalıştırınca)
# ============================================================
if __name__ == "__main__":
    from tea_detector import TeaDetector

    MODEL_RF = r"C:\Users\berhan\Desktop\tez\TeaAnalyzer\models\rf_model_binary.pkl"
    MODEL_YOLO = r"C:\Users\berhan\Desktop\tez\TeaAnalyzer\models\tea_yolo.pt"
    TEST_IMG = r"C:\Users\berhan\Desktop\tez\TeaAnalyzer\4.png"

    print("Analiz motoru + YOLO test ediliyor...\n")

    # 1. YOLO ile çay maskesi
    detector = TeaDetector(MODEL_YOLO)
    img_bgr = cv2.imread(TEST_IMG)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tea_mask, n, conf = detector.detect_tea_mask(img_rgb)
    print(f"YOLO: {n} çay segmenti, güven={conf:.3f}")

    # 2. RF analizi (çay maskesiyle)
    analyzer = TeaAnalyzer(MODEL_RF)
    result = analyzer.analyze(
        TEST_IMG,
        tea_mask=tea_mask,
        tea_info={"n_segments": n, "confidence": conf}
    )

    print(f"\nDosya: {result.filename}")
    print(f"Analiz modu: {result.analysis_mode}")
    print(f"Çay kapsamı (görüntünün %?'si çay): %{result.tea_coverage_pct:.1f}")
    print(f"Çay alanı içinde:")
    print(f"  Yeşil örtü: %{result.vegetation_pct:.1f}")
    print(f"  Gölge: %{result.shadow_pct:.1f}")
    print(f"  Diğer: %{result.other_pct:.1f}")
    print(f"Sağlık skoru: {result.health_score:.1f}/100")
    print(f"ExG: {result.exg_mean:.1f}")

    # 3. Karşılaştırma: çay maskesi olmadan
    print("\n--- Karşılaştırma: çay maskesi OLMADAN ---")
    result2 = analyzer.analyze(TEST_IMG)
    print(f"Tüm görüntü: yeşil=%{result2.vegetation_pct:.1f}, "
          f"gölge=%{result2.shadow_pct:.1f}, "
          f"sağlık={result2.health_score:.1f}")