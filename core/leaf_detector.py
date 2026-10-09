"""Yaprak tespit, hastalık ve yaş analizi pipeline'ı."""

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO


@dataclass
class LeafResult:
    leaf_id: int
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2)
    detection_confidence: float
    health_label: str
    health_confidence: float
    age_label: str
    age_confidence: float
    crop_rgb: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class LeafAnalysisResult:
    image_path: str
    original_rgb: np.ndarray = field(repr=False)
    annotated_rgb: np.ndarray = field(repr=False)
    leaves: List[LeafResult] = field(default_factory=list)

    @property
    def total_leaves(self) -> int:
        return len(self.leaves)

    @property
    def healthy_count(self) -> int:
        return sum(1 for lf in self.leaves if _is_healthy(lf.health_label))

    @property
    def diseased_count(self) -> int:
        return self.total_leaves - self.healthy_count


def _is_healthy(label: str) -> bool:
    return any(kw in label for kw in ("Sağlıklı", "Healthy", "healthy", "Normal", "normal"))


class LeafDetector:
    """
    Üç aşamalı yaprak analizi:
      1. tea-leaf-detection-model.pt  — YOLO ile yaprak tespiti
      2. best_tealeaf_model.pth      — EfficientNet-B0 (7 sınıf) hastalık sınıflaması (.h5 de desteklenir)
      3. tea-leaf-age.pt              — YOLO ile yaş/olgunluk sınıflaması
    """

    _DEFAULT_HEALTH_LABELS = ["Sağlıklı", "Hastalıklı"]

    def __init__(self, detection_path, health_path, age_path, detection_conf=0.1):
        self.detection_conf = detection_conf
        self.detector = YOLO(str(detection_path))
        self.age_model = YOLO(str(age_path))
        self.health_backend = None  # "torch" | "keras"
        self.health_model, self.health_classes = self._load_health_model(str(health_path))
        # Tespit edilen her hastalık türüne kalıcı bir ID atanır (ilk görülme sırasına göre).
        # Aynı detektör örneği tüm fotoğrafları analiz ettiğinden ID'ler fotoğraflar arası tutarlıdır.
        self._disease_ids: dict = {}
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------
    # Model yükleme
    # ------------------------------------------------------------------

    def _load_health_model(self, path: str):
        # Keras/TensorFlow modeli (.h5 / .keras)
        if path.lower().endswith((".h5", ".keras")):
            return self._load_keras_health_model(path)

        # PyTorch modeli (.pt / .pth)
        self.health_backend = "torch"
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[LeafDetector] Health model yüklenemedi: {e}")
            return None, None

        # Tam model nesnesi
        if isinstance(data, torch.nn.Module):
            data.eval()
            return data, None

        if not isinstance(data, dict):
            return None, None

        # Checkpoint dict — 'model' anahtarı
        if "model" in data and isinstance(data["model"], torch.nn.Module):
            model = data["model"]
            model.eval()
            classes = data.get("class_names", data.get("names", data.get("classes")))
            return model, classes

        # State-dict çözümleme
        sd_key = next((k for k in ("state_dict", "model_state_dict", "net") if k in data), None)
        sd = data[sd_key] if sd_key else data
        classes = data.get("class_names", data.get("names", data.get("classes")))

        arch = str(data.get("arch", "")).lower()
        num_classes = int(data.get("num_classes") or self._infer_num_classes(sd))

        # Mimari checkpoint'te belirtilmişse direkt onu kur; yoksa sırayla dene.
        if "efficientnet" in arch:
            builders = [self._build_efficientnet]
        elif "resnet" in arch:
            builders = [self._build_resnet50]
        else:
            builders = [self._build_resnet50, self._build_efficientnet]

        for builder in builders:
            model = builder(num_classes)
            try:
                model.load_state_dict(sd, strict=True)
                model.eval()
                return model, classes
            except Exception:
                continue

        print("[LeafDetector] Health model mimarisi belirlenemedi.")
        return None, None

    def _load_keras_health_model(self, path: str):
        try:
            import tensorflow as tf  # noqa: F401
        except Exception as e:
            print(f"[LeafDetector] TensorFlow yüklü değil, .h5 health modeli yüklenemedi: {e}")
            return None, None
        try:
            from tensorflow.keras.models import load_model
            model = load_model(path, compile=False)
            self.health_backend = "keras"
            return model, None
        except Exception as e:
            print(f"[LeafDetector] Keras health modeli yüklenemedi: {e}")
            return None, None

    @staticmethod
    def _infer_num_classes(sd: dict) -> int:
        for key in ("fc.weight", "classifier.weight", "head.weight",
                    "classifier.1.weight", "classifier.6.weight", "_fc.weight"):
            if key in sd:
                return int(sd[key].shape[0])
        return 2

    @staticmethod
    def _build_resnet50(num_classes: int) -> torch.nn.Module:
        import torchvision.models as models
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m

    @staticmethod
    def _build_efficientnet(num_classes: int) -> torch.nn.Module:
        import torchvision.models as models
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_f, num_classes)
        return m

    # ------------------------------------------------------------------
    # Tek yaprak analizi
    # ------------------------------------------------------------------

    def _analyze_health(self, crop_pil: Image.Image) -> Tuple[str, float]:
        if self.health_model is None:
            return "Model Yok", 0.0
        if self.health_backend == "keras":
            return self._analyze_health_keras(crop_pil)
        try:
            tensor = self._transform(crop_pil).unsqueeze(0)
            with torch.no_grad():
                out = self.health_model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                idx = int(probs.argmax().item())
                conf = float(probs[idx].item())

            if self.health_classes:
                classes = list(self.health_classes) if not isinstance(self.health_classes, list) else self.health_classes
                label = classes[idx] if idx < len(classes) else f"Sınıf {idx}"
            else:
                label = (self._DEFAULT_HEALTH_LABELS[idx]
                         if idx < len(self._DEFAULT_HEALTH_LABELS)
                         else f"Sınıf {idx}")
            return label, conf
        except Exception as e:
            print(f"[LeafDetector] Health analiz hatası: {e}")
            return "Bilinmiyor", 0.0

    def _analyze_health_keras(self, crop_pil: Image.Image) -> Tuple[str, float]:
        try:
            # Modelin beklediği giriş boyutunu oku (None, H, W, C)
            shape = getattr(self.health_model, "input_shape", None)
            h = shape[1] if shape and shape[1] else 224
            w = shape[2] if shape and shape[2] else 224

            img = crop_pil.convert("RGB").resize((w, h))
            arr = np.asarray(img, dtype="float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = self.health_model.predict(arr, verbose=0)[0]
            preds = np.asarray(preds).ravel()

            if preds.shape[0] == 1:
                # Tek çıkışlı sigmoid (ikili sınıflama)
                p = float(preds[0])
                idx = 1 if p >= 0.5 else 0
                conf = p if idx == 1 else 1.0 - p
            else:
                idx = int(np.argmax(preds))
                conf = float(preds[idx])

            labels = list(self.health_classes) if self.health_classes else self._DEFAULT_HEALTH_LABELS
            label = labels[idx] if idx < len(labels) else f"Sınıf {idx}"
            return label, conf
        except Exception as e:
            print(f"[LeafDetector] Keras health analiz hatası: {e}")
            return "Bilinmiyor", 0.0

    def _analyze_age(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        try:
            results = self.age_model(crop_bgr, verbose=False)
            r = results[0]
            # Sınıflama modeli
            if hasattr(r, "probs") and r.probs is not None:
                idx = int(r.probs.top1)
                conf = float(r.probs.top1conf.item())
                label = r.names.get(idx, f"Sınıf {idx}")
                return label, conf
            # Tespit modeli (en yüksek güvene sahip kutu)
            if r.boxes is not None and len(r.boxes) > 0:
                best = max(r.boxes, key=lambda b: float(b.conf[0]))
                idx = int(best.cls[0].item())
                conf = float(best.conf[0].item())
                label = r.names.get(idx, f"Sınıf {idx}")
                return label, conf
        except Exception as e:
            print(f"[LeafDetector] Age analiz hatası: {e}")
        return "Bilinmiyor", 0.0

    def disease_id(self, label: str) -> int:
        """Hastalık etiketine kalıcı bir ID döndürür (ilk görüldüğünde atanır)."""
        if label not in self._disease_ids:
            self._disease_ids[label] = len(self._disease_ids) + 1
        return self._disease_ids[label]

    # ------------------------------------------------------------------
    # Ana analiz
    # ------------------------------------------------------------------

    def analyze(self, image_path: str) -> LeafAnalysisResult:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        det_results = self.detector(str(image_path), conf=self.detection_conf, verbose=False)
        annotated = img_rgb.copy()
        leaves: List[LeafResult] = []

        raw_boxes = det_results[0].boxes if det_results[0].boxes is not None else []

        for box in raw_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0].item())

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = img_rgb[y1:y2, x1:x2]

            leaf_no = len(leaves) + 1
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            crop_pil = Image.fromarray(crop_rgb)

            health_label, health_conf = self._analyze_health(crop_pil)
            age_label, age_conf = self._analyze_age(crop_bgr)

            # Kutu rengi: sağlıklı → yeşil (yazısız), hastalıklı → kırmızı (hastalık ID'si yazılır).
            if _is_healthy(health_label):
                color = (34, 197, 94)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            else:
                color = (220, 53, 53)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                # Kutunun sol üstüne o hastalığın ID numarası yazılır.
                cv2.putText(
                    annotated, f"#{self.disease_id(health_label)}",
                    (x1 + 4, max(y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )

            leaves.append(LeafResult(
                leaf_id=leaf_no,
                bbox=(x1, y1, x2, y2),
                detection_confidence=det_conf,
                health_label=health_label,
                health_confidence=health_conf,
                age_label=age_label,
                age_confidence=age_conf,
                crop_rgb=crop_rgb,
            ))

        return LeafAnalysisResult(
            image_path=str(image_path),
            original_rgb=img_rgb,
            annotated_rgb=annotated,
            leaves=leaves,
        )
