import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ==============================
# AYARLAR
# ==============================
script_dir = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(script_dir, "best_tealeaf_model.pth")

# Argüman verilmezse kullanılacak varsayılan görsel
default_input = r"C:\Users\genca\OneDrive\Desktop\test.jpg"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes):
    """Eğitimdeki ile aynı mimariyi kur (EfficientNet-B0)."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model_path):
    """Metadata'lı checkpoint'i yükle; eski format (sadece state_dict) ile de uyumlu."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint["class_names"]
        image_size = checkpoint.get("image_size", 224)
        state_dict = checkpoint["model_state_dict"]
        print(f"Model yüklendi (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc {checkpoint.get('val_acc', 0):.4f})")
    else:
        # Eski format: yalnızca state_dict kaydedilmiş; sınıf isimleri yok.
        state_dict = checkpoint
        # Son katmanın çıkış boyutundan sınıf sayısını çıkar
        num_classes = state_dict["classifier.1.weight"].shape[0]
        class_names = [f"sinif_{i}" for i in range(num_classes)]
        image_size = 224
        print("UYARI: Eski format model (sınıf isimleri yok), genel isimler kullanılıyor.")

    model = build_model(len(class_names))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, class_names, image_size


def get_transform(image_size):
    """Eğitimdeki valid/test ile AYNI ön işleme."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict_image(model, transform, class_names, image_path, topk=3):
    """Tek bir görseli sınıflandır, ilk topk tahmini döndür."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    k = min(topk, len(class_names))
    top_probs, top_idxs = torch.topk(probs, k)
    results = [(class_names[i], p.item()) for p, i in zip(top_probs, top_idxs)]
    return results


def collect_images(path):
    """Verilen yol bir dosyaysa onu, klasörse içindeki tüm görselleri döndür."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(path, f))
        return files
    raise FileNotFoundError(f"Yol bulunamadı: {path}")


def main():
    parser = argparse.ArgumentParser(description="Tea Leaf hastalık tahmini")
    parser.add_argument("input", nargs="?", default=default_input,
                        help="Görsel dosyası veya görsel klasörü yolu "
                             "(verilmezse varsayılan test.jpg kullanılır)")
    parser.add_argument("--model", default=default_model_path, help="Model (.pth) yolu")
    parser.add_argument("--topk", type=int, default=3, help="Gösterilecek tahmin sayısı")
    args = parser.parse_args()

    print(f"Cihaz: {device}")
    model, class_names, image_size = load_checkpoint(args.model)
    transform = get_transform(image_size)

    images = collect_images(args.input)
    if not images:
        print("Görsel bulunamadı.")
        return

    print(f"\n{len(images)} görsel işleniyor...\n")
    print("=" * 60)

    for img_path in images:
        try:
            results = predict_image(model, transform, class_names, img_path, args.topk)
        except Exception as e:
            print(f"{os.path.basename(img_path)} -> HATA: {e}")
            continue

        top_name, top_prob = results[0]
        print(f"\n{os.path.basename(img_path)}")
        print(f"  TAHMİN: {top_name}  (%{top_prob * 100:.2f})")
        if len(results) > 1:
            print("  Diğer olasılıklar:")
            for name, prob in results[1:]:
                print(f"    - {name}: %{prob * 100:.2f}")

    print("\n" + "=" * 60)
    print("Tamamlandı.")


if __name__ == "__main__":
    main()
