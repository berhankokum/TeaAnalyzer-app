"""
Vejetasyon İndeksi Gösterim Yardımcıları
========================================
Ham indeks değerlerini (ExG, GLI, VARI) son kullanıcı için anlamlı
hale getiren ORTAK mantık. Hem arayüz (ui/single_analysis.py) hem de
PDF raporu (core/report.py) bu modülü kullanır; böylece eşikler ve
durum renkleri tek bir yerden yönetilir.

Burada YENİ bir "tek sağlık skoru" üretilmez; sadece üç indeks ayrı
ayrı normalize edilip (Düşük / Orta / Yüksek) olarak yorumlanır.
"""

# --------------------------------------------------------------------------
# DURUM RENKLERİ (mevcut rapor/arayüz paletiyle uyumlu)
# --------------------------------------------------------------------------
COLOR_LOW = "#d84315"    # kırmızı  -> Düşük
COLOR_MID = "#f9a825"    # sarı     -> Orta
COLOR_HIGH = "#2e7d32"   # yeşil    -> Yüksek

# --------------------------------------------------------------------------
# İNDEKS YAPILANDIRMASI
# --------------------------------------------------------------------------
# DİKKAT: low/high eşikleri BAŞLANGIÇ (kalibre edilebilir) değerlerdir.
# RGB tabanlı vejetasyon indeksleri aydınlatma/pozlama koşullarına
# duyarlı olduğundan, bu eşikler kendi veri setinizden kalibre edilmelidir.
#   - min/max : normalize() için ölçek sınırları (indeksin teorik aralığı)
#   - low     : bu değerin ALTI  -> "Düşük"
#   - high    : bu değer ve ÜSTÜ -> "Yüksek"  (low ile high arası -> "Orta")
INDEX_CONFIG = {
    "ExG":  {"label": "Canlılık (yeşillik yoğunluğu)", "min": 0.0,  "max": 255.0, "low": 60.0, "high": 130.0},
    "GLI":  {"label": "Yeşil örtü indeksi",            "min": -1.0, "max": 1.0,   "low": 0.2,  "high": 0.4},
    "VARI": {"label": "Örtü yoğunluğu",                "min": -1.0, "max": 1.0,   "low": 0.2,  "high": 0.4},
}

# Ham değerin kaç ondalıkla gösterileceği (ExG tam sayı, diğerleri 3 ondalık)
INDEX_DECIMALS = {"ExG": 0, "GLI": 3, "VARI": 3}


def format_value(name, value):
    """Ham indeks değerini tipine göre biçimlendir (ExG tam sayı, diğerleri 3 ondalık)."""
    decimals = INDEX_DECIMALS.get(name, 3)
    return f"{value:.{decimals}f}"


def normalize(name, value):
    """İndeks değerini 0–100 arası yüzdeye dönüştür: clamp((v-min)/(max-min)*100, 0, 100)."""
    cfg = INDEX_CONFIG[name]
    lo, hi = cfg["min"], cfg["max"]
    if hi <= lo:
        return 0.0
    pct = (value - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, pct))


def status(name, value):
    """
    İndeks değerine göre (etiket, renk) döndür:
        value < low          -> ("Düşük",  kırmızı)
        low <= value < high  -> ("Orta",   sarı)
        value >= high        -> ("Yüksek", yeşil)
    """
    cfg = INDEX_CONFIG[name]
    if value < cfg["low"]:
        return ("Düşük", COLOR_LOW)
    elif value < cfg["high"]:
        return ("Orta", COLOR_MID)
    return ("Yüksek", COLOR_HIGH)


def index_rows(result):
    """
    AnalysisResult'tan ExG/GLI/VARI için gösterime hazır satırlar üretir.
    Her satır bir dict:
        name        -> "ExG" | "GLI" | "VARI"
        label       -> kullanıcı dostu açıklama
        value       -> ham değer (float)
        value_str   -> biçimlenmiş ham değer
        norm        -> 0–100 normalize yüzde (çubuk uzunluğu için)
        status      -> "Düşük" | "Orta" | "Yüksek"
        color       -> durum rengi (hex)
    """
    mapping = [
        ("ExG", result.exg_mean),
        ("GLI", result.gli_mean),
        ("VARI", result.vari_mean),
    ]
    rows = []
    for name, value in mapping:
        cfg = INDEX_CONFIG[name]
        st_label, st_color = status(name, value)
        rows.append({
            "name": name,
            "label": cfg["label"],
            "value": value,
            "value_str": format_value(name, value),
            "norm": normalize(name, value),
            "status": st_label,
            "color": st_color,
        })
    return rows
