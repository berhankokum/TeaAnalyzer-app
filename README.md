# TeaAnalyzer

Drone görüntülerinde çay alanı analizi ve yakın çekim fotoğraflarda yaprak hastalık/yaş tahmini için Python ve PyQt6 ile geliştirilmiş masaüstü uygulaması.

**Model ağırlıkları bu depoya dahil değildir.** Projede kullanılan modeller Muhammed Berhan Kokum tarafından eğitilmiştir. Bu depo, görüntü işleme, model çıkarımı, masaüstü arayüz ve raporlama kaynak kodlarını içerir. Eğitim kodu ve eğitim veri setleri depoda bulunmaz.

> Kaynak kodunu incelemek için modellere gerek yoktur. Analiz çalıştırmak için aşağıda belirtilen uyumlu model dosyalarının yerel olarak sağlanması gerekir. Mevcut uygulama, RF modeli bulunmadığında başlangıçta uyarı vererek kapanır; model gerektirmeyen bir demo modu yoktur.

## Özellikler

- **Alan analizi:** YOLO11 segmentasyon modeliyle çay alanlarının tespiti ve Random Forest ile piksel tabanlı analiz.
- **Görüntü göstergeleri:** ExG, GLI ve VARI indeksleri ile HSV tabanlı renk tonu dağılımı.
- **Yaprak analizi:** YOLO ile yaprak tespiti, EfficientNet-B0 ile yedi sınıflı hastalık sınıflandırması ve ayrı bir YOLO modeliyle yaş aralığı tahmini.
- **Masaüstü arayüz:** Tek görüntü ve klasör bazlı toplu alan analizi; birden fazla yaprak fotoğrafı için sınıf dağılımları ve işaretlenmiş görüntüler.
- **Arka plan analizi:** QThread ile analiz işlerinin arayüz iş parçacığından ayrı yürütülmesi.
- **Dışa aktarım:** Tekli/toplu alan analizi için PDF, toplu alan analizi için Excel çıktısı.

Yaprak analiz ekranındaki özet için PDF/Excel dışa aktarımı bulunmaz. Uygulama kayıtlı görüntü dosyalarıyla çalışır; canlı kamera veya drone bağlantısı içermez.

## Analiz akışı

```text
Alan görüntüsü
  -> YOLO çay segmentasyonu
  -> Çay maskesi + Random Forest piksel analizi
  -> Yüzdeler, vejetasyon indeksleri ve renk dağılımı
  -> Arayüz / PDF / Excel

Yaprak fotoğrafları
  -> YOLO yaprak tespiti
  -> Yaprak kırpmaları
  -> EfficientNet-B0 hastalık tahmini + YOLO yaş tahmini
  -> İşaretlenmiş görüntüler ve toplu sınıf dağılımı
```

Çay maskesi elde edilemediğinde alan akışı tüm görüntü analizine dönebilir. Bu durumda sonuçlar yalnızca çay alanını temsil etmez.

## Kurulum

Python 3.12 ile ayrı bir sanal ortam oluşturulması önerilir. Python 3.11 altı sürümler, seçilen scikit-learn sürümü için uygun değildir. Arayüz ve PDF font yolları Windows odaklıdır; diğer işletim sistemlerinde tam çalışma doğrulanmamıştır.

### Windows PowerShell

```powershell
git clone https://github.com/berhankokum/TeaAnalyzer-app.git
cd TeaAnalyzer-app
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.12 kurulu olmalıdır. Sanal ortamın Python yolunu doğrudan kullanmak, PowerShell aktivasyon betiğine ihtiyaç bırakmaz.

`requirements.txt`, uygulamanın doğrudan bağımlılıklarını listeler; tamamen kilitlenmiş veya uçtan uca doğrulanmış bir eğitim ortamı değildir. scikit-learn 1.8.0 ve Ultralytics 8.3.240 seçimleri mevcut model kayıtlarından alınmıştır. Yaprak tespit modeli Ultralytics 8.2.38 ile kaydedildiği için modeller sağlandıktan sonra ortak ortamda yükleme/çıkarım uyumluluğu ayrıca kontrol edilmelidir.

PyTorch ve torchvision birlikte uyumlu sürümlerle kurulmalıdır. CUDA kullanmak isteniyorsa donanıma uygun kurulum için [PyTorch kurulum sayfası](https://pytorch.org/get-started/locally/) kullanılabilir. GPU zorunlu değildir; çıkarım hızı donanıma bağlıdır.

scikit-learn modelinin kayıt sürümü korunmalıdır; farklı sürümler arasında model yükleme uyumluluğu garanti edilmez. Bkz. [scikit-learn model saklama belgesi](https://scikit-learn.org/stable/model_persistence.html).

## Model dosyaları

Model dosyalarını aşağıdaki adlarla `models/` klasörüne yerleştirin:

| Dosya | Görev | Gereklilik |
|---|---|---|
| `rf_model_binary.pkl` | Random Forest piksel sınıflandırması | Mevcut uygulamanın açılması ve alan analizi için gerekli |
| `tea_yolo.pt` | YOLO11 çay alanı segmentasyonu | Çay alanına göre maskeleme için gerekli; yoksa tüm görüntü analizine dönülür |
| `tea-leaf-detection-model.pt` | YOLO yaprak tespiti | Yaprak analizi için gerekli |
| `best_tealeaf_model.pth` | EfficientNet-B0 hastalık sınıflandırması | Yaprak analizi için gerekli |
| `tea-leaf-age.pt` | YOLO yaş aralığı tespiti | Yaprak analizi için gerekli |

```text
models/
  README.md
  rf_model_binary.pkl             # Yerel, Git dışında
  tea_yolo.pt                     # Yerel, Git dışında
  tea-leaf-detection-model.pt      # Yerel, Git dışında
  best_tealeaf_model.pth           # Yerel, Git dışında
  tea-leaf-age.pt                 # Yerel, Git dışında
```

**Bu depoda model indirme bağlantısı veya otomatik model temini bulunmaz.** Genel amaçlı YOLO ağırlıklarını yeniden adlandırmak bu özel modellerin yerini tutmaz. Giriş özellikleri, mimari ve sınıf sıralaması uygulamayla uyuşmalıdır. Model sözleşmeleri için [models/README.md](models/README.md) dosyasına bakın.

## Çalıştırma ve kullanım

Modelleri yerleştirdikten sonra proje kökünden:

```powershell
.\.venv\Scripts\python.exe main.py
```

1. Ana menüden **Alan Tabanlı Analiz** veya **Yaprak Tabanlı Analiz** seçin.
2. Alan analizinde tek bir görüntü seçin veya toplu analiz sekmesinden bir klasör belirleyin. Klasör taraması alt klasörleri kapsamaz.
3. Yaprak analizinde en az üç fotoğraf seçin. Fotoğraflar arasında geçiş yaparak tespit sonuçlarını ve sınıf dağılımlarını inceleyin.
4. Alan analizi sonuçlarını ilgili PDF veya Excel düğmesiyle kaydedin.

Arayüzde JPG, JPEG ve PNG dosyaları desteklenir. Analiz verileri kalıcı bir veritabanına yazılmaz; raporlar kullanıcı tarafından seçilen konuma kaydedilir.

Bağımsız hastalık sınıflandırma aracı da kullanılabilir. Girdi ve model yolunu açıkça belirtin:

```powershell
.\.venv\Scripts\python.exe -m core.predict_tealeaf .\foto.jpg --model .\models\best_tealeaf_model.pth --topk 3
```

Bu komut verilen görüntüyü doğrudan sınıflandırır; yaprak tespit/kırpma ve yaş tahmini zincirini çalıştırmaz. Örnekteki `foto.jpg` depoya dahil değildir; kendi görüntünüzün yolunu kullanın.

## Proje yapısı

```text
TeaAnalyzer-app/
  main.py                  # Uygulama ve pencere geçişleri
  core/
    analyzer.py            # RF analizi, indeksler ve renk dağılımı
    pipeline.py            # YOLO ve RF akışının birleştirilmesi
    tea_detector.py        # Çay segmentasyonu
    leaf_detector.py       # Yaprak, hastalık ve yaş analizi
    predict_tealeaf.py     # Komut satırı hastalık sınıflandırıcısı
    index_display.py       # Ortak indeks gösterim mantığı
    report.py              # PDF rapor üretimi
  ui/                      # PyQt6 ekranları ve analiz iş parçacıkları
  models/README.md         # Model dosyası sözleşmeleri
  requirements.txt
  .gitignore
  .gitattributes
```

## Mevcut sınırlar

- Bağımsız etiketli test veri seti ve yayımlanmış uçtan uca doğruluk ölçümü bu depoda bulunmaz. Model tahminleri ve renk indeksleri kesin hastalık teşhisi olarak değerlendirilmemelidir.
- Model yükleme/çıkarım hatalarında dönen `Model Yok` veya `Bilinmiyor` etiketleri mevcut yaprak özetinde hastalıklı sayısına dahil olabilir. Böyle bir durumda özeti geçerli hastalık oranı olarak kullanmayın.
- Alan akışı çay maskesi yerine tüm görüntüye döndüğünde bazı PDF açıklamaları analiz kapsamını doğru yansıtmayabilir.
- RGB indeksleri ışık ve pozlamadan etkilenir; gösterim eşikleri başlangıç değerleridir. Kodda hesaplanan `Diğer` oranı düşük RF güvenini gösterir, doğrulanmış toprak/yol sınıfı değildir.
- Toplu dosya hataları, uzun PDF içerikleri ve çalışan analiz sırasında kapanış için ek iyileştirmeler gerekir.
- Otomatik uygulama test paketi ve CI yapılandırması bulunmaz. `core/analyzer.py` ve `core/tea_detector.py` sonundaki eski manuel deneme blokları yerel yollara bağlıdır; belgelenen kullanım `main.py` ve yukarıdaki açık argümanlı CLI komutudur.

## Kullanılan teknolojiler

Python, PyQt6, OpenCV, NumPy, scikit-learn, Ultralytics YOLO, PyTorch, torchvision, Pillow, pandas, openpyxl ve ReportLab.

Varsayılan yaprak hastalık akışı PyTorch kullanır. Kodda alternatif Keras yükleme desteği vardır, ancak `.h5/.keras` modelleri bu projedeki varsayılan dosya setinin parçası değildir ve TensorFlow temel bağımlılıklara eklenmemiştir.
