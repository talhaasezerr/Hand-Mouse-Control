# Hand-Mouse-Control

# Hand Mouse Control 🖐️🖱️

El hareketleriyle fare kontrolü sağlayan Python projesi. MediaPipe ve OpenCV kullanarak gerçek zamanlı el takibi yaparak bilgisayarınızı ellerinizle kontrol edebilirsiniz.

## 🌟 Özellikler

- 👆 İşaret parmağı ile fare imleci kontrolü
- 👍 Başparmak hareketi ile sol tıklama
- 🎯 Çift tıklama desteği
- 🖱️ Sürükle-bırak (drag & drop) özelliği
- 🎚️ One Euro Filter ile yumuşak hareket
- ⏸️ Pause/Resume fonksiyonu

## 📋 Gereksinimler

- Python 3.9+ (önerilen) veya Python 3.8 (MediaPipe 0.10.9 ile)
- Webcam
- Windows 10/11

## 🔧 Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/KULLANICI_ADINIZ/hand-mouse-control.git
cd hand-mouse-control
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

**Python 3.8 kullanıyorsanız:**
```bash
pip install mediapipe==0.10.9 opencv-python pyautogui numpy
```

**Python 3.9+ kullanıyorsanız:**
```bash
pip install mediapipe opencv-python pyautogui numpy
```

## 🚀 Kullanım

Programı çalıştırın:
```bash
python hand_mouse_control.py
```

### Kontroller

- **İşaret parmağı**: Fare imlecini hareket ettirin
- **Başparmağı bükmek**: Sol tıklama (kısa süre)
- **Başparmağı bükülü tutmak**: Sürükle-bırak (drag)
- **Hızlı çift katlama**: Çift tıklama
- **Q tuşu**: Programdan çık
- **P tuşu**: Duraklatma/Devam ettirme

## ⚙️ Ayarlar

[hand_mouse_control.py](hand_mouse_control.py) dosyasında yapılabilecek ayarlar:

```python
CAM_INDEX = 0              # Kamera indeksi
FRAME_W, FRAME_H = 960, 540  # Kamera çözünürlüğü
MARGIN = 0.12              # Aktif alan kenar boşluğu
MAX_STEP_PX = 25           # Maksimum hareket hızı
THUMB_FOLD_ON = 115        # Başparmak katlama açı eşiği
```

## 📊 Nasıl Çalışır?

1. **El Algılama**: MediaPipe ile gerçek zamanlı el landmark'ları tespit edilir
2. **Parametre Okuma**: İşaret parmağı pozisyonu fare koordinatına dönüştürülür
3. **Filtreleme**: One Euro Filter ile hareket stabilize edilir
4. **Jest Tanıma**: Başparmak açısı ile tıklama/sürükleme algılanır
5. **Fare Kontrolü**: PyAutoGUI ile sistem fare kontrolü sağlanır

## 🐛 Bilinen Sorunlar

- Python 3.8 ile MediaPipe 0.10.32+ uyumsuzluk (`'type' object is not subscriptable` hatası)
  - **Çözüm**: MediaPipe 0.10.9 kullanın veya Python 3.9+ yükseltin

## 📝 Lisans

MIT License

## 👨‍💻 Geliştirici

Geliştiren: TALHA

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için lütfen önce bir issue açarak neyi değiştirmek istediğinizi tartışın.

## ⭐ Beğendiyseniz

Projeyi beğendiyseniz ⭐ vermeyi unutmayın!
