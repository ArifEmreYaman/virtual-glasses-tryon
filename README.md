# 👓 Virtual Glasses Try-On (AR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![Open3D](https://img.shields.io/badge/Open3D-3D%20Rendering-red)

## 📖 Proje Hakkında
Bu uygulama, **Python**, **OpenCV** ve **MediaPipe** teknolojilerini kullanarak geliştirilmiş, gerçek zamanlı bir **Sanal Gözlük Deneme (Virtual Try-On)** sistemidir. Proje, yüzdeki 468 landmark noktasını analiz ederek 3D objeleri perspektife uygun şekilde yerleştirir.

### Öne Çıkan Mühendislik Çözümleri:
* **Temporal Smoothing:** Baş hareketlerindeki titremeleri önleyen özel stabilizasyon filtresi.
* **Adaptive Scaling:** Kullanıcının yüz genişliği ve göz mesafesine göre dinamik olarak boyutlandırılan 3D modeller.
* **6DOF Takip:** `solvePnP` algoritması ile hassas rotasyon ve pozisyon kestirimi.

## 🛠️ Kurulum
1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/ArifEmreYaman/virtual-glasses-tryon.git](https://github.com/ArifEmreYaman/virtual-glasses-tryon.git)
   cd virtual-glasses-tryon

Gerekli kütüphaneleri yükleyin:
pip install -r requirements.txt

💻 Kullanım
Uygulamayı başlatmak için:
python main.py

  Tuşlar: 1-4 (Model Değiştir), S (Smoothing Aç/Kapat), D (Debug Mod), ESC (Çıkış).

🧪 Diagnostik Testler
Sistemi doğrulamak için tests/ klasöründeki araçları kullanabilirsiniz:

  Kamera Testi: python tests/test_box.py

  Model Testi: python tests/test_obj_loader.py
