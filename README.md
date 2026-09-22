<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF6B6B,100:6A5ACD&height=180&section=header&text=Termal%20Kamera%20ile%20Yara%20Tespiti&fontSize=32&fontColor=ffffff&animation=fadeIn&desc=Derin%20%C3%96%C4%9Frenme%20%26%20G%C3%B6r%C3%BCnt%C3%BC%20%C4%B0%C5%9Fleme%20ile%20Yara%20Analiz%20Sistemi&descSize=15&descAlignY=62"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

## 📖 Proje Hakkında

Bu proje, termal kamera görüntüleri üzerinden **yapay zeka destekli yara tespiti ve analizi** yapan uçtan uca bir sistemdir. TensorFlow ve MobileNetV2 mimarisi kullanılarak eğitilen üç ayrı derin öğrenme modeli, bir Kotlin mobil uygulaması üzerinden kullanıcıya sunulmaktadır. Kullanıcı, görüntü üzerinde manuel olarak yara alanı seçimi yaparak yara boyutunu hesaplayabilmekte ve yapay zeka destekli analiz sonuçlarını görüntüleyebilmektedir.

## ✨ Özellikler

- 🔍 **Yara Var/Yok Sınıflandırması** — termal görüntüde yara bulunup bulunmadığını tespit eden ilk model
- 📊 **Yara Evresi Tahmini** — tespit edilen yaranın hangi evrede olduğunu belirleyen ikinci model
- ⚠️ **Risk Seviyesi Tahmini** — yara bulunmayan görüntüler için risk seviyesini öngören üçüncü model
- 📱 **Mobil Entegrasyon** — Kotlin ile geliştirilen mobil uygulama üzerinden manuel yara alanı seçimi ve boyut hesaplama
- 📈 **Detaylı Performans Değerlendirmesi** — Accuracy, Precision, Recall, F1-Score ve Confusion Matrix metrikleri

## 🧠 Model Mimarisi

| Model | Görev | Yaklaşım |
|---|---|---|
| Model 1 | Yara Var / Yok Sınıflandırması | Transfer Learning + Fine-Tuning (MobileNetV2) |
| Model 2 | Yara Evresi Tahmini | Transfer Learning + Fine-Tuning (MobileNetV2) |
| Model 3 | Risk Seviyesi Tahmini (yara yoksa) | Transfer Learning + Fine-Tuning (MobileNetV2) |

Veri kümesi, model eğitiminden önce çeşitli ön işleme adımlarından geçirilmiş; ardından her üç model de Transfer Learning ve Fine-Tuning teknikleriyle eğitilip optimize edilmiştir.

## 🛠️ Kullanılan Teknolojiler

**Model Eğitimi & Analiz**
- Python, TensorFlow, Keras
- Pandas, NumPy
- Matplotlib, Seaborn (görselleştirme)

**Mobil Uygulama**
- Kotlin
- Kamera/galeri entegrasyonu, manuel alan seçimi ile ölçüm

## 📊 Değerlendirme Metrikleri

Model performansları aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Eğitim ve değerlendirme süreci Matplotlib ve Seaborn kullanılarak görselleştirilmiştir.

<!--
## 🖼️ Ekran Görüntüleri
Buraya mobil uygulama ve model çıktılarına ait ekran görüntüleri eklenebilir:
<p align="center">
  <img src="screenshots/1.png" width="250"/>
  <img src="screenshots/2.png" width="250"/>
</p>
-->

## 🚀 Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/zeynepakyol2/Termal_Kamera.git
cd Termal_Kamera

# Gerekli Python kütüphanelerini yükleyin
pip install -r requirements.txt

# Model eğitim/tahmin scriptini çalıştırın
python train_model.py
```

> Mobil uygulama tarafı için Kotlin projesini Android Studio ile açıp derleyebilirsiniz.

## 📌 Kullanım

1. Termal görüntüyü sisteme yükleyin
2. Model 1, görüntüde yara olup olmadığını tespit eder
3. Yara tespit edilirse Model 2 yara evresini tahmin eder
4. Yara tespit edilmezse Model 3 risk seviyesini öngörür
5. Mobil uygulamada, kullanıcı görüntü üzerinde manuel alan seçimi yaparak yara boyutunu ölçebilir

## 👩‍💻 Geliştirici

**Zeynep Akyol**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zeynepakyol-a074982b8/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/zeynepakyol2)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=vercel&logoColor=white)](https://www.zeynepakyol.com/)

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6A5ACD,100:FF6B6B&height=100&section=footer"/>
</p>
