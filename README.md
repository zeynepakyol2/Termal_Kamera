<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

## Proje Hakkında

Bu repo, termal kamera görüntüleri üzerinden **yara tespiti, yara evresi ve risk seviyesi tahmini** yapan derin öğrenme modellerinin eğitim, değerlendirme ve görselleştirme sürecini içerir. TensorFlow ve MobileNetV2 mimarisi kullanılarak üç ayrı model geliştirilmiş, Transfer Learning ve Fine-Tuning teknikleriyle optimize edilmiştir.

> Mobil repo linki ---> https://github.com/zeynepakyol2/Termal-kamera-mobil

## Model Mimarisi

| Model | Görev | Yaklaşım |
|---|---|---|
| Model 1 | Yara Var / Yok Sınıflandırması | Transfer Learning + Fine-Tuning (MobileNetV2) |
| Model 2 | Yara Evresi Tahmini | Transfer Learning + Fine-Tuning (MobileNetV2) |
| Model 3 | Risk Seviyesi Tahmini (yara yoksa) | Transfer Learning + Fine-Tuning (MobileNetV2) |

Veri seti, model eğitiminden önce çeşitli ön işleme adımlarından geçirilmiş; ardından her üç model de Transfer Learning ve Fine-Tuning teknikleriyle eğitilip optimize edilmiştir.

Veri seti hasta mahremiyeti nedeniyle repoya eklenmemiştir.

## Kullanılan Teknolojiler

- **Dil:** Python
- **Derin Öğrenme:** TensorFlow, Keras
- **Görüntü İşleme:** OpenCV
- **Veri Analizi:** Pandas, NumPy
- **Görselleştirme:** Matplotlib, Seaborn

## Değerlendirme Metrikleri

Model performansları aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Eğitim ve değerlendirme süreci Matplotlib ve Seaborn kullanılarak görselleştirilmiştir.

## Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/zeynepakyol2/Termal_Kamera.git
cd Termal_Kamera

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```





<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:6A5ACD,100:FF6B6B&height=100&section=footer"/>
</p>
