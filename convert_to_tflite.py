import tensorflow as tf
import os
import sys


h5_model_path = "model_files/Basınç_yarası_evreleri_1_best.h5"  
tflite_output_path = "model_files/Basınç_yarası_evreleri_1_converted_model.tflite"

# Klasör yoksa oluştur
os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)

print("TensorFlow sürümü:", tf.__version__)
print("Model yükleniyor:", h5_model_path)

try:
    model = tf.keras.models.load_model(h5_model_path)
    print("H5 modeli başarıyla yüklendi.")
except Exception as e:
    print("Model yüklenirken hata:", e)
    sys.exit(1)

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    print("Dönüşüm işlemi tamamlandı (bellekte oluşturuldu).")
except Exception as e:
    print("Dönüştürme sırasında hata:", e)
    sys.exit(1)

try:
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite modeli başarıyla kaydedildi: {tflite_output_path}")
    print(f"Boyut: {len(tflite_model) / 1024:.2f} KB")
except Exception as e:
    print("Model kaydedilemedi:", e)
