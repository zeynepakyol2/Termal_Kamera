import tensorflow as tf

# 1. Modelin hata vermemesi için özel fonksiyonları buraya da ekliyoruz
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# 2. Dosya yollarını direkt kodun içine yazıyoruz (Senin istediğin gibi)
keras_path = "models/wound_model.keras"
tflite_path = "models/wound_model.tflite"


# 3. Modeli yükle
print("Model yükleniyor...")
model = tf.keras.models.load_model(
    keras_path,
    custom_objects={
        "dice_loss": dice_loss,
        "dice_coef": dice_coef
    }
)

# 4. TFLite'a dönüştür ve Android için optimize et
print("TFLite formatına dönüştürülüyor ve optimize ediliyor...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 5. Dosyayı kaydet
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Harika! TFLite modeli başarıyla oluşturuldu: {tflite_path}")
print(f"Model boyutu: {len(tflite_model) / 1024:.2f} KB")