""" Yara var mı yok mu ?"""
# -------------------- KÜTÜPHANELER --------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model

# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224)        # Model giriş boyutu
BATCH_SIZE = 32              # Veri işleme boyutu
INITIAL_EPOCHS = 150         # Maksimum eğitim iterasyonu
LEARNING_RATE = 0.0001       # Başlangıç öğrenme oranı
MIN_DELTA = 0.00001          # Erken durdurma hassasiyeti
PATIENCE = 30                # Erken durdurma sabrı

# -------------------- VERİ YÜKLEME --------------------




def load_and_preprocess_data(data_dir, classes):
    """
    Termal görüntüleri yükler ve normalizasyon uygular

    Args:
        data_dir (str): Veri klasörü yolu
        classes (list): Sınıf etiketleri

    Returns:
        tuple: (images, labels, filenames)
    """
    images = []
    labels = []
    filenames = []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        print(f"{class_name} yükleniyor...")

        
        for root, dirs, files in os.walk(class_path):
            for img_file in files:
                try:
                    img_path = os.path.join(root, img_file)
                    img_path = os.path.normpath(img_path)
                    # Yalnızca görüntü dosyalarıyla çalış
                    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    # Görüntüyü yükle
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        color_mode='rgb',
                        target_size=IMG_SIZE
                    )

                    
                    img_array = tf.keras.preprocessing.image.img_to_array(img).astype("float32")
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    # Listeye ekle
                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_path)

                except Exception as e:
                    print(f"Hata: {img_file} yüklenemedi - {str(e)}")


    return np.array(images), np.array(labels), filenames

# -------------------- MODEL MİMARİSİ --------------------
# def create_advanced_model(input_shape, num_classes):
#     """
#     Optimize edilmiş CNN modelini oluşturur

#     Mimari Özellikleri:
#     - 3 Konvolüsyon Bloğu
#     - Global Average Pooling
#     - Gelişmiş Regularizasyon
#     - Entegre Veri Artırma

#     Args:
#         input_shape (tuple): Giriş görüntü boyutu
#         num_classes (int): Sınıf sayısı

#     Returns:
#         tf.keras.Model: Derlenmiş model
#     """
#     # Giriş katmanı ve veri artırma
#     inputs = layers.Input(shape=input_shape)

#     # Entegre veri artırma (sadece eğitimde aktif)
#     x = layers.RandomFlip("horizontal")(inputs)
#     x = layers.RandomContrast(0.1)(x)# Termal kontrast varyasyonu
#     x=  layers.RandomRotation(0.1)(x)
#     x = layers.RandomZoom(0.1)(x)

#     # 1. Konvolüsyon Bloğu
#     x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x) #Feature maplerde normalizasyon
#     x = layers.MaxPooling2D(2)(x)
#     x = layers.Dropout(0.2)(x)

#     # 2. Konvolüsyon Bloğu
#     x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D(2)(x)
#     x = layers.Dropout(0.3)(x)

#     # 3. Konvolüsyon Bloğu
#     x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
#     x = layers.GlobalAveragePooling2D()(x)  # Parametre optimizasyonu
#     x = layers.Dropout(0.5)(x)

    
#     # Çıkış Katmanı
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     # Modeli Derle
#     model = models.Model(inputs, outputs)

#     optimizer = tf.keras.optimizers.Adam(
#         learning_rate=LEARNING_RATE,
#         clipnorm=1.0  # Gradyan patlamalarını önle
#     )

#     model.compile(
#         optimizer=optimizer,
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model



import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def create_mobilenet_model(input_shape, num_classes):
    # Veri Artırma Katmanları
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1)
    ])

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # İlk aşamada donduruyoruz

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) # Kapasiteyi artırdık
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base_model # base_model'i de döndürüyoruz ki sonra açabilelim

#-------------------- CALLBACK'LER --------------------


def get_callbacks(model_name):
    """
    Eğitim sürecini yöneten callback'leri oluşturur

    Returns:
        list: Callback listesi
    """
    return [
        # Erken Durdurma: Validation loss'ta iyileşme olmazsa dur
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),

        # Öğrenme Oranı Optimizasyonu
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),

        # En iyi modeli kaydet
        callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),

        # Eğitim loglarını kaydet
        callbacks.CSVLogger(f'{model_name}_training_log.csv')
    ]

# -------------------- GÖRSELLEŞTİRME --------------------


def visualize_results(history, y_true, y_pred, CLASSES):
    """
    Eğitim sonuçlarını ve performans metriklerini görselleştirir

    Args:
        history: Eğitim geçmişi
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        classes: Sınıf isimleri
    """
    # Accuracy/Loss Grafikleri
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim')
    plt.plot(history.history['val_accuracy'], label='Doğrulama')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epok')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim')
    plt.plot(history.history['val_loss'], label='Doğrulama')
    plt.title('Model Kaybı')
    plt.xlabel('Epok')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # Karışıklık matrisini hesapla
    cm = confusion_matrix(y_true, y_pred, normalize='true')*100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)

    plt.title("Tahmin Karmaşıklık Matrisi (%)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    

    # PNG olarak kaydet
    output_path = "../analysis_results/confusion_matrix_yara_varmi_yok_mu1.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

# -------------------- TENSORFLOW LITE DÖNÜŞÜMÜ --------------------


def convert_to_tflite(model, model_name):
    """
    TensorFlow modelini Android için TFLite formatına dönüştürür

    Args:
        model: Eğitilmiş TensorFlow modeli (bu parametreyi kullanmayacağız)
        model_name (str): Model dosyası için temel isim

    Returns:
        str: Oluşturulan TFLite dosyasının yolu
    """
    print("\nModel TensorFlow Lite formatına dönüştürülüyor...")

    # H5 dosyasını kontrol et
    h5_path = f"{model_name}_final.h5"

    if not os.path.exists(h5_path):
        print(f"HATA: {h5_path} dosyası bulunamadı!")
        print("İpucu: Önce modeli eğitip kaydetmelisiniz.")
        return None

    try:
        # Dosyadan doğrudan TFLite dönüşümü yapmayı dene
        print(f"'{h5_path}' dosyasından TFLite dönüşümü yapılıyor...")

        # Manuel olarak TFLite dönüşümü yapalım
        import subprocess
        import sys

        tflite_filename = f"{model_name}.tflite"

        # Python script dosyası oluştur
        converter_script = "convert_to_tflite.py"
        with open(converter_script, "w") as f:
            f.write("""
import sys
import tensorflow as tf

# H5 dosyasını yükle
h5_path = sys.argv[1]
tflite_path = sys.argv[2]

# Modeli yükle
model = tf.keras.models.load_model(h5_path)

# TFLite dönüştürücüsünü oluştur
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimizasyon seçenekleri
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Dönüşümü gerçekleştir
tflite_model = converter.convert()

# TFLite modelini dosyaya kaydet
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite modeli başarıyla oluşturuldu: {tflite_path}")
print(f"Model boyutu: {len(tflite_model) / 1024:.2f} KB")
""")

        # Script'i ayrı bir Python işleminde çalıştır
        # Python 3.12 yerine kullanılabilir başka bir Python sürümünüz varsa, onu kullanın
        python_cmd = sys.executable  # Mevcut Python yorumlayıcısı
        cmd = [python_cmd, converter_script, h5_path, tflite_filename]

        print(f"Dönüşüm komutu çalıştırılıyor: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(result.stdout)
            print(f"Dönüşüm başarılı: {tflite_filename}")

            # Geçici script dosyasını temizle
            if os.path.exists(converter_script):
                os.remove(converter_script)

            return tflite_filename
        else:
            print(f"Dönüşüm başarısız. Hata: {result.stderr}")

            # Alternatif yöntem önerisi
            print("\nAlternatif çözüm önerileri:")
            print("1. Python 3.8 veya 3.9 sürümü kullanarak dönüşümü tekrar deneyin.")
            print(
                "2. TensorFlow'u pip ile güncellemeyi deneyin: pip install tensorflow==2.10.0")
            print(
                "3. Modeli daha düşük bir Python sürümünde (3.8/3.9) eğitip tekrar dönüştürün.")

            return None

    except Exception as e:
        print(f"TFLite dönüşümünde beklenmeyen hata: {str(e)}")
        print("\nPython 3.12 ile TensorFlow uyumsuzluğu yaşanıyor olabilir.")
        print("Önerilen çözümler:")
        print("1. Python 3.8 veya 3.9 sürümü kullanarak dönüşümü tekrar deneyin.")
        print(
            "2. TensorFlow'u pip ile güncellemeyi deneyin: pip install tensorflow==2.10.0")
        return None


def main():
    # Yapılandırma
    DATA_DIR = "wound_detection_dataset"
    CLASSES = ["Yara_Var", "Yara_Yok"]
    MODEL_NAME = "wound_detection_guncel_model"


    # import os
    # import numpy as np
    # from sklearn.utils import resample
    # import shutil

    # def list_all_images(folder_path):
    #     all_images = []
    #     subfolders = []

    #     # Alt klasörleri geziyor
    #     for root, dirs, files in os.walk(folder_path):
    #         for sub in dirs:
    #             subfolders.append(os.path.join(root, sub))
    #         break  # sadece bir seviye aşağı git

    #     # Alt klasörlerdeki tüm görselleri okuyor
    #     for sub in subfolders:
    #         files = [os.path.join(sub, f) for f in os.listdir(sub) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    #         all_images.extend(files)

    #     return all_images, subfolders


    # def oversample_multilevel(var_path, yok_path):

    #     var_images, var_folders = list_all_images(var_path)
    #     yok_images, yok_folders = list_all_images(yok_path)

    #     print("Toplam Yara Var görsel:", len(var_images))
    #     print("Toplam Yara Yok görsel:", len(yok_images))

    #     # Oversample gerekli mi?
    #     if len(var_images) < len(yok_images):
    #         minority_images = var_images
    #         minority_folders = var_folders
    #         difference = len(yok_images) - len(var_images)
    #         print("Yara_Var çoğaltılacak:", difference)

    #     else:
    #         minority_images = yok_images
    #         minority_folders = yok_folders
    #         difference = len(var_images) - len(yok_images)
    #         print("Yara_Yok çoğaltılacak:", difference)

    #     # Yeni örnekleri seç
    #     extra_samples = resample(minority_images, replace=True, n_samples=difference)

    #     # Alt klasörlere eşit dağıtmak için döngü
    #     folder_idx = 0

    #     for img_path in extra_samples:
    #         current_folder = minority_folders[folder_idx % len(minority_folders)]
    #         file_name = os.path.basename(img_path)
    #         dst = os.path.join(current_folder, "aug_" + file_name)
    #         shutil.copy(img_path, dst)

    #         folder_idx += 1

    #     print("Oversampling tamamlandı!")
    #     print(f"{difference} adet yeni görsel oluşturuldu.")


    
    # oversample_multilevel(
    #     var_path="Termal_Dataset/Yara_Var",
    #     yok_path="Termal_Dataset/Yara_Yok"
    # )





    # 1. Veri Yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y)))}")

    # 2. Veri Bölme
    print("\n[2/7] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,#sınıfları dengeli böler
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test,
        test_size=0.5,
        stratify=y_val_test,
        random_state=42
    )

   

    print(
        f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")

    # 3. Sınıf Ağırlıkları
    print("\n[3/7] Sınıf ağırlıkları hesaplanıyor...")

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
  


    class_weights = {i: w for i, w in enumerate(class_weights)}


    print("Sınıf Ağırlıkları:", class_weights)

   # 4. Model Oluşturma
    model, base_model = create_mobilenet_model((*IMG_SIZE, 3), len(CLASSES))

    # 5. Model Eğitimi (Aşama 1: Sadece Kafa Kısmı)
    print("\n[5/7] Aşama 1: Üst katmanlar eğitiliyor...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=20, batch_size=BATCH_SIZE, class_weight=class_weights,
              callbacks=get_callbacks(MODEL_NAME + "_stage1"))

    # Aşama 2: Fine-Tuning (MobileNet'in kilitlerini açıyoruz)
    print("\n[5.1/7] Aşama 2: Fine-tuning başlatılıyor...")
    base_model.trainable = True
    # Sadece son katmanları açmak (opsiyonel ama daha kararlı):
    # for layer in base_model.layers[:100]: layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Çok düşük hız
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=INITIAL_EPOCHS, batch_size=BATCH_SIZE, 
                        class_weight=class_weights, callbacks=get_callbacks(MODEL_NAME))

    # 6. Değerlendirme ve Raporlama
    print("\n[6/7] Performans değerlendiriliyor...")
    # En iyi modeli yükle
    model = load_model(f'{MODEL_NAME}_best.h5')
   
    # Test seti değerlendirme
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)#tahminlerini y_test ile karşılaştırır
    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    
    from sklearn.metrics import recall_score
    

    y_pred = model.predict(X_test)  # Probabilities

    y_pred_classes = np.argmax(y_pred, axis=1)  # Sınıf etiketleri

   
    # Detaylı Rapor
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test,y_pred_classes, target_names=CLASSES,digits=4))

    # Sonuçları Kaydet
    results_df = pd.DataFrame({
        'Dosya': filenames[-len(y_test):],
        'Gerçek': [CLASSES[i] for i in y_test],
        'Tahmin': [CLASSES[i] for i in y_pred_classes],
        'Yara_Var': y_pred[:, 0],
        'Yara_Yok': y_pred[:, 1],
        
    })
    results_df.to_csv(f'{MODEL_NAME}_predictions.csv', index=False)
    print("\nTahminler CSV'ye kaydedildi.")

    # Görselleştirme
    visualize_results(history, y_test, y_pred_classes, CLASSES)

    # Modeli Kaydet
    model.save(f'{MODEL_NAME}_final.h5')
    print("\nFinal model kaydedildi.")

    # 7. TensorFlow Lite Dönüşümü
    print("\n[7/7] Android için TFLite modeli oluşturuluyor...")
    tflite_path = convert_to_tflite(model, MODEL_NAME)
    if tflite_path:
        print(f"\nAndroid'de kullanılmaya hazır TFLite modeli: {tflite_path}")
    else:
        print("\nTFLite dönüşümü başarısız oldu. Manuel olarak dönüştürmeyi deneyin.")


if __name__ == "__main__":
    main()