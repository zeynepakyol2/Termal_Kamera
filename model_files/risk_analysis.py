"""
BASINÇ YARASI RİSK SINIFLANDIRMA SİSTEMİ - OPTİMİZE EDİLMİŞ SÜRÜM
Termal görüntülerden basınç yarası risk seviyelerini sınıflandıran gelişmiş CNN modeli

Ana İyileştirmeler:
- Geliştirilmiş model mimarisi
- Optimize edilmiş hiperparametreler
- Termal görüntülere özel veri artırma
- Detaylı hata analizi ve görselleştirme
- Kararlı eğitim için regularizasyon teknikleri
"""

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
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (256, 256)        # Model giriş boyutu
BATCH_SIZE = 8               # Veri işleme boyutu
INITIAL_EPOCHS = 40          # Maksimum eğitim iterasyonu
LEARNING_RATE = 0.00001       # Başlangıç öğrenme oranı
MIN_DELTA = 0.00001          # Erken durdurma hassasiyeti
PATIENCE = 12                # Erken durdurma sabrı

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

        for img_file in os.listdir(class_path):
            try:
                # Görüntüyü gri tonlamalı olarak yükle
                img_path = os.path.join(class_path, img_file)
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    color_mode='rgb',  # Renkli görüntüleri yüklemek için değiştirildi
                    target_size=IMG_SIZE
                )

                #1) RGB → HSV → V (termal parlaklık kanalı)
                img_array = tf.keras.preprocessing.image.img_to_array(img)  # RGB
                # hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
                # T = hsv[:,:,2].astype(np.float32)   # sadece parlaklık → termal proxy

                # # Numpy array'e çevir ve normalizasyon yap
                # # img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array) # [0-1] aralığına normalizasyon

                # # Kanal boyutunu ekle (224,224,3)
                # # Renkli görüntüler için kanal boyutu zaten mevcut, bu nedenle ekleme gerekmez
                # T = cv2.normalize(T, None, 0.0, 1.0, cv2.NORM_MINMAX)
                # T = np.expand_dims(T, axis=-1)
                # T = np.repeat(T, 3, axis=-1) 
                # img_array = T

                
                images.append(img_array)
                labels.append(class_idx)
                filenames.append(img_file)

            except Exception as e:
                print(f"Hata: {img_file} yüklenemedi - {str(e)}")

    return np.array(images), np.array(labels), filenames

# -------------------- MODEL MİMARİSİ --------------------

def sparse_focal_loss(gamma=2.0, alpha=0.25):
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=tf.shape(y_pred)[-1])

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn

FOCAL_ALPHA = [2.0896919, 2.1128857, 2.1081853]
FOCAL_LOSS = sparse_focal_loss(gamma=2.0, alpha=FOCAL_ALPHA)

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

#     x = layers.RandomFlip("horizontal")(inputs)
#     x = layers.RandomContrast(0.03)(x)
#     x = layers.RandomRotation(0.02)(x)
#     x = layers.RandomZoom(0.03)(x)

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
#     optimizer=optimizer,
#     loss=FOCAL_LOSS,
#     metrics=['accuracy']
#     )

#     return model


def create_efficientnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Data Augmentation (Biraz daha agresif hale getirdik)
    x = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1) # Termal görüntülerde parlaklık değişimi önemlidir
    ])(inputs)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # ÖNEMLİ: İlk aşamada donduruyoruz
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x) # Nöron sayısını artırdık
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    # İlk aşama için yüksek learning rate (0.001)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=FOCAL_LOSS, metrics=['accuracy'])
    return model, base_model




#-------------------- CALLBACK'LER --------------------


def get_callbacks(model_name):
    """
    Eğitim sürecini yöneten callback'leri oluşturur

    Returns:
        list: Callback listesi
    """
    return[
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
            save_weights_only=False,
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
    output_path = "../analysis_results/confusion_matrix_risk_analysis1.png"
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

# -------------------- ANA İŞLEM --------------------


def main():
    # Yapılandırma
    DATA_DIR = "wound_risk_dataset"
    CLASSES = ["Goreceli_Risk", "Yuksek_Risk", "Cok_Yuksek_Risk"]
    MODEL_NAME = "wound_risk_model_guncel"



    # 1. Veri Yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y)))}")

    # 2. Veri Bölme
    print("\n[2/7] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y,
        test_size=0.2,
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
    # class_weights = compute_class_weight(
    #         'balanced',
    #         classes=np.unique(y_train),
    #         y=y_train
    #   )
    
    #class_weights = {i: w for i, w in enumerate(class_weights)}
    class_weights = {
    0: 1.0,
    1: 3.5,
    2: 2.8
    }

    # idx_y = CLASSES.index("Yuksek_Risk")  
    # class_weights[idx_y] *= 1.2    #Yüksek risk ağırlığını artır

    print("Sınıf Ağırlıkları:", class_weights)

    # #4. Model Oluşturma
    print("\n[4/7] Model inşa ediliyor...")
    model, base_model = create_efficientnet_model((*IMG_SIZE, 3), len(CLASSES))
    model.summary()

    # 5. Model Eğitimi
    print("\n[5/7] Model eğitimi başlatılıyor...")
 # 5. Model Eğitimi (İki Aşamalı Strateji)
    print("\n[5/7] Aşama 1: Üst katmanlar eğitiliyor (Warm-up)...")
    # Burada sadece senin eklediğin Dense katmanları eğitilecek, EfficientNet donuk kalacak.
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15, # İlk aşama için 15 epok yeterli
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_NAME + "_stage1"), # Stage 1 için ayrı log tutabilirsin
        verbose=1
    )

    # 6. Aşama: Fine-Tuning (Tüm katmanları açıyoruz)
    print("\n[6/7] Aşama 2: Fine-tuning başlatılıyor (Tüm katmanlar açıldı)...")
    
    # EfficientNet dahil tüm katmanları eğitilebilir yapıyoruz
    for layer in model.layers:
        layer.trainable = True
    
    # ÇOK ÖNEMLİ: Katmanları açtıktan sonra modeli ÇOK DÜŞÜK bir hızla tekrar derlemelisin
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 1e-3'ten 1e-5'e düşürdük
        loss=FOCAL_LOSS, 
        metrics=['accuracy']
    )
    
    # Asıl eğitim burada başlıyor
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25, # Toplamda 40 epok (15+25) yapmış oluyoruz
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_NAME),
        verbose=1
    )

    # 7. Değerlendirme ve Raporlama (Bundan sonrası senin kodunla aynı devam ediyor)
    print("\n[7/7] Performans değerlendiriliyor...")
    # ... (Geri kalan değerlendirme kodlarını buraya ekle)

    # 6. Değerlendirme ve Raporlama
    print("\n[6/7] Performans değerlendiriliyor...")
    # En iyi modeli yükle
    model, base_model = create_efficientnet_model((*IMG_SIZE, 3), len(CLASSES))
    model.load_model(f'{MODEL_NAME}_best.h5')
   
    # Test seti değerlendirme
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)#tahminlerini y_test ile karşılaştırır
    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    

    y_pred = model.predict(X_test)#tahminleri yazdırır
    y_pred_classes = np.argmax(y_pred, axis=1).copy()


    from sklearn.metrics import f1_score,recall_score

    # idx_G = 0   # Goreceli_Risk
    # idx_Y = 1   # Yuksek_Risk
    # idx_C = 2 
    # # Denenecek threshold aralıkları
    # T_Y_values = [0.25,0.26,0.27,0.28 ,0.29,0.30,0.31,0.32,0.33,0.34,0.35]
    # T_C_values = [0.10,0.15,0.20,0.30,0.38, 0.40, 0.42,0.43,0.44,0.45]
    # T_G_values = [0.50,0.55,0.59,0.60,0.61,0.62,0.64,0.66,0.68,0.70]
    # DELTA_values = [0.01,0.02,0.03,0.04]

    # best_f1 = 0
    # best_params = None
    # best_preds = None

    # for T_G in T_G_values:
    #     for T_Y in T_Y_values:
    #         for T_C in T_C_values:
    #             for DELTA in DELTA_values:
    #                 y_pred_classes = np.full(len(y_pred), -1, dtype=int)
    #                 for i, p in enumerate(y_pred):
    #                     pG, pY, pC = p[idx_G], p[idx_Y], p[idx_C]

                    
    #                     if (pC >= T_C) and ((pC - pY) >= DELTA*0.5) :
    #                         y_pred_classes[i] = idx_C
    #                     elif (pY >= T_Y and (pY - pG) >= DELTA*0.5):
    #                         y_pred_classes[i] = idx_Y
    #                     elif (pY >= T_Y and (pY - pC)>= DELTA*0.5):
    #                         y_pred_classes[i] = idx_Y
    #                     elif pG >= T_G and ((pG - max(pY, pC)) >= DELTA):
    #                         y_pred_classes[i] = idx_G
    #                     else:
    #                         y_pred_classes[i] = idx_Y 
                        

    #                 recall_Y = recall_score(y_test,y_pred_classes,  labels=[idx_Y], average='macro')
    #                 recall_C = recall_score(y_test, y_pred_classes, labels=[idx_C], average='macro')

    #                 if recall_Y>= 0.29 and recall_C >= 0.29:
    #                     macro_f1 = f1_score(y_test, y_pred_classes, average="macro")
    #                     if macro_f1 > best_f1:
    #                         best_f1 = macro_f1
    #                         best_params = (T_Y, T_C,T_G, DELTA)
    #                         best_preds = y_pred_classes.copy()
    # if best_params is None:
    #     print(" Hiçbir kombinasyon recall ≥ 0.29 şartını sağlamadı.")
    #     return
        
    # print("EN İYİ SONUÇ")
    # print("T_Y =", best_params[0], "| T_C =", best_params[1],"| T_G =", best_params[2], "| DELTA =", best_params[3])
    # print("Macro F1 Skoru:", best_f1)



    # Detaylı Rapor
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test,y_pred_classes, target_names=CLASSES,digits=4))

    # Sonuçları Kaydet
    results_df = pd.DataFrame({
        'Dosya': filenames[-len(y_test):],
        'Gerçek': [CLASSES[i] for i in y_test],
        'Tahmin': [CLASSES[i] for i in y_pred_classes],
        'Göreceli_Olasılık': y_pred[:, 0],
        'Yüksek_Olasılık': y_pred[:, 1],
        'Çok_Yüksek_Olasılık': y_pred[:, 2]
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
