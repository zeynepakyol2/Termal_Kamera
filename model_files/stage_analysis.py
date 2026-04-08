# -------------------- KÜTÜPHANELER --------------------
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model

# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 80
LEARNING_RATE = 1e-4
MIN_DELTA = 1e-5
PATIENCE = 15

RESULTS_DIR = "analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- VERİ YÜKLEME --------------------
def load_and_preprocess_data(data_dir, classes):
    images = []
    labels = []
    filenames = []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.exists(class_path):
            print(f"UYARI: Klasör bulunamadı -> {class_path}")
            continue

        print(f"{class_name} yükleniyor...")

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_path,
                    color_mode="rgb",
                    target_size=IMG_SIZE
                )

                img_array = tf.keras.preprocessing.image.img_to_array(img).astype("float32")
                img_array /= 255.0

                images.append(img_array)
                labels.append(class_idx)
                filenames.append(img_file)

            except Exception as e:
                print(f"Hata: {img_file} yüklenemedi -> {e}")

    return np.array(images), np.array(labels), np.array(filenames)

# -------------------- MODEL --------------------
def create_advanced_model(input_shape, num_classes):
    # Google'ın milyonlarca resimle eğitilmiş MobileNetV2 modelini indiriyoruz
    # include_top=False diyerek beynin sadece "görme" kısmını alıyoruz, sınıflandırma kısmını atıyoruz
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Önceden eğitilmiş bu devasa beynin ağırlıklarını donduruyoruz ki bozulmasın
    base_model.trainable = False 

    # Kendi katmanlarımızı ekliyoruz
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # 6 sınıflı kendi yara çıktımızı bağlıyoruz
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
# -------------------- CALLBACK --------------------
def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f"{model_name}_best.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        callbacks.CSVLogger(f"{model_name}_training_log.csv")
    ]

# -------------------- GÖRSELLEŞTİRME --------------------
def visualize_results(history, y_true, y_pred, classes):
    # Accuracy / Loss
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Eğitim")
    plt.plot(history.history["val_accuracy"], label="Doğrulama")
    plt.title("Model Doğruluğu")
    plt.xlabel("Epok")
    plt.ylabel("Doğruluk")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Eğitim")
    plt.plot(history.history["val_loss"], label="Doğrulama")
    plt.title("Model Kaybı")
    plt.xlabel("Epok")
    plt.ylabel("Kayıp")
    plt.legend()

    plt.tight_layout()
    acc_loss_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(acc_loss_path, bbox_inches="tight")
    plt.close()

    # Confusion matrix (%)
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title("Tahmin Karmaşıklık Matrisi (%)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Eğitim grafiği kaydedildi: {acc_loss_path}")
    print(f"Karmaşıklık matrisi kaydedildi: {cm_path}")

# -------------------- TFLITE --------------------
def convert_to_tflite(model_name):
    print("\nModel TensorFlow Lite formatına dönüştürülüyor...")

    h5_path = f"{model_name}_final.h5"
    tflite_filename = f"{model_name}.tflite"

    if not os.path.exists(h5_path):
        print(f"HATA: {h5_path} bulunamadı.")
        return None

    try:
        model = load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_filename, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite modeli oluşturuldu: {tflite_filename}")
        print(f"Model boyutu: {len(tflite_model) / 1024:.2f} KB")
        return tflite_filename

    except Exception as e:
        print(f"TFLite dönüşümünde hata: {e}")
        return None

# -------------------- ANA AKIŞ --------------------
def main():
    DATA_DIR = "wound_stage_dataset"
    CLASSES = ["Derin Doku Hasarı", "Evre 1", "Evre 2", "Evre 3", "Evre 4", "Evrelendirilemeyen"]
    MODEL_NAME = "wound_stage"

    # 1. Veri yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)

    if len(X) == 0:
        print("Veri bulunamadı. Program sonlandırıldı.")
        return

    # 2. Veri bölme (GÜNCELLENDİ: %80 Train, %10 Val, %10 Test)
    print("\n[2/7] Veri 3'e bölünüyor (Train, Val, Test)...")
    
    # Önce %80 Eğitim, %20 Geçici (Temp) olarak ayırıyoruz
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, filenames, test_size=0.20, stratify=y, random_state=42
    )

    # Sonra o %20'lik Geçici kısmı tam ortadan ikiye bölüyoruz (%10 Val, %10 Test)
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Eğitim (Train): {len(X_train)}")
    print(f"Doğrulama (Validation): {len(X_val)}")
    print(f"Test: {len(X_test)}")

    # 3. Sınıf ağırlıkları (Değişmedi)
    print("\n[3/7] Sınıf ağırlıkları hesaplanıyor...")
    classes_in_train = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_in_train,
        y=y_train
    )
    class_weights = dict(zip(classes_in_train, weights))

    # 4. Model oluşturma (Değişmedi)
    print("\n[4/7] Model inşa ediliyor...")
    model = create_advanced_model((*IMG_SIZE, 3), len(CLASSES))

    # 5. Eğitim (GÜNCELLENDİ: Validation artık X_val üzerinden yapılıyor)
    print("\n[5/7] Model eğitimi başlatılıyor...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val), # BURASI GÜNCELLENDİ
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_NAME),
        verbose=1
    )

    # En iyi modeli yükle (Değişmedi)
    best_model_path = f"{MODEL_NAME}_best.h5"
    if os.path.exists(best_model_path):
        model = load_model(best_model_path)

    # 6. Değerlendirme (GÜNCELLENDİ: Artık X_test ile dürüst ölçüm yapıyoruz)
    print("\n[6/7] Performans değerlendiriliyor (Gerçek Test Seti)...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0) # BURASI GÜNCELLENDİ
    print(f"\nGerçek Test Doğruluğu: {test_acc:.4f}")
    print(f"Gerçek Test Kaybı: {test_loss:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nSınıflandırma Raporu (Test Seti):")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES, digits=4))

    # CSV Kaydı ve Görselleştirme (X_test ve y_test olarak güncellendi)
    results_dict = {
        "Dosya": f_test,
        "Gerçek": [CLASSES[i] for i in y_test],
        "Tahmin": [CLASSES[i] for i in y_pred_classes]
    }
    for i, class_name in enumerate(CLASSES):
        safe_name = class_name.replace(" ", "_")
        results_dict[f"Olasilik_{safe_name}"] = y_pred[:, i]

    results_df = pd.DataFrame(results_dict)
    csv_path = f"{MODEL_NAME}_predictions.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTahminler CSV'ye kaydedildi: {csv_path}")

    # Görselleştirme
    visualize_results(history, y_test, y_pred_classes, CLASSES)

    # Final model kaydet
    final_model_path = f"{MODEL_NAME}_final.h5"
    model.save(final_model_path)
    print(f"\nFinal model kaydedildi: {final_model_path}")

    # 7. TFLite
    print("\n[7/7] Android için TFLite modeli oluşturuluyor...")
    tflite_path = convert_to_tflite(MODEL_NAME)
    if tflite_path:
        print(f"\nAndroid'de kullanılmaya hazır TFLite modeli: {tflite_path}")
    else:
        print("\nTFLite dönüşümü başarısız oldu.")

if __name__ == "__main__":
    main()