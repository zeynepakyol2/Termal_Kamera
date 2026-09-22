"""
Yara var mi yok mu? - MobileNetV2 Transfer Learning

"""

# -------------------- KÜTÜPHANELER --------------------
from datetime import datetime
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                              accuracy_score, recall_score)
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from focal_loss import SparseCategoricalFocalLoss


# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224)          
BATCH_SIZE = 32

INITIAL_EPOCHS = 30             # Aşama 1: dondurulmuş base ile eğitim tavanı
FINE_TUNE_EPOCHS = 30           # Aşama 2: fine-tuning eğitim tavanı
FINE_TUNE_AT_LAYER = 100        # Bu indeksten SONRAKİ katmanların kilidi açılır
                               

LR_INITIAL = 1e-3               # Aşama 1 öğrenme oranı 
LR_FINE_TUNE = 1e-5             # Aşama 2 öğrenme oranı 

MIN_DELTA = 1e-3
PATIENCE = 12
L2_REG = 1e-4
FOCAL_GAMMA = 2.0
WOUND_RECALL_BOOST = 1.3
MIN_WOUND_RECALL = 0.75


# -------------------- AÇMA/KAPAMA ANAHTARLARI --------------------
ENABLE_DIAGNOSTICS = True       
ENABLE_TFLITE_QUANTIZATION = True  


# -------------------- VERİ YÜKLEME --------------------

def load_and_preprocess_data(data_dir, classes):
    """
    Görüntüleri yükler ve MobileNetV2 (224,224,3) ister

    Returns:
        tuple: (images, labels, filenames) - üçü de aynı sırada, birebir hizalı
    """
    images, labels, filenames = [], [], []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        print(f"{class_name} yükleniyor...")

        for root, _, files in os.walk(class_path):
            for img_file in files:
                try:
                    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    img_path = os.path.normpath(os.path.join(root, img_file))
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, color_mode='rgb', target_size=IMG_SIZE
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img).astype("float32")
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_path)

                except Exception as e:
                    print(f"Hata: {img_file} yüklenemedi - {str(e)}")

    return np.array(images), np.array(labels), np.array(filenames)


# -------------------- MODEL --------------------

def create_mobilenetv2_model(input_shape, num_classes):
    """
    MobileNetV2 tabanlı transfer learning modeli oluşturur.

    Returns:
        (model, base_model) - base_model'i ayrı döndürüyoruz çünkü fine-tuning
        aşamasında onun trainable durumunu ve içindeki katmanları değiştireceğiz.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,      
        weights='imagenet'
    )
    base_model.trainable = False   # Aşama 1: tamamen dondur

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)   # training=False: BatchNorm istatistiklerini bozma
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="mobilenetv2_wound_model")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=SparseCategoricalFocalLoss(gamma=FOCAL_GAMMA),
        metrics=['accuracy']
    )


# -------------------- CALLBACK'LER --------------------

def get_callbacks(model_name, patience=PATIENCE):
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, min_delta=MIN_DELTA,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'{model_name}_best.keras', save_best_only=True,
            monitor='val_loss', mode='min', verbose=1
        ),
        callbacks.CSVLogger(f'{model_name}_training_log.csv')
    ]


# -------------------- İKİ AŞAMALI EĞİTİM --------------------

def train_two_phase(model, base_model, X_train, y_train, X_val, y_val,
                     class_weights, model_name):
    """
    Aşama 1: base_model dondurulmuş halde, sadece yeni katmanları eğit.
    Aşama 2: base_model'in üst katmanlarının kilidini aç, düşük LR ile
             tüm modeli birlikte ince ayar et (fine-tune).
    """
    print("\n--- AŞAMA 1: Feature extraction (base dondurulmuş) ---")
    compile_model(model, LR_INITIAL)
    history_1 = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=INITIAL_EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(f"{model_name}_phase1")
    )

    print("\n--- AŞAMA 2: Fine-tuning (üst katmanların kilidi açık) ---")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    compile_model(model, LR_FINE_TUNE)   
    history_2 = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_1.epoch[-1] + 1,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(f"{model_name}_phase2")
    )

    return history_1, history_2


# -------------------- GÖRSELLEŞTİRME --------------------

def visualize_results(history_1, history_2, y_true, y_pred, CLASSES, y_all):
    acc = history_1.history['accuracy'] + history_2.history['accuracy']
    val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    loss = history_1.history['loss'] + history_2.history['loss']
    val_loss = history_1.history['val_loss'] + history_2.history['val_loss']
    switch_epoch = len(history_1.history['accuracy'])

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Eğitim')
    plt.plot(val_acc, label='Doğrulama')
    plt.axvline(switch_epoch, linestyle='--', color='gray', label='Fine-tuning başlangıcı')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epok'); plt.ylabel('Doğruluk'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Eğitim')
    plt.plot(val_loss, label='Doğrulama')
    plt.axvline(switch_epoch, linestyle='--', color='gray', label='Fine-tuning başlangıcı')
    plt.title('Model Kaybı')
    plt.xlabel('Epok'); plt.ylabel('Kayıp'); plt.legend()

    plt.tight_layout()
    plt.savefig("../analysis_results/detect_wound_accuracy_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    counts = np.bincount(y_all)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(CLASSES, counts)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  str(int(bar.get_height())), ha='center')
    plt.title("Veri Kümesindeki Görüntü Sayıları")
    plt.xlabel("Sınıf"); plt.ylabel("Görüntü Sayısı")
    plt.savefig("../analysis_results/detect_wound_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Tahmin Karmaşıklık Matrisi (%)")
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
    plt.savefig("../analysis_results/detect_wound_confusion_matrix.png", bbox_inches="tight")
    plt.show()



# -------------------- TAHMİN (TEST-TIME AUGMENTATION) --------------------

def predict_tta(model, X, batch_size=32):
    """Orijinal + yatay + dikey çevrilmiş görüntülerin tahminlerini ortalar."""
    preds_orig = model.predict(X, batch_size=batch_size, verbose=0)
    preds_h = model.predict(X[:, :, ::-1, :], batch_size=batch_size, verbose=0)
    preds_v = model.predict(X[:, ::-1, :, :], batch_size=batch_size, verbose=0)
    return (preds_orig + preds_h + preds_v) / 3.0


# -------------------- TFLITE'A DÖNÜŞTÜRME --------------------

def convert_to_tflite(model, output_path, quantize=ENABLE_TFLITE_QUANTIZATION):
    """
    Eğitilmiş Keras modelini TFLite (.tflite) formatına dönüştürüp diske yazar.

    NOT: Focal loss sadece EĞİTİM sırasında (loss hesaplaması için) gerekliydi.
    TFLite dönüşümü modelin ileri-yön (forward pass/inference) grafiğini
    aldığı için loss fonksiyonuyla ilgilenmez; bu yüzden custom_objects
    burada gerekmiyor.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TFLite model kaydedildi -> {output_path} ({size_mb:.2f} MB)")


# -------------------- ANA AKIŞ --------------------

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    DATA_DIR = "wound_detection_dataset"
    CLASSES = ["wound", "no_wound"]   
    MODEL_NAME = f"wound_detection_mobilenetv2_{today}"

    # 1. Veri Yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y)))}")

    # 2. Veri Bölme
    print("\n[2/7] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test, f_train, f_val_test = train_test_split(
        X, y, filenames, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_val_test, y_val_test, f_val_test, test_size=0.5, stratify=y_val_test, random_state=42
    )
    print(f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")

   

    # 3. Sınıf Ağırlıkları
    print("\n[3/7] Sınıf ağırlıkları hesaplanıyor...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weights)}
    class_weights[0] *= WOUND_RECALL_BOOST
    print("Sınıf Ağırlıkları (wound boost uygulanmış):", class_weights)

    # 4. Model Oluşturma ve İki Aşamalı Eğitim
    print("\n[4/7] MobileNetV2 modeli oluşturuluyor...")
    model, base_model = create_mobilenetv2_model((*IMG_SIZE, 3), len(CLASSES))

    print("\n[5/7] Model eğitiliyor (Aşama 1 + Aşama 2)...")
    history_1, history_2 = train_two_phase(
        model, base_model, X_train, y_train, X_val, y_val, class_weights, MODEL_NAME
    )

    # 6. Değerlendirme
    print("\n[6/7] Performans değerlendiriliyor...")
    model = load_model(
        f'{MODEL_NAME}_phase2_best.keras',
        custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss}
    )
    # Grad-CAM için base_model referansını yüklenen modelden tekrar al
    base_model = model.get_layer('mobilenetv2_1.00_224') if 'mobilenetv2_1.00_224' in \
        [l.name for l in model.layers] else next(l for l in model.layers if isinstance(l, tf.keras.Model))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Doğruluğu (ham): {test_acc:.4f} | Test Kaybı: {test_loss:.4f}")

    # ---- EŞİK OPTİMİZASYONU (validation setinde, TTA ile) ----
    print("\nTTA ile doğrulama tahminleri üretiliyor...")
    val_pred = predict_tta(model, X_val)

    candidates = []
    for t in np.arange(0.3, 0.7, 0.02):
        preds = np.where(val_pred[:, 0] > t, 0, 1)
        acc = accuracy_score(y_val, preds)
        wound_recall = recall_score(y_val, preds, pos_label=0)
        candidates.append((t, acc, wound_recall))

    qualifying = [c for c in candidates if c[2] >= MIN_WOUND_RECALL]
    if qualifying:
        best_thresh, best_acc, best_wound_recall = max(qualifying, key=lambda c: c[1])
        print(f"({len(qualifying)} eşik, wound recall >= {MIN_WOUND_RECALL:.2f} şartını sağladı)")
    else:
        best_thresh, best_acc, best_wound_recall = max(candidates, key=lambda c: c[2])
        print(f"UYARI: hiçbir eşik wound recall >= {MIN_WOUND_RECALL:.2f} şartını sağlamadı, "
              f"en iyi wound recall'lu eşik seçildi.")

    val_f1 = f1_score(y_val, np.where(val_pred[:, 0] > best_thresh, 0, 1), average='macro')
    print(f"\nEn iyi eşik: {best_thresh:.2f} (Val Doğruluk: {best_acc:.4f}, "
          f"Val Wound Recall: {best_wound_recall:.4f}, Val Macro F1: {val_f1:.4f})")

    print("\nTTA ile test tahminleri üretiliyor...")
    y_pred = predict_tta(model, X_test)
    y_pred_classes = np.where(y_pred[:, 0] > best_thresh, 0, 1)

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES, digits=4))

    tta_test_acc = accuracy_score(y_test, y_pred_classes)
    print(f"TTA + optimize edilmiş eşik ile Test Doğruluğu: {tta_test_acc:.4f}")

    # Sonuçları Kaydet
    results_df = pd.DataFrame({
        'Dosya': f_test,
        'Gerçek': [CLASSES[i] for i in y_test],
        'Tahmin': [CLASSES[i] for i in y_pred_classes],
        'Yara_Var': y_pred[:, 0],
        'Yara_Yok': y_pred[:, 1],
    })
    results_df.to_csv(f'{MODEL_NAME}_predictions.csv', index=False)
    print("\nTahminler CSV'ye kaydedildi.")


    # Görselleştirme
    visualize_results(history_1, history_2, y_test, y_pred_classes, CLASSES, y)

    # Modeli TFLite Olarak Kaydet
    print("\n[7/7] Model TFLite formatına dönüştürülüyor...")
    convert_to_tflite(model, f'{MODEL_NAME}_final.tflite')


if __name__ == "__main__":
    main()