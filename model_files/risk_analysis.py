"""
Yara olmayan görsellerde risk analizi yapar ve sınıflandırır-EfficientNetB0 Transfer Learning

"""

# -------------------- KÜTÜPHANELER --------------------
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.models import Model




# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224) 
BATCH_SIZE = 8

WARMUP_EPOCHS = 15
FINE_TUNE_EPOCHS = 15 

LR_INITIAL = 1e-3
LR_FINE_TUNE = 5e-6  

MIN_DELTA = 1e-5
PATIENCE_PHASE1 = 12
PATIENCE_PHASE2 = 6  

UNFREEZE_LAST_N_LAYERS = 30
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 1.0

CLASS_WEIGHTS = {0: 0.65, 1: 1.80, 2: 2.20}#goreceli, yuksek, cok_yuksek


# -------------------- VERİ YÜKLEME --------------------
def load_and_preprocess_data(data_dir, classes):
    """
    Termal görüntüleri yükler ve EfficientNetB0 için hazırlar.

    Görüntüler RGB olarak okunur çünkü ImageNet ağırlıklı EfficientNetB0
    üç kanallı giriş bekler.

    Returns:
        images: Görüntü dizisi
        labels: Sayısal sınıf etiketleri
        filenames: Görüntülerin tam dosya yolları
    """
    images, labels, filenames = [], [], []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        print(f"{class_name} yükleniyor...")

        if not os.path.isdir(class_path):
            print(f"UYARI: Klasör bulunamadı -> {class_path}")
            continue

        for root, _, files in os.walk(class_path):
            for img_file in files:
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                try:
                    img_path = os.path.normpath(os.path.join(root, img_file))
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        color_mode="rgb",
                        target_size=IMG_SIZE,
                    )

                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array.astype("float32")
                    img_array = tf.keras.applications.efficientnet.preprocess_input(
                        img_array
                    )

                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_path)

                except Exception as e:
                    print(f"Hata: {img_file} yüklenemedi - {e}")

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(filenames),
    )


# -------------------- FOCAL LOSS --------------------
def sparse_focal_loss(gamma=2.0, alpha=1.0):
    """
    Sparse categorical focal loss oluşturur.

    Focal Loss, modelin kolay örneklerden çok zor/yanlış sınıflandırılan
    örneklere odaklanmasını sağlar. Sınıf dengesizliği burada DEĞİL,
    model.fit'e verilen class_weight ile ele alınır (çift ağırlıklandırmayı
    önlemek için).
    """
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = alpha * tf.pow(1.0 - y_pred, gamma)
        loss = focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return loss_fn


FOCAL_LOSS = sparse_focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)


# -------------------- MODEL --------------------
def create_efficientnet_model(input_shape, num_classes):
    """
    EfficientNetB0 tabanlı transfer learning modeli oluşturur.

    İlk aşamada EfficientNetB0 tamamen dondurulur. Böylece yalnızca üzerine
    eklenen Dense katmanları eğitilir.

    Returns:
        model: Tam sınıflandırma modeli
        base_model: Fine-tuning sırasında kısmen açılacak EfficientNetB0 modeli
    """
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.10),
        ],
        name="data_augmentation",
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = BatchNormalization(name="head_batch_norm")(x)
    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="classifier_dense",
    )(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", name="risk_output")(x)

    model = Model(inputs, outputs, name="efficientnetb0_wound_risk_model")
    return model, base_model


def unfreeze_model_partial(base_model, num_layers_to_unfreeze):
    """
    Fine-tuning için backbone'u KISMEN açar.

    """
    base_model.trainable = True

    freeze_until = max(0, len(base_model.layers) - num_layers_to_unfreeze)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(
        f"Backbone kısmi açıldı: {trainable_count}/{len(base_model.layers)} "
        f"katman eğitilebilir (BatchNorm katmanları hariç)."
    )


def compile_model(model, learning_rate):
    """Modeli verilen learning rate ile derler."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=FOCAL_LOSS,
        metrics=["accuracy"],
    )


# -------------------- CALLBACK'LER --------------------
def get_callbacks(model_name, phase_name, patience):
   
    checkpoint_path = f"{model_name}_{phase_name}_best.keras"

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        ),
        callbacks.CSVLogger(f"{model_name}_{phase_name}_training_log.csv"),
    ]


# -------------------- İKİ AŞAMALI EĞİTİM --------------------
def train_two_phase(
    model,
    base_model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    model_name,
    enable_fine_tuning=True,
):
    """
    Modeli iki aşamada eğitir.

    Aşama 1:
        EfficientNetB0 dondurulur ve yalnızca sınıflandırma katmanları eğitilir.

    Aşama 2 (opsiyonel, enable_fine_tuning=True ise):
        EfficientNetB0'ın yalnızca son UNFREEZE_LAST_N_LAYERS katmanı açılır
        (BatchNorm hariç) ve çok düşük learning rate ile fine-tuning yapılır.
    """
    print("\n--- AŞAMA 1: Feature extraction (EfficientNet dondurulmuş) ---")
    compile_model(model, LR_INITIAL)

    history_1 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=WARMUP_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase1", PATIENCE_PHASE1),
        verbose=1,
    )

    if not enable_fine_tuning:
        print("\nFine-tuning devre dışı bırakıldı (ENABLE_FINE_TUNING=False).")
        return history_1, None

    print(
        f"\n--- AŞAMA 2: Kısmi fine-tuning "
        f"(son {UNFREEZE_LAST_N_LAYERS} katman açık, BatchNorm donuk) ---"
    )
    unfreeze_model_partial(base_model, UNFREEZE_LAST_N_LAYERS)

    # Katmanların trainable durumu değiştiği için modeli tekrar derlemek gerekir.
    compile_model(model, LR_FINE_TUNE)

    phase1_epochs_completed = len(history_1.epoch)
    total_epochs = phase1_epochs_completed + FINE_TUNE_EPOCHS

    history_2 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        initial_epoch=phase1_epochs_completed,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase2", PATIENCE_PHASE2),
        verbose=1,
    )

    best_phase1_val_loss = min(history_1.history["val_loss"])
    best_phase2_val_loss = min(history_2.history["val_loss"])

    if best_phase2_val_loss > best_phase1_val_loss:
        print(
            "\nUYARI: Fine-tuning sonrası en iyi doğrulama kaybı "
            f"({best_phase2_val_loss:.4f}), Aşama 1'deki en iyi değerden "
            f"({best_phase1_val_loss:.4f}) daha kötü. "
            f"Aşama 1 checkpoint'i ({model_name}_phase1_best.keras) "
            "geri yükleniyor."
        )
        model.load_weights(f"{model_name}_phase1_best.keras")

    return history_1, history_2


# -------------------- GÖRSELLEŞTİRME --------------------
def visualize_results(history_1, history_2, y_true, y_pred, classes):
    """Eğitim/doğrulama grafikleri ve confusion matrix oluşturur."""
    os.makedirs("../analysis_results", exist_ok=True)

    accuracy = list(history_1.history["accuracy"])
    val_accuracy = list(history_1.history["val_accuracy"])
    loss = list(history_1.history["loss"])
    val_loss = list(history_1.history["val_loss"])
    switch_epoch = len(history_1.history["accuracy"])

    if history_2 is not None:
        accuracy += history_2.history["accuracy"]
        val_accuracy += history_2.history["val_accuracy"]
        loss += history_2.history["loss"]
        val_loss += history_2.history["val_loss"]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label="Eğitim")
    plt.plot(val_accuracy, label="Doğrulama")
    if history_2 is not None:
        plt.axvline(switch_epoch, linestyle="--", label="Fine-tuning başlangıcı")
    plt.title("Model Doğruluğu")
    plt.xlabel("Epok")
    plt.ylabel("Doğruluk")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Eğitim")
    plt.plot(val_loss, label="Doğrulama")
    if history_2 is not None:
        plt.axvline(switch_epoch, linestyle="--", label="Fine-tuning başlangıcı")
    plt.title("Model Kaybı")
    plt.xlabel("Epok")
    plt.ylabel("Kayıp")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"../analysis_results/wound_risk_accuracy_loss.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Tahmin Karmaşıklık Matrisi (%)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(
        f"../analysis_results/confusion_matrix_risk_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# -------------------- DEĞERLENDİRME --------------------
def evaluate_model(model, X_test, y_test, f_test, classes, model_name):
    """Modeli test verisi üzerinde değerlendirir ve sonuçları CSV'ye kaydeder."""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nSınıflandırma Raporu:")
    print(
        classification_report(
            y_test,
            y_pred_classes,
            target_names=classes,
            digits=4,
        )
    )

    results_df = pd.DataFrame(
        {
            "Dosya": f_test,
            "Gerçek": [classes[i] for i in y_test],
            "Tahmin": [classes[i] for i in y_pred_classes],
            "Göreceli_Olasılık": y_pred[:, 0],
            "Yüksek_Olasılık": y_pred[:, 1],
            "Çok_Yüksek_Olasılık": y_pred[:, 2],
        }
    )

    csv_path = f"{model_name}_predictions.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nTahminler CSV'ye kaydedildi -> {csv_path}")

    return y_pred, y_pred_classes


# -------------------- TFLITE'A DÖNÜŞTÜRME --------------------
def convert_to_tflite(model, output_path):
    """Eğitilmiş Keras modelini TensorFlow Lite formatına dönüştürür."""
    print("\nModel TensorFlow Lite formatına dönüştürülüyor...")

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"TFLite model kaydedildi -> {output_path} ({size_mb:.2f} MB)")
        return output_path

    except Exception as e:
        print(f"TFLite dönüşümünde hata oluştu: {e}")
        return None


# -------------------- ANA AKIŞ --------------------
def main():
    today = datetime.now().strftime("%Y-%m-%d")
    DATA_DIR = "wound_detection_dataset/no_wound"
    CLASSES = ["Goreceli_Risk", "Yuksek_Risk", "Cok_Yuksek_Risk"]
    MODEL_NAME = f"wound_risk_model_{today}"


    ENABLE_FINE_TUNING = True

    # 1. Veri Yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)

    if len(X) == 0:
        raise RuntimeError(
            "Hiç görüntü yüklenemedi. DATA_DIR ve sınıf klasörlerini kontrol edin."
        )

    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES))))}")

    # 2. Veri Bölme
    print("\n[2/7] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test, f_train, f_val_test = train_test_split(
        X,
        y,
        filenames,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_val_test,
        y_val_test,
        f_val_test,
        test_size=0.50,
        stratify=y_val_test,
        random_state=42,
    )

    print(
        f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}"
    )

    # 3. Sınıf Ağırlıkları
    print("\n[3/7] Sınıf ağırlıkları hazırlanıyor...")
    class_weights = CLASS_WEIGHTS.copy()
    print("Sınıf Ağırlıkları:", class_weights)

    # 4. Model Oluşturma
    print("\n[4/7] EfficientNetB0 modeli oluşturuluyor...")
    model, base_model = create_efficientnet_model(
        (*IMG_SIZE, 3),
        len(CLASSES),
    )
    model.summary()

    # 5. İki Aşamalı Eğitim
    print("\n[5/7] Model eğitiliyor (Aşama 1 + Aşama 2)...")
    history_1, history_2 = train_two_phase(
        model,
        base_model,
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        MODEL_NAME,
        enable_fine_tuning=ENABLE_FINE_TUNING,
    )

    # 6. Değerlendirme
    print("\n[6/7] Performans değerlendiriliyor...")
    _, y_pred_classes = evaluate_model(
        model,
        X_test,
        y_test,
        f_test,
        CLASSES,
        MODEL_NAME,
    )

    visualize_results(
        history_1,
        history_2,
        y_test,
        y_pred_classes,
        CLASSES,
    )

    # 6. Model Kaydetme
    final_model_path = f"{MODEL_NAME}_final.keras"
    model.save(final_model_path)
    print(f"\nFinal model kaydedildi -> {final_model_path}")

    # 7. TFLite Dönüşümü
    print("\n[7/7] Android için TFLite modeli oluşturuluyor...")
    tflite_path = convert_to_tflite(
        model,
        f"{MODEL_NAME}_final.tflite",
    )

    if tflite_path:
        print(f"\nAndroid'de kullanılmaya hazır model -> {tflite_path}")
    else:
        print("\nTFLite dönüşümü başarısız oldu.")


if __name__ == "__main__":
    main()