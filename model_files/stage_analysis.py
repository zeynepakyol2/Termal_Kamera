"""
Yara olan görsellerde evre analizi yapar ve sınıflandırır-MobileNetV2 Transfer Learning

Evre 4'te yalnızca 30 civarı görüntü var (test setinde 3 örnek!). Bu sayıyla o
sınıf için ölçülen hiçbir metrik güvenilir değil.

"""

# -------------------- KÜTÜPHANELER --------------------
import os
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers



# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224) 
BATCH_SIZE = 16

WARMUP_EPOCHS = 40
FINE_TUNE_EPOCHS = 25

LR_INITIAL = 1e-3
LR_FINE_TUNE = 2e-6 

MIN_DELTA = 1e-4
PATIENCE_PHASE1 = 15  
PATIENCE_PHASE2 = 10

UNFREEZE_LAST_N_LAYERS = 15

FT_WARMUP_EPOCHS = 4

DENSE_UNITS = 128
DROPOUT_RATE = 0.3
L2_LAMBDA = 3e-5

# Evre 4 çok az örnekli olduğu için Evre 3 ile birleştirme seçeneği.
# İlk denemede True önerilir.
MERGE_RARE_CLASSES = True

MAX_CLASS_WEIGHT = 4.0

RESULTS_DIR = "../analysis_results"
os.makedirs("../analysis_results", exist_ok=True)


# -------------------- VERİ YÜKLEME --------------------
def load_and_preprocess_data(data_dir, classes):
    """
    Yara görüntülerini yükler ve MobileNetV2 için hazırlar.

    ÖNEMLİ: Görüntüler /255.0 ile DEĞİL, mobilenet_v2.preprocess_input ile
    ölçeklenir ([-1, 1] aralığı). ImageNet ağırlıklarının doğru çalışması
    için bu zorunludur.

    Returns:
        images: Görüntü dizisi
        labels: Sayısal sınıf etiketleri
        filenames: Görüntülerin tam dosya yolları
    """
    images, labels, filenames = [], [], []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        print(f"{class_name} yükleniyor...")

        if not os.path.isdir(class_path):
            print(f"UYARI: Klasör bulunamadı -> {class_path}")
            continue

        count = 0
        for root, _, files in os.walk(class_path):
            for img_file in files:
                if not img_file.lower().endswith(valid_ext):
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
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                        img_array
                    )

                    images.append(img_array)
                    labels.append(class_idx)
                    filenames.append(img_path)
                    count += 1

                except Exception as e:
                    print(f"Hata: {img_file} yüklenemedi -> {e}")

        print(f"  -> {count} görüntü")

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(filenames),
    )


def merge_rare_classes(y, classes):
    """
    Evre 3 ve Evre 4'ü tek sınıfta birleştirir.

    Evre 4 ~30 örnekle istatistiksel olarak ölçülemeyecek kadar küçük.
    Birleştirme hem klinik olarak savunulabilir (her ikisi de tam kat doku
    kaybı) hem de modele daha sağlam bir öğrenme sinyali verir.

    Returns:
        y_new: Yeniden haritalanmış etiketler
        classes_new: Yeni sınıf isimleri listesi
    """
    if "Evre 3" not in classes or "Evre 4" not in classes:
        print("UYARI: Evre 3 / Evre 4 bulunamadı, birleştirme atlandı.")
        return y, classes

    idx_3 = classes.index("Evre 3")
    idx_4 = classes.index("Evre 4")

    # Yeni sınıf listesi: Evre 4 çıkarılır, Evre 3 yeniden adlandırılır
    classes_new = [c for i, c in enumerate(classes) if i != idx_4]
    classes_new[classes_new.index("Evre 3")] = "Evre 3-4"

    # Eski indeksten yeni indekse harita
    old_to_new = {}
    for old_idx, old_name in enumerate(classes):
        if old_idx == idx_4:
            target_name = "Evre 3-4"
        elif old_idx == idx_3:
            target_name = "Evre 3-4"
        else:
            target_name = old_name
        old_to_new[old_idx] = classes_new.index(target_name)

    y_new = np.array([old_to_new[int(v)] for v in y], dtype=np.int32)

    print(f"Sınıflar birleştirildi: {len(classes)} -> {len(classes_new)}")
    print(f"Yeni sınıflar: {classes_new}")

    return y_new, classes_new


# -------------------- MODEL --------------------
def create_model(input_shape, num_classes):
    """
    MobileNetV2 tabanlı transfer learning modeli oluşturur.

    İlk aşamada backbone tamamen dondurulur, yalnızca yeni eklenen
    sınıflandırma katmanları eğitilir.

    Returns:
        model: Tam sınıflandırma modeli
        base_model: Fine-tuning sırasında kısmen açılacak MobileNetV2
    """
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="classifier_dense",
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="stage_output")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_wound_stage_model")
    return model, base_model


def unfreeze_model_partial(base_model, num_layers_to_unfreeze):
    """
    Fine-tuning için backbone'u KISMEN açar.

    BatchNormalization katmanları dondurulur            
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
    """
    Modeli verilen learning rate ile derler.

    Sparse_categorical_crossentropy + tam sayı etiketler kullanılıyor
    (one-hot değil)
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# -------------------- CALLBACK'LER --------------------
def get_callbacks(model_name, phase_name, patience):
    """Eğitimi izleyen ve en iyi modeli kaydeden callback'leri oluşturur."""
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
        callbacks.CSVLogger(
            os.path.join(
                RESULTS_DIR, f"{model_name}_{phase_name}_training_log.csv"
            )
        ),
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
    num_classes,
    model_name,
    enable_fine_tuning=True,
):
    """
    Modeli iki aşamada eğitir.

    Aşama 1: Backbone dondurulur, yalnızca sınıflandırma katmanları eğitilir.
    Aşama 2: Backbone'un son UNFREEZE_LAST_N_LAYERS katmanı açılır
             (BatchNorm hariç) ve LR birkaç epokta rampalanarak çok düşük
             bir tavana çıkarılır.
    """
    print("\n--- AŞAMA 1: Feature extraction (backbone dondurulmuş) ---")
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
        print("\nFine-tuning devre dışı bırakıldı.")
        return history_1, None

    print(
        f"\n--- AŞAMA 2: Kısmi fine-tuning "
        f"(son {UNFREEZE_LAST_N_LAYERS} katman açık, BatchNorm donuk, "
        f"{FT_WARMUP_EPOCHS} epokluk LR rampası) ---"
    )
    unfreeze_model_partial(base_model, UNFREEZE_LAST_N_LAYERS)


    compile_model(model, LR_FINE_TUNE / 20.0)

    phase1_epochs_completed = len(history_1.epoch)
    total_epochs = phase1_epochs_completed + FINE_TUNE_EPOCHS

    def ft_lr_schedule(epoch, current_lr):
        step = epoch - phase1_epochs_completed
        if step < FT_WARMUP_EPOCHS:
            return LR_FINE_TUNE * (step + 1) / FT_WARMUP_EPOCHS
        return current_lr

    lr_ramp_callback = callbacks.LearningRateScheduler(ft_lr_schedule, verbose=1)

    history_2 = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=total_epochs,
        initial_epoch=phase1_epochs_completed,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[lr_ramp_callback] + get_callbacks(
            model_name, "phase2", PATIENCE_PHASE2
        ),
        verbose=1,
    )

    # Fine-tuning modeli kötüleştirdiyse Aşama 1'e geri dön
    best_p1 = min(history_1.history["val_loss"])
    best_p2 = min(history_2.history["val_loss"])

    if best_p2 > best_p1:
        print(
            f"\nUYARI: Fine-tuning sonrası en iyi doğrulama kaybı ({best_p2:.4f}), "
            f"Aşama 1'deki değerden ({best_p1:.4f}) daha kötü. "
            "Aşama 1 checkpoint'i geri yükleniyor."
        )
        model.load_weights(f"{model_name}_phase1_best.keras")

    return history_1, history_2


# -------------------- PRIOR ÖLÇEKLEME --------------------
def tune_class_priors(model, X_val, y_val, num_classes, n_iter=300, seed=42):
    """
    Softmax çıktılarını ölçekleyen katsayıları VALIDATION setinde arar.

    Azınlık sınıflarının karar sınırını kaydırır - modeli yeniden eğitmeden
    macro-F1'i iyileştirebilir. Rastgele arama kullanılır çünkü 5-6 sınıfta
    tam ızgara arama kombinatoryal olarak çok büyür.

    ÖNEMLİ: Katsayılar yalnızca validation'da seçilir, test setinde
    yalnızca bir kez uygulanır. Aksi halde bulunan iyileşme sahte olur.
    """
    rng = np.random.default_rng(seed)
    val_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

    base_score = f1_score(
        y_val, np.argmax(val_probs, axis=1), average="macro", zero_division=0
    )
    best_prior = np.ones(num_classes)
    best_score = base_score

    for _ in range(n_iter):
        prior = rng.uniform(0.6, 2.5, size=num_classes)
        preds = np.argmax(val_probs * prior, axis=1)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score, best_prior = score, prior

    print(f"\nPrior ölçekleme (validation macro-F1):")
    print(f"  Ölçeklemesiz : {base_score:.4f}")
    print(f"  Ölçeklemeli  : {best_score:.4f}")
    print(f"  En iyi prior : {np.round(best_prior, 3).tolist()}")

    if best_score <= base_score + 1e-6:
        print("  -> İyileşme yok, ölçekleme uygulanmayacak.")
        return np.ones(num_classes)

    # Validation seti küçükse bulunan katsayı gürültüye uymuş olabilir.
    if len(y_val) < 100:
        print(
            f"  UYARI: Validation seti küçük ({len(y_val)} örnek). "
            "Bu katsayılar gürültüye uymuş olabilir; test sonucunu "
            "temkinli değerlendirin."
        )

    return best_prior


# -------------------- GÖRSELLEŞTİRME --------------------
def visualize_results(history_1, history_2, y_true, y_pred, classes, model_name):
    """Eğitim/doğrulama grafikleri ve confusion matrix oluşturur."""
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
    curves_path = os.path.join(
        RESULTS_DIR, f"{model_name}_training_curves.png"
    )
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix: hem yüzde hem ham sayı gösterilir.
    # Ham sayı kritik - az örnekli sınıflarda yüzde yanıltıcıdır.
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm_raw, np.where(row_sums == 0, 1, row_sums), dtype=float
    ) * 100

    annot = np.empty_like(cm_raw, dtype=object)
    for i in range(cm_raw.shape[0]):
        for j in range(cm_raw.shape[1]):
            annot[i, j] = f"{cm_pct[i, j]:.1f}%\n(n={cm_raw[i, j]})"

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 9},
    )
    plt.title("Tahmin Karmaşıklık Matrisi (% ve ham sayı)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    cm_path = os.path.join(
        RESULTS_DIR, f"{model_name}_confusion_matrix.png"
    )
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nEğitim grafiği kaydedildi -> {curves_path}")
    print(f"Karmaşıklık matrisi kaydedildi -> {cm_path}")


# -------------------- DEĞERLENDİRME --------------------
def evaluate_model(
    model, X_test, y_test, f_test, classes, model_name, prior=None
):
    """Modeli test verisi üzerinde değerlendirir ve sonuçları CSV'ye kaydeder."""
    num_classes = len(classes)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Doğruluğu (ölçeklemesiz): {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    if prior is not None and not np.allclose(prior, 1.0):
        y_prob_adj = y_prob * prior
        y_prob_adj = y_prob_adj / y_prob_adj.sum(axis=1, keepdims=True)
        y_pred_classes = np.argmax(y_prob_adj, axis=1)
        print("Prior ölçekleme uygulandı.")
    else:
        y_prob_adj = y_prob
        y_pred_classes = np.argmax(y_prob, axis=1)

    print("\nSınıflandırma Raporu (Test Seti):")
    print(
        classification_report(
            y_test,
            y_pred_classes,
            labels=range(num_classes),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    macro_f1 = f1_score(y_test, y_pred_classes, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")

    # Az örnekli sınıflar için uyarı
    test_counts = Counter(y_test.tolist())
    for idx, name in enumerate(classes):
        n = test_counts.get(idx, 0)
        if n < 10:
            print(
                f"UYARI: '{name}' test setinde yalnızca {n} örnek içeriyor. "
                "Bu sınıfın metrikleri istatistiksel olarak güvenilir değil."
            )

    results_dict = {
        "Dosya": f_test,
        "Gerçek": [classes[i] for i in y_test],
        "Tahmin": [classes[i] for i in y_pred_classes],
        "Dogru_mu": [
            "DOĞRU" if a == b else "YANLIŞ"
            for a, b in zip(y_test, y_pred_classes)
        ],
        "Guven": y_prob_adj.max(axis=1),
    }
    for i, class_name in enumerate(classes):
        safe_name = class_name.replace(" ", "_").replace("-", "_")
        results_dict[f"Olasilik_{safe_name}"] = y_prob_adj[:, i]

    results_df = pd.DataFrame(results_dict)
    csv_path = os.path.join(
        RESULTS_DIR, f"{model_name}_predictions.csv"
    )
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTahminler CSV'ye kaydedildi -> {csv_path}")

    return y_prob_adj, y_pred_classes, macro_f1


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
    DATA_DIR = "wound_detection_dataset/wound"
    CLASSES = [
        "Derin Doku Hasarı",
        "Evre 1",
        "Evre 2",
        "Evre 3",
        "Evre 4",
        "Evrelendirilemeyen",
    ]
    MODEL_NAME = f"wound_stage_model_{today}"

    ENABLE_FINE_TUNING = True
    ENABLE_PRIOR_TUNING = True

    # 1. Veri Yükleme
    print("\n[1/8] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)

    if len(X) == 0:
        raise RuntimeError(
            "Hiç görüntü yüklenemedi. DATA_DIR ve sınıf klasörlerini kontrol edin."
        )

    classes = list(CLASSES)
    print(f"\nToplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(classes, np.bincount(y, minlength=len(classes))))}")

    # 2. Az örnekli sınıfları birleştirme
    print("\n[2/8] Sınıf birleştirme kontrolü...")
    if MERGE_RARE_CLASSES:
        y, classes = merge_rare_classes(y, classes)
        print(
            f"Yeni dağılım: "
            f"{dict(zip(classes, np.bincount(y, minlength=len(classes))))}"
        )
    else:
        print("Birleştirme kapalı (MERGE_RARE_CLASSES=False).")

    num_classes = len(classes)

    # 4.Veri Bölme
    print("\n[3/8] Veri bölünüyor (%70 eğitim / %15 doğrulama / %15 test)...")
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, filenames, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")
    print(f"Eğitim dağılımı: {np.bincount(y_train, minlength=num_classes).tolist()}")
    print(f"Test dağılımı  : {np.bincount(y_test, minlength=num_classes).tolist()}")

    # 4. Sınıf Ağırlıkları (üst sınırlı)
    print("\n[4/8] Sınıf ağırlıkları hesaplanıyor...")
    classes_in_train = np.unique(y_train)
    raw_weights = compute_class_weight(
        class_weight="balanced", classes=classes_in_train, y=y_train
    )
   
    clipped = np.clip(raw_weights, None, MAX_CLASS_WEIGHT)
    class_weights = {int(c): float(w) for c, w in zip(classes_in_train, clipped)}

    for c, w in class_weights.items():
        print(f"  {classes[c]}: {w:.3f}")

    # 5. Model Oluşturma
    print("\n[5/8] MobileNetV2 modeli inşa ediliyor...")
    model, base_model = create_model((*IMG_SIZE, 3), num_classes)
    model.summary()

    # 6. İki Aşamalı Eğitim
    print("\n[6/8] Model eğitiliyor (Aşama 1 + Aşama 2)...")
    history_1, history_2 = train_two_phase(
        model,
        base_model,
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        num_classes,
        MODEL_NAME,
        enable_fine_tuning=ENABLE_FINE_TUNING,
    )

    # 7. Prior ölçekleme (validation üzerinde) + Değerlendirme (test üzerinde)
    print("\n[7/8] Performans değerlendiriliyor...")
    prior = None
    if ENABLE_PRIOR_TUNING:
        prior = tune_class_priors(model, X_val, y_val, num_classes)

    _, y_pred_classes, macro_f1 = evaluate_model(
        model, X_test, y_test, f_test, classes, MODEL_NAME, prior=prior
    )

    visualize_results(
        history_1, history_2, y_test, y_pred_classes, classes, MODEL_NAME
    )


    # Final modeli tarihle kaydet
    final_model_path = f"{MODEL_NAME}_final.keras"
    model.save(final_model_path)
    print(f"Final model kaydedildi -> {final_model_path}")

    # 8. TFLite Dönüşümü
    print("\n[8/8] Android için TFLite modeli oluşturuluyor...")
    tflite_path = convert_to_tflite(
        model, f"{MODEL_NAME}_final.tflite"
    )

    if tflite_path:
        print(f"\nAndroid'de kullanılmaya hazır model -> {tflite_path}")
        if prior is not None and not np.allclose(prior, 1.0):
            print(
                "NOT: Prior ölçekleme kullanıldı. Android tarafında da aynı "
                f"katsayıları uygulamanız gerekir: {np.round(prior, 4).tolist()}"
            )
    else:
        print("\nTFLite dönüşümü başarısız oldu.")


if __name__ == "__main__":
    main()