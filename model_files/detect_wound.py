""" Yara var mı yok mu ? (TERMAL görüntüler için) """
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
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout,
                                      BatchNormalization, Conv2D, MaxPooling2D,
                                      Reshape, Multiply)
from focal_loss import SparseCategoricalFocalLoss

# -------------------- YAPILANDIRMA --------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 80                  # early stopping zaten daha erken durduracak, bu bir tavan
LEARNING_RATE = 5e-4
MIN_DELTA = 1e-3             # anlamlı bir iyileşme eşiği (eskisi çok küçüktü, gürültüyü "iyileşme" sayıyordu)
PATIENCE = 15                 # MIN_DELTA gerçekçi olduğu için patience'ı da kısalttık
L2_REG = 1e-4
FOCAL_GAMMA = 2.0            # sınıf dengesizliği için focal loss gücü
SE_REDUCTION = 8             # Squeeze-Excitation daralma oranı
WOUND_RECALL_BOOST = 1.3     # "wound" sınıfının recall'unu artırmak için ekstra ağırlık çarpanı
MIN_WOUND_RECALL = 0.75      # eşik seçerken "wound" recall'unun düşemeyeceği klinik minimum

# Tanı çıktıları için klasör
DIAG_DIR = "../analysis_results/diagnostics"

# -------------------- VERİ YÜKLEME --------------------


def load_and_preprocess_data(data_dir, classes):
    """
    Görüntüleri yükler ve normalizasyon uygular.

    Returns:
        tuple: (images, labels, filenames) - üçü de aynı sırada, birebir hizalı
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


def se_block(x, reduction=SE_REDUCTION):
    """
    Squeeze-and-Excitation bloğu: her kanalın ne kadar 'önemli' olduğunu öğrenir
    ve ona göre yeniden ölçekler. Thermal görüntülerde, ısı farkını taşıyan
    kanallara/bölgelere daha fazla ağırlık vermesine yardımcı olur.
    """
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(filters // reduction, 4), activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])


def create_custom_cnn(input_shape, num_classes):
    """
    ImageNet önyargısı taşımayan, sıfırdan eğitilen CNN.
    Termal görüntülerdeki ısı/doku örüntülerini doğrudan veriden öğrenir.
    SE-attention blokları ile hangi kanalların/bölgelerin önemli olduğunu öğrenir.
    500-2000 görüntü/sınıf için biraz daha derin bir kapasite hedeflendi.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        # Not: termal görüntülerde renk/parlaklık genelde SICAKLIK bilgisini taşır.
        # Bu yüzden RandomBrightness/RandomContrast BİLİNÇLİ OLARAK eklenmedi -
        # aksi halde model için anlamlı olan ısı farklarını bozabilir.
    ], name="data_augmentation")

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)

    def conv_block(x, filters, drop_rate):
        x = Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
        x = BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = se_block(x)
        x = MaxPooling2D()(x)
        x = Dropout(drop_rate)(x)
        return x

    x = conv_block(x, 32, 0.10)
    x = conv_block(x, 64, 0.15)
    x = conv_block(x, 128, 0.20)
    x = conv_block(x, 256, 0.25)
    x = conv_block(x, 512, 0.30)   # yeni: 5. blok, ek kapasite -> son Conv2D, Grad-CAM burayı kullanacak

    x = GlobalAveragePooling2D(name="last_conv_gap")(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, output, name="custom_thermal_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=SparseCategoricalFocalLoss(gamma=FOCAL_GAMMA),
        metrics=['accuracy']
    )
    return model


# -------------------- CALLBACK'LER --------------------


def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'{model_name}_best.keras',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        callbacks.CSVLogger(f'{model_name}_training_log.csv')
    ]


# -------------------- GÖRSELLEŞTİRME --------------------


def visualize_results(history, y_true, y_pred, CLASSES, y_all):
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
    plt.savefig("../analysis_results/detect_wound_accuracy_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    counts = np.bincount(y_all)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(CLASSES, counts)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  str(int(bar.get_height())), ha='center')
    plt.title("Veri Kümesindeki Görüntü Sayıları")
    plt.xlabel("Sınıf")
    plt.ylabel("Görüntü Sayısı")
    plt.savefig("../analysis_results/detect_wound_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Tahmin Karmaşıklık Matrisi (%)")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.savefig("../analysis_results/detect_wound_confusion_matrix.png", bbox_inches="tight")
    plt.show()


# -------------------- TANI ARAÇLARI --------------------


def save_class_sample_grid(X, y, filenames, CLASSES, n_per_class=8):
    """
    Her sınıftan rastgele örnekleri yan yana kaydeder.
    AMAÇ: Eğitimden önce siz de "wound" ve "no_wound" görüntülerine bakıp
    gerçekten ayırt edilebilir mi, kontrol edebilesiniz. Model bunu göremiyorsa
    büyük ihtimalle siz de göremezsiniz (tam olarak "kısmen" dediğiniz durum).
    """
    os.makedirs(DIAG_DIR, exist_ok=True)
    # görüntüleri [-1,1] aralığından [0,1] aralığına geri çeviriyoruz (mobilenet_v2 preprocess tersine)
    X_vis = (X + 1.0) / 2.0

    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(2 * n_per_class, 2 * len(CLASSES)))
    rng = np.random.default_rng(42)
    for class_idx, class_name in enumerate(CLASSES):
        idxs = np.where(y == class_idx)[0]
        chosen = rng.choice(idxs, size=min(n_per_class, len(idxs)), replace=False)
        for col, i in enumerate(chosen):
            ax = axes[class_idx, col] if len(CLASSES) > 1 else axes[col]
            ax.imshow(np.clip(X_vis[i], 0, 1))
            ax.set_title(class_name, fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{DIAG_DIR}/class_samples_preview.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Tanı: sınıf örnek grid'i kaydedildi -> {DIAG_DIR}/class_samples_preview.png")


def save_misclassified_samples(X_test, y_test, y_pred_classes, f_test, CLASSES, max_save=40):
    """
    Yanlış sınıflandırılan test görüntülerini gerçek/tahmin etiketiyle diske kaydeder.
    Bunlara bakarak: gerçekten hatalı mı, yoksa sınır durumu (ambiguous) mu, görebilirsiniz.
    """
    out_dir = f"{DIAG_DIR}/misclassified"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    wrong_idxs = np.where(y_test != y_pred_classes)[0][:max_save]
    for i in wrong_idxs:
        gercek = CLASSES[y_test[i]]
        tahmin = CLASSES[y_pred_classes[i]]
        src = f_test[i]
        dst_name = f"gercek_{gercek}__tahmin_{tahmin}__{os.path.basename(src)}"
        try:
            shutil.copy(src, os.path.join(out_dir, dst_name))
        except Exception as e:
            print(f"Kopyalanamadı: {src} - {e}")

    print(f"Tanı: {len(wrong_idxs)} yanlış sınıflandırılan görüntü kaydedildi -> {out_dir}")


def grad_cam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Modelin görüntünün hangi bölgesine bakarak karar verdiğini gösteren ısı haritası.
    Eğer model sürekli alakasız bölgelere (arka plan, çerçeve kenarı) bakıyorsa,
    bu genelde veri/etiket kalitesi sorununa işarettir.
    """
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array[np.newaxis, ...])
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_examples(model, X_test, y_test, f_test, CLASSES, last_conv_layer_name, n_examples=6):
    """Birkaç test örneği için Grad-CAM ısı haritasını orijinal görüntüyle yan yana kaydeder."""
    try:
        os.makedirs(DIAG_DIR, exist_ok=True)
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(X_test), size=min(n_examples, len(X_test)), replace=False)

        fig, axes = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
        for col, i in enumerate(idxs):
            img = X_test[i]
            heatmap = grad_cam(model, img, last_conv_layer_name)
            heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()

            img_vis = np.clip((img + 1.0) / 2.0, 0, 1)

            axes[0, col].imshow(img_vis)
            axes[0, col].set_title(f"Gerçek: {CLASSES[y_test[i]]}", fontsize=8)
            axes[0, col].axis('off')

            axes[1, col].imshow(img_vis)
            axes[1, col].imshow(heatmap_resized, cmap='jet', alpha=0.5)
            axes[1, col].set_title("Modelin baktığı yer", fontsize=8)
            axes[1, col].axis('off')

        plt.tight_layout()
        plt.savefig(f"{DIAG_DIR}/gradcam_examples.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Tanı: Grad-CAM örnekleri kaydedildi -> {DIAG_DIR}/gradcam_examples.png")
    except Exception as e:
        print(f"Grad-CAM oluşturulamadı (kritik değil, devam ediliyor): {e}")


# -------------------- TAHMİN (TEST-TIME AUGMENTATION) --------------------


def predict_tta(model, X, batch_size=32):
    """
    Test-Time Augmentation: orijinal + yatay çevrilmiş + dikey çevrilmiş görüntülerin
    tahminlerini ortalar. Eğitimi etkilemez, sadece tahmin anında kullanılır;
    genelde ekstra maliyetsiz 1-3 puanlık doğruluk artışı sağlar.
    """
    preds_orig = model.predict(X, batch_size=batch_size, verbose=0)
    preds_h = model.predict(X[:, :, ::-1, :], batch_size=batch_size, verbose=0)
    preds_v = model.predict(X[:, ::-1, :, :], batch_size=batch_size, verbose=0)
    return (preds_orig + preds_h + preds_v) / 3.0


def main():
    today = datetime.now().strftime("%Y-%m-%d")

    DATA_DIR = "wound_detection_dataset"
    CLASSES = ["wound", "no_wound"]   # index 0 = wound, index 1 = no_wound
    MODEL_NAME = f"wound_detection_model_{today}"

    # 1. Veri Yükleme
    print("\n[1/7] Veri yükleniyor...")
    X, y, filenames = load_and_preprocess_data(DATA_DIR, CLASSES)
    print(f"Toplam örnek sayısı: {len(X)}")
    print(f"Sınıf dağılımı: {dict(zip(CLASSES, np.bincount(y)))}")

    # 2. Veri Bölme (filenames'i de aynı split ile bölüyoruz ki hizalı kalsın)
    print("\n[2/7] Veri bölünüyor...")
    X_train, X_val_test, y_train, y_val_test, f_train, f_val_test = train_test_split(
        X, y, filenames,
        test_size=0.3,
        stratify=y,
        random_state=42
    )
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_val_test, y_val_test, f_val_test,
        test_size=0.5,
        stratify=y_val_test,
        random_state=42
    )

    print(f"Eğitim: {len(X_train)}, Doğrulama: {len(X_val)}, Test: {len(X_test)}")

    # 2.1 TANI: eğitimden önce sınıf örneklerine bakalım
    print("\n[2.1/7] Tanı: sınıf örnekleri kaydediliyor (eğitimden önce gözle kontrol için)...")
    save_class_sample_grid(X, y, filenames, CLASSES)

    # 3. Sınıf Ağırlıkları (focal loss ile birlikte hafif bir destek olarak kullanılıyor)
    print("\n[3/7] Sınıf ağırlıkları hesaplanıyor...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(class_weights)}
    # "wound" sınıfının recall'u precision'dan düşük çıkıyordu -> ekstra ağırlık veriyoruz
    class_weights[0] *= WOUND_RECALL_BOOST
    print("Sınıf Ağırlıkları (wound boost uygulanmış):", class_weights)

    # 4. Model Oluşturma ve Eğitim
    print("\n[4/7] Model oluşturuluyor (sıfırdan eğitilen termal-CNN)...")
    model = create_custom_cnn((*IMG_SIZE, 3), len(CLASSES))
    last_conv_layer_name = [l.name for l in model.layers if isinstance(l, Conv2D)][-1]

    print("\n[5/7] Model eğitiliyor...")
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(MODEL_NAME)
    )

    # 6. Değerlendirme
    print("\n[6/7] Performans değerlendiriliyor...")
    # NOT: focal loss özel bir kayıp fonksiyonu olduğu için custom_objects ile belirtilmeli,
    # aksi halde load_model hata verir.
    model = load_model(
        f'{MODEL_NAME}_best.keras',
        custom_objects={'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss}
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Doğruluğu: {test_acc:.4f}")
    print(f"Test Kaybı: {test_loss:.4f}")

    # ---- EŞİK OPTİMİZASYONU (validation setinde, TTA ile) ----
    # DİKKAT: y_pred[:, 0] = "wound" (sınıf 0) olma olasılığıdır.
    # Bu olasılık eşiği geçiyorsa örnek "wound" (0) olarak etiketlenmeli, "no_wound" (1) değil.
    #
    # ÖNEMLİ: Ham doğruluğu maksimize etmek, dengesiz veri setinde çoğunluk sınıfına
    # (no_wound) yaslanmayı ödüllendirir ve wound recall'unu düşürür. Bu yüzden önce
    # "wound recall >= MIN_WOUND_RECALL" şartını sağlayan eşikleri buluyoruz, sonra
    # bunlar arasından en yüksek doğruluğu vereni seçiyoruz.
    print("\nTTA (Test-Time Augmentation) ile doğrulama tahminleri üretiliyor...")
    val_pred = predict_tta(model, X_val)

    candidates = []
    for t in np.arange(0.3, 0.7, 0.02):
        preds = np.where(val_pred[:, 0] > t, 0, 1)   # > t ise sınıf 0 (wound)
        acc = accuracy_score(y_val, preds)
        wound_recall = recall_score(y_val, preds, pos_label=0)
        candidates.append((t, acc, wound_recall))

    qualifying = [c for c in candidates if c[2] >= MIN_WOUND_RECALL]
    if qualifying:
        best_thresh, best_acc, best_wound_recall = max(qualifying, key=lambda c: c[1])
        print(f"({len(qualifying)} eşik, wound recall >= {MIN_WOUND_RECALL:.2f} şartını sağladı)")
    else:
        # Hiçbir eşik şartı sağlamıyorsa: en azından en iyi wound recall'u veren eşiği seç
        best_thresh, best_acc, best_wound_recall = max(candidates, key=lambda c: c[2])
        print(f"UYARI: hiçbir eşik wound recall >= {MIN_WOUND_RECALL:.2f} şartını sağlamadı, "
              f"en iyi wound recall'lu eşik seçildi.")

    val_f1 = f1_score(y_val, np.where(val_pred[:, 0] > best_thresh, 0, 1), average='macro')
    print(f"\nEn iyi eşik (threshold): {best_thresh:.2f} "
          f"(Val Doğruluk: {best_acc:.4f}, Val Wound Recall: {best_wound_recall:.4f}, Val Macro F1: {val_f1:.4f})")

    print("\nTTA (Test-Time Augmentation) ile test tahminleri üretiliyor...")
    y_pred = predict_tta(model, X_test)
    y_pred_classes = np.where(y_pred[:, 0] > best_thresh, 0, 1)   # düzeltildi

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES, digits=4))

    tta_test_acc = accuracy_score(y_test, y_pred_classes)
    print(f"TTA + optimize edilmiş eşik ile Test Doğruluğu: {tta_test_acc:.4f}")
    print(f"(Not: yukarıdaki 'Test Doğruluğu: {test_acc:.4f}' TTA'sız, ham argmax ile hesaplanmıştı)")

    # Sonuçları Kaydet (artık f_test ile birebir hizalı)
    results_df = pd.DataFrame({
        'Dosya': f_test,
        'Gerçek': [CLASSES[i] for i in y_test],
        'Tahmin': [CLASSES[i] for i in y_pred_classes],
        'Yara_Var': y_pred[:, 0],
        'Yara_Yok': y_pred[:, 1],
    })
    results_df.to_csv(f'{MODEL_NAME}_predictions.csv', index=False)
    print("\nTahminler CSV'ye kaydedildi.")

    # 6.1 TANI: yanlış sınıflandırılan görüntüleri ve Grad-CAM ısı haritalarını kaydet
    print("\n[6.1/7] Tanı: yanlış sınıflandırılan örnekler ve Grad-CAM kaydediliyor...")
    save_misclassified_samples(X_test, y_test, y_pred_classes, f_test, CLASSES)
    save_gradcam_examples(model, X_test, y_test, f_test, CLASSES, last_conv_layer_name)

    # Görselleştirme
    visualize_results(history, y_test, y_pred_classes, CLASSES, y)

    # Modeli Kaydet
    model.save(f'{MODEL_NAME}_final.keras')
    print("\nFinal model kaydedildi.")


if __name__ == "__main__":
    main()
