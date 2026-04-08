import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4
EPOCHS = 120

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"

MODEL_SAVE_PATH = "models/wound_model.keras"
PRED_MASK_DIR = "outputs/predicted_masks"
OVERLAY_DIR = "outputs/overlays"
PLOT_DIR = "outputs/plots"
# Eğer klasör zaten varsa, içindeki her şeyle birlikte sil
if os.path.exists(OVERLAY_DIR):
    shutil.rmtree(OVERLAY_DIR)
if os.path.exists(PRED_MASK_DIR):
    shutil.rmtree(PRED_MASK_DIR)


os.makedirs("models", exist_ok=True)
os.makedirs(PRED_MASK_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype(np.float32) / 255.0
    return image

def load_mask(path):
    # 1. Maskeyi renkli oku ve RGB'ye çevir (GRAYSCALE DEĞİL!)
    mask = cv2.imread(path, cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Maske okunamadı: {path}")
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    # 2. SADECE YARA RENGİNİ FİLTRELE
    # Eğer CVAT yarayı Kırmızı tonlarında kaydettiyse:
    # Kırmızının yoğun, yeşil ve mavinin az olduğu pikselleri seç.
    lower_red = np.array([100, 0, 0])   # Alt kırmızı sınırı
    upper_red = np.array([255, 50, 50]) # Üst kırmızı sınırı
    
    # Sadece bu renk aralığındaki pikselleri beyaz (255), diğerlerini siyah (0) yap
    binary_mask = cv2.inRange(mask, lower_red, upper_red)
    
    # 3. Modelin anlayacağı [0, 1] formatına çevir
    binary_mask = (binary_mask > 0).astype(np.float32)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    
    return binary_mask
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*")))
mask_paths = sorted(glob(os.path.join(MASK_DIR, "*")))

print("Toplam görüntü:", len(image_paths))
print("Toplam maske:", len(mask_paths))

assert len(image_paths) > 0, "Hiç görüntü bulunamadı."
assert len(image_paths) == len(mask_paths), "Görüntü ve maske sayısı eşit değil."

X = np.array([load_image(p) for p in image_paths], dtype=np.float32)
Y = np.array([load_mask(p) for p in mask_paths], dtype=np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


X_train, X_test, Y_train, Y_test, train_paths, test_paths = train_test_split(
    X, Y, image_paths, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, Y_train.shape)
print("Test:", X_test.shape, Y_test.shape)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# MİNİ U-NET
def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 32)
    s3, p3 = encoder_block(p2, 64)

    b1 = conv_block(p3, 128)

    d1 = decoder_block(b1, s3, 64)
    d2 = decoder_block(d1, s2, 32)
    d3 = decoder_block(d2, s1, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d3)

    return models.Model(inputs, outputs, name="MiniUNet")

model = build_unet((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,
    metrics=[dice_coef]
)

print(model.summary())


# EĞİTİM
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)



model.save(MODEL_SAVE_PATH)
print(f"Model kaydedildi: {MODEL_SAVE_PATH}")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["dice_coef"], label="train_dice")
plt.plot(history.history["val_dice_coef"], label="val_dice")
plt.legend()
plt.title("Dice")

plot_path = os.path.join(PLOT_DIR, "training_plot.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Grafik kaydedildi: {plot_path}")


preds = model.predict(X_test, verbose=1)

# Binary mask
pred_masks = (preds > 0.6).astype(np.uint8)

from sklearn.metrics import classification_report, confusion_matrix

# Gerçek ve tahmin maskelerini 1 boyuta indir
y_true_flat = Y_test.flatten().astype(np.uint8)
y_pred_flat = pred_masks.flatten().astype(np.uint8)

# Dice hesaplama fonksiyonu (numpy ile)
def dice_score_np(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)

    return (2.0 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )


dice_scores = []

for i in range(len(pred_masks)):
    dice = dice_score_np(Y_test[i], pred_masks[i])
    dice_scores.append(dice)

print("\n=== TEST DICE SCORES ===")

for i, d in enumerate(dice_scores):
    print(f"{os.path.basename(test_paths[i])} -> Dice: {d:.4f}")

print("\nAverage Dice Score:", np.mean(dice_scores))
print("Median Dice Score:", np.median(dice_scores))



print("\n=== PIXEL-LEVEL CLASSIFICATION REPORT ===")
print(classification_report(
    y_true_flat,
    y_pred_flat,
    target_names=["Background", "Wound"],
    digits=4
))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true_flat, y_pred_flat))


for i, pred_mask in enumerate(pred_masks):
    base_name = os.path.splitext(os.path.basename(test_paths[i]))[0]

    # Tahmin maskesini kaydet
    mask_to_save = (pred_mask.squeeze() * 255).astype(np.uint8)
    mask_path = os.path.join(PRED_MASK_DIR, f"{base_name}.png")
    cv2.imwrite(mask_path, mask_to_save)

    # Overlay oluştur
    image_rgb = (X_test[i] * 255).astype(np.uint8).copy()
    overlay = image_rgb.copy()

    # Yara bölgesini kırmızı göster
    overlay[pred_mask.squeeze() == 1] = [255, 0, 0]

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_path = os.path.join(OVERLAY_DIR, f"{base_name}.png")
    cv2.imwrite(overlay_path, overlay_bgr)

    print(f"{base_name} için maske ve overlay kaydedildi.")

print("\nTahmin maskeleri ve overlay dosyaları kaydedildi.")
print("İşlem tamamlandı.")