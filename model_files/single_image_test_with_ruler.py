import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ruler_utils import detect_ruler_and_scale

MODEL_PATH = "models/wound_model.keras"
IMAGE_PATH = "test_image.jpg"
OUTPUT_DIR = "single_test_output"
RESULT_TXT_PATH = os.path.join(OUTPUT_DIR, "measurement_results.txt")

IMG_HEIGHT = 256
IMG_WIDTH = 256



PRED_THRESHOLD = 0.6 #0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)




def load_image_for_model(path):
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    original_bgr = image_bgr.copy()
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    resized_rgb = cv2.resize(original_rgb, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized_rgb.astype(np.float32) / 255.0

    return original_bgr, original_rgb, normalized


# YARA ÖLÇÜM

def calculate_area_from_mask(mask_binary):
    return int(np.sum(mask_binary > 0))

def calculate_main_contour(mask_binary):
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def contour_bbox(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def save_results_to_txt(path, image_name, mm_per_pixel, area_px, width_px, height_px, area_mm2, width_mm, height_mm):

    with open(path, "w", encoding="utf-8") as f:

        f.write("===== YARA ÖLÇÜM SONUCU =====\n\n")

        f.write(f"Görüntü: {image_name}\n")
        f.write(f"Ölçek: 1 pixel = {mm_per_pixel:.5f} mm\n\n")

        f.write("--- SONUÇLAR ---\n")
        f.write(f"Alan (pixel): {area_px}\n")
        f.write(f"Genişlik (pixel): {width_px}\n")
        f.write(f"Yükseklik (pixel): {height_px}\n\n")

        f.write(f"Alan (mm²): {area_mm2:.2f}\n")
        f.write(f"Genişlik (mm): {width_mm:.2f}\n")
        f.write(f"Yükseklik (mm): {height_mm:.2f}\n")




model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "dice_loss": dice_loss,
        "dice_coef": dice_coef
    }
)

print("Model yüklendi.")


original_bgr, original_rgb, input_image = load_image_for_model(IMAGE_PATH)


# YARA MASKESİ
pred = model.predict(np.expand_dims(input_image, axis=0), verbose=0)[0]
raw_pred = pred.squeeze()

raw_pred_img = (raw_pred * 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUTPUT_DIR, "raw_prediction.png"), raw_pred_img)

pred_mask_small = (raw_pred > PRED_THRESHOLD).astype(np.uint8)

orig_h, orig_w = original_rgb.shape[:2]
pred_mask = cv2.resize(pred_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "predicted_mask.png"),
    (pred_mask * 255).astype(np.uint8)
)


# CETVELDEN ÖLÇEK HESABI
mm_per_pixel, debug_info = detect_ruler_and_scale(original_bgr, debug=True)

if mm_per_pixel is None:
    print("Cetvel tespiti başarısız:", debug_info.get("error", "bilinmeyen hata"))
    if "blue_mask" in debug_info:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "ruler_blue_mask.png"), debug_info["blue_mask"])
    if "threshold" in debug_info:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "ruler_threshold.png"), debug_info["threshold"])
    raise SystemExit

print(f"1 pixel = {mm_per_pixel:.5f} mm")

# debug görselleri kaydet
if "blue_mask" in debug_info:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ruler_blue_mask.png"), debug_info["blue_mask"])
if "threshold" in debug_info:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ruler_threshold.png"), debug_info["threshold"])
if "line_vis" in debug_info:
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ruler_lines.png"), debug_info["line_vis"])


# YARA BOYUT HESABI
area_px = calculate_area_from_mask(pred_mask)
wound_cnt = calculate_main_contour(pred_mask)

if wound_cnt is None:
    print("Yara bulunamadı veya maske boş.")
    raise SystemExit

x, y, w, h = contour_bbox(wound_cnt)

width_mm = w * mm_per_pixel
height_mm = h * mm_per_pixel
area_mm2 = area_px * (mm_per_pixel ** 2)

print("\n--- SONUÇLAR ---")
print(f"Alan (pixel): {area_px}")
print(f"Genişlik (pixel): {w}")
print(f"Yükseklik (pixel): {h}")
print(f"Alan (mm²): {area_mm2:.2f}")
print(f"Genişlik (mm): {width_mm:.2f}")
print(f"Yükseklik (mm): {height_mm:.2f}")



overlay = original_rgb.copy()

# Yara konturu ve bbox
cv2.drawContours(overlay, [wound_cnt], -1, (0, 255, 0), 2)
cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 2)


if "ruler_box" in debug_info:
    rx, ry, rw, rh = debug_info["ruler_box"]
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)


red_overlay = original_rgb.copy()
red_overlay[pred_mask == 1] = [255, 0, 0]
blended = cv2.addWeighted(original_rgb, 0.7, red_overlay, 0.3, 0)

cv2.drawContours(blended, [wound_cnt], -1, (0, 255, 0), 2)
cv2.rectangle(blended, (x, y), (x + w, y + h), (255, 255, 255), 2)

if "ruler_box" in debug_info:
    cv2.rectangle(blended, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)

text1 = f"Area: {area_mm2:.1f} mm2"
text2 = f"W: {width_mm:.1f} mm | H: {height_mm:.1f} mm"
text3 = f"1 px = {mm_per_pixel:.4f} mm"

cv2.putText(blended, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(blended, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(blended, text3, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "result_overlay.png"),
    cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
)

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(original_rgb)
plt.title("Orijinal")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(pred_mask, cmap="gray")
plt.title("Tahmin Maske")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(debug_info["blue_mask"], cmap="gray")
plt.title("Cetvel Maskesi")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(blended)
plt.title("Ölçüm Sonucu")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "summary.png"))
plt.close()

print(f"\nÇıktılar kaydedildi: {OUTPUT_DIR}")