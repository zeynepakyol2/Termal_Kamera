import cv2
import numpy as np


def group_close_values(values, max_gap=8):
    if len(values) == 0:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for v in values[1:]:
        if abs(v - groups[-1][-1]) <= max_gap:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [int(np.mean(g)) for g in groups]


def detect_ruler_and_scale(image_bgr, debug=False):
    debug_dict = {}

    # 1) Mavi cetveli bul
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([85, 30, 20])
    upper_blue = np.array([145, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"error": "Cetvel bulunamadı", "blue_mask": blue_mask}

    ruler_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(ruler_cnt)

    ruler_crop = image_bgr[y:y+h, x:x+w].copy()
    debug_dict["blue_mask"] = blue_mask
    debug_dict["ruler_box"] = (x, y, w, h)
    debug_dict["ruler_crop"] = ruler_crop.copy()

    # 2) Cetvel crop içinde işaretleri ayır
    gray = cv2.cvtColor(ruler_crop, cv2.COLOR_BGR2GRAY)

    # Histogram equalization biraz kontrast artırır
    gray = cv2.equalizeHist(gray)

    # Cetvel üzerindeki koyu işaretler beyaz olacak 
    _, th = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Gürültü azalt
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    debug_dict["threshold"] = th

    # 3) Sütun yoğunluğu analizi
    col_sum = np.sum(th > 0, axis=0)

    # Çok kısa gürültüleri atmak için eşik
    # Cetvel yüksekliğinin belli bir oranından fazla olan sütunları al
    min_col_strength = max(2, int(h * 0.10))

    candidate_xs = np.where(col_sum >= min_col_strength)[0].tolist()
    debug_dict["candidate_xs"] = candidate_xs
    debug_dict["col_sum"] = col_sum.tolist()

    if len(candidate_xs) < 2:
        return None, {**debug_dict, "error": "Cetvel işaretleri bulunamadı"}

    # Yakın sütunları grupla
    tick_xs = group_close_values(candidate_xs, max_gap=8)
    debug_dict["grouped_tick_xs"] = tick_xs

    if len(tick_xs) < 2:
        return None, {**debug_dict, "error": "Yeterli cetvel işareti yok"}

    # 4) Ardışık işaret aralıkları
    distances = [tick_xs[i+1] - tick_xs[i] for i in range(len(tick_xs)-1)]
    distances = [d for d in distances if d > 2]
    debug_dict["distances"] = distances

    if len(distances) == 0:
        return None, {**debug_dict, "error": "İşaret aralıkları hesaplanamadı"}

    # En sık küçük aralık ~ 1 mm kabul
    pixel_per_mm = float(np.median(distances))
    mm_per_pixel = 1.0 / pixel_per_mm

    # Görsel debug
    line_vis = ruler_crop.copy()
    for tx in tick_xs:
        cv2.line(line_vis, (tx, 0), (tx, h - 1), (0, 255, 0), 1)

    debug_dict["line_vis"] = line_vis
    debug_dict["pixel_per_mm"] = pixel_per_mm
    debug_dict["mm_per_pixel"] = mm_per_pixel

    return mm_per_pixel, debug_dict