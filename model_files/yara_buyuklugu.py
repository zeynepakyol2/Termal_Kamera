import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fotoğrafı yükle
img = cv2.imread("cetvel_yara.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)


edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# En uzun yatay çizgiyi bul 
longest = None
max_len = 0

for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.hypot(x2 - x1, y2 - y1)
    
   
    if abs(y1 - y2) < 5 and length > max_len:
        max_len = length
        longest = (x1, y1, x2, y2)


x1,y1,x2,y2 = longest
cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)


pixel_length = max_len


real_mm = 10  

mm_per_pixel = real_mm / pixel_length
print("1 pixel =", mm_per_pixel, "mm")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Cetvel Tespit")
plt.show()




_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cnt = max(contours, key=cv2.contourArea)


x,y,w,h = cv2.boundingRect(cnt)


width_mm = w * mm_per_pixel
height_mm = h * mm_per_pixel

print("Yara Genişlik:", width_mm, "mm")
print("Yara Yükseklik:", height_mm, "mm")

cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Yara Ölçümü")
plt.show()
