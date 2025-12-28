import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CSV dosyasını oku
df = pd.read_csv("../model_files/yara_var_mi_yok_mu1_model_predictions.csv")

y_true = df["Gerçek"]
y_pred = df["Tahmin"]

labels=["Yara_Var", "Yara_Yok"]

# Karışıklık matrisini hesapla
cm = confusion_matrix(y_true, y_pred,labels=labels, normalize='true')*100

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.title("Tahmin Karmaşıklık Matrisi (%)")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")

print("✅ Karışıklık matrisi başarıyla oluşturuldu.")
# PNG olarak kaydet
output_path = "../analysis_results/yara_var_mi_yok_mu1.png"
plt.savefig(output_path, bbox_inches="tight")