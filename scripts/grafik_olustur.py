import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV dosyasını oku
df = pd.read_csv("../model_files/Basinc_yarasi_evreleri_predictions.csv")

# Doğruluk hesapla
df["Doğru"] = df["Gerçek"] == df["Tahmin"]

# Sınıf bazlı istatistikleri çıkar
results = df.groupby("Gerçek")["Doğru"].agg(
    total="count",
    correct="sum"
)
results["wrong"] = results["total"] - results["correct"]
results["correct_pct"] = (results["correct"] / results["total"]) * 100
results["wrong_pct"] = (results["wrong"] / results["total"]) * 100

# Grafik çiz
plt.figure(figsize=(8, 5))
plt.bar(results.index, results["correct_pct"], label="Doğru Tahmin (%)", color="green")
plt.bar(results.index, results["wrong_pct"], bottom=results["correct_pct"], label="Yanlış Tahmin (%)", color="red")

plt.title("Sınıf Bazlı Tahmin Performansı")
plt.ylabel("Yüzde (%)")
plt.legend()

# Klasör yoksa oluştur
os.makedirs("analysis_results", exist_ok=True)

# PNG olarak kaydet
output_path = "../analysis_results/evre_grafik.png"
plt.savefig(output_path, bbox_inches="tight")

print(f"✅ Grafik başarıyla kaydedildi: {output_path}")