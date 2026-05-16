# Prediksi Churn Siswa - Pintar-Bersama EdTech

Repositori ini berisi rancangan awal dan proses analisis data untuk memprediksi probabilitas siswa berhenti menggunakan platform pendidikan Pintar-Bersama (churn), serta mengelompokkan profil belajar mereka.

# Kode yang berisi data awal (kotor)

<img width="1141" height="514" alt="dataawal" src="https://github.com/user-attachments/assets/83c0623e-2155-4c79-89a0-61dea3e5168f" />

# Kode yang berisi data Bersih

<img width="715" height="614" alt="databersih" src="https://github.com/user-attachments/assets/a41cd7c1-4625-404e-b2cc-827fc0e0ae86" />

# Kode untuk membuat Visualisasi di py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

df = pd.read_csv("data_final.csv.xls")

PALETTE = {
    "quiz": "#4C72B0", "quiz_edge": "#2c4a7c",
    "login": "#DD8452", "login_edge": "#a05520",
    "avg_line": "#2ca02c",
}

plt.figure("Analisis Skor Kuis", figsize=(10, 6))
plt.hist(df["skor_kuis"], bins=10, color=PALETTE["quiz"], edgecolor=PALETTE["quiz_edge"], alpha=0.8)
overall_avg = df["skor_kuis"].mean()
plt.axvline(overall_avg, color=PALETTE["avg_line"], linestyle="--", linewidth=2, label=f"Rata-rata: {overall_avg:.2f}")
plt.title("Distribusi Skor Kuis Siswa", fontsize=14, fontweight="bold")
plt.xlabel("Rentang Skor Kuis")
plt.ylabel("Jumlah Siswa")
plt.legend()
plt.gca().spines[["top", "right"]].set_visible(False)

plt.figure("Analisis Login", figsize=(10, 6))
max_freq = int(df["login_frekuensi"].max())
bins_login = np.arange(0.5, max_freq + 1.5, 1)
plt.hist(df["login_frekuensi"], bins=bins_login, color=PALETTE["login"], edgecolor=PALETTE["login_edge"], linewidth=0.7, rwidth=0.85)
mean_freq = df["login_frekuensi"].mean()
plt.axvline(mean_freq, color=PALETTE["avg_line"], linestyle="--", label=f"Mean: {mean_freq:.2f}")
plt.title("Distribusi Frekuensi Login", fontsize=14, fontweight="bold")
plt.xlabel("Frekuensi Login")
plt.ylabel("Jumlah Siswa")
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.gca().spines[["top", "right"]].set_visible(False)

plt.show()
