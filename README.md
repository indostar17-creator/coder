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

# Kode untuk membuat memprediksi akurasi churn dan k means

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. pemmbersihan data mentah
(
df = pd.read_csv("data_siswa_pintar_bersama_Training (3).csv")
df = df.dropna()
df = df[(df['waktu_modul'] >= 0) & (df['durasi_klik'] >= 0)]
Q1 = df['durasi_klik'].quantile(0.25)
Q3 = df['durasi_klik'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df['durasi_klik'] <= upper_bound].copy()

# 2. K-MEANS CLUSTERING (Bagi 3 tipe siswa)
fitur = ['durasi_klik', 'waktu_modul', 'skor_kuis', 'login_frekuensi']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[fitur])

kmeans = KMeans(n_clusters=3, random_state=42)
df['klaster'] = kmeans.fit_predict(X_scaled)

# 3. NENTUIN TARGET CHURN (Klaster 1 itu siswa pasif/nilai anjlok)
df['churn'] = (df['klaster'] == 1).astype(int)

# 4. ANN & SMOTE PIPELINE
X = df[['durasi_klik', 'waktu_modul', 'ulang_video', 'skor_kuis', 'login_frekuensi']]
y = df['churn']

# Split data 80% belajar, 20% ujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pake SMOTE biar adil datanya
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Scale data ANN
X_train_sm_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# Otak ANN (Hidden layer: 64, 32, 16 neuron)
ann = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42)
ann.fit(X_train_sm_scaled, y_train_sm)

# Test akurasi
prediksi = ann.predict(X_test_scaled)
print("=== HASIL PREDIKSI ANN ===")
print(f"Akurasi: {accuracy_score(y_test, prediksi)*100:.2f}%\n")
print(classification_report(y_test, prediksi))
