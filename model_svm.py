# Import Library

# Data manipulation & visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Preprocessing & feature engineering
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Model saving
import joblib

# Data splitting
from sklearn.model_selection import train_test_split

# Support Vector Machine (utama)
from sklearn.svm import SVC

# Algoritma pembanding (opsional)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluasi Model Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

import gdown

# Download Dataset dari Google Drive (Dataset Jurnal 2)
file_id = "1fc6Lw8Hh2JVQ9wgtYv6olkmG2ZKei9VR"
url = f"https://drive.google.com/uc?id={file_id}"
output = "Dataset_Diabetes.csv"

gdown.download(url, output, quiet=False)

# Load Dataset & Tampilkan Data Awal
df = pd.read_csv("Dataset_Diabetes.csv")
df.head(5)

# Cek Missing Values pada Dataset
df.isnull().sum()

# Cek Baris & Kolom
df.shape

# Cek Data Duplikat Dataset
df.duplicated().sum()

# Hapus Data Duplikat dari Dataset
df.drop_duplicates(inplace=True)

# Cek Baris & Kolom Stlh Cek Duplikasi
df.shape

# Cek Tipedata per Kolom
df.dtypes

# Memisahkan Fitur Numerik (scaling/normalisasi) dan Kategorikal (encoding)
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Visualisasi Distribusi Fitur Numerik
plt.figure(figsize=(14, 10))
for feature in numeric_features:
  plt.figure(figsize=(8, 6))
  sns.histplot(x=df[feature])
  plt.title(f"Distribution of {feature}")
  plt.show()
plt.tight_layout()
plt.show()

# Visualisasi Distribusi Fitur Kategorikal
plt.figure(figsize=(14, 10))
for feature in categorical_features:
  plt.figure(figsize=(8, 6))
  sns.histplot(x=df[feature])
  plt.title(f"Distribution of {feature}")
  plt.show()
plt.tight_layout()
plt.show()

# Normalisasi Fitur Numerik dengan MinMaxScaler
minmmax_scaler = MinMaxScaler()
df[numeric_features] = minmmax_scaler.fit_transform(df[numeric_features])
df.head()

# Encoding Fitur Kategorikal dengan LabelEncoder
encoder = LabelEncoder()
df[categorical_features] = df[categorical_features].apply(encoder.fit_transform)
df.head()

# Memisahkan Fitur (X) dan Label/Target (y)
x = df.drop("class", axis=1)
y = df["class"]

features = x.columns

# Membagi Dataset menjadi Data Latih dan Data Uji
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training Model SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(x_train, y_train)

# Prediksi Data Uji dan Evaluasi Model SVM
y_pred = svm.predict(x_test)

accuration = accuracy_score(y_test, y_pred)
print("Accuracy:", accuration)
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Mengukur Importance Fitur dengan Permutation Importance
from sklearn.inspection import permutation_importance
perm = permutation_importance(svm, x_test, y_test, n_repeats=10, random_state=42)

# Menampilkan Hasil Feature Importance SVM
import pandas as pd
feat_imp = pd.DataFrame({
    "Fitur": features,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

print(feat_imp)

# Menyimpan Model SVM dan Objek Preprocessing
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

scaler = {}
for col in numeric_features:
    sca = MinMaxScaler()
    df[col] = sca.fit_transform(df[[col]]).ravel()
    scaler[col] = sca

joblib.dump({
    "model": svm,
    "scaler": scaler,
    "encoders": encoders,
    "obj_cols": categorical_features,
    "num_cols": numeric_features,
    "feature_order": list(x.columns)
}, "svm_model.pkl")

print("Model SVM dan preprocessing berhasil disimpan!")

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load artefak model
saved = joblib.load("svm_model.pkl")
svm = saved["model"]
scalers = saved["scaler"]        # dict {col: MinMaxScaler}
encoders = saved["encoders"]      # dict {col: LabelEncoder}
obj_cols_tr = saved.get("obj_cols")
num_cols_tr = saved.get("num_cols")
feature_order = saved.get("feature_order")

# 2. Data baru
data_baru = pd.DataFrame([{
    "Age": 42,
    "Gender": "Male",
    "Polyuria": "No",
    "Polydipsia": "No",
    "sudden weight loss": "No",
    "weakness": "No",
    "Polyphagia": "No",
    "Genital thrush": "No",
    "visual blurring": "No",
    "Itching": "No",
    "Irritability": "No",
    "delayed healing": "No",
    "partial paresis": "No",
    "muscle stiffness": "No",
    "Alopecia": "No",
    "Obesity": "No"
}])

# 3. Bersihkan feature_order (hapus 'class')
if feature_order is not None:
    feature_order = [f for f in feature_order if f != "class"]

# 4. Tentukan kolom kategorikal & numerik
if obj_cols_tr is None and isinstance(encoders, dict):
    obj_cols_tr = list(encoders.keys())
if num_cols_tr is None:
    num_cols_tr = data_baru.select_dtypes(include=[np.number]).columns.tolist()

# 5. Encode kolom kategorikal
def safe_add_unseen_and_transform(le: LabelEncoder, s: pd.Series):
    vals = s.astype(str)
    unseen = sorted(set(vals.unique()) - set(le.classes_))
    if unseen:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=object)])
    return le.transform(vals)

if isinstance(encoders, dict):
    for col in obj_cols_tr:
        if col in data_baru.columns:
            data_baru[col] = safe_add_unseen_and_transform(encoders[col], data_baru[col])

# 6. Scale kolom numerik (per kolom)
for col in num_cols_tr:
    if col in scalers:
        data_baru[col] = scalers[col].transform(data_baru[[col]])

# 7. Urutkan fitur sesuai training
if feature_order:
    final_cols = [c for c in feature_order if c in data_baru.columns]
    data_baru = data_baru[final_cols]

# 8. Prediksi dan tampilkan hasil class saja
pred = svm.predict(data_baru)
print(f"class predict : {pred[0]}")
