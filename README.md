# 🛵 Zomato Delivery Time Prediction

## 📌 Project Overview
Project ini bertujuan untuk memprediksi **estimasi waktu pengiriman makanan (ETA)** secara akurat untuk layanan *food delivery* seperti Zomato.

Ketidakpastian waktu pengiriman adalah salah satu penyebab utama ketidakpuasan pelanggan. Dengan menggunakan Machine Learning, kita dapat memprediksi durasi pengiriman (dalam menit) berdasarkan faktor-faktor seperti:
* **Jarak Pengiriman** (dihitung menggunakan Haversine Formula).
* **Rating Kurir** (Kinerja masa lalu).
* **Kondisi Cuaca & Lalu Lintas**.
* **Waktu Masak (Prep Time)**.

## 🚀 Live Demo
Coba aplikasi prediksi ini secara langsung di sini:
👉 **[Zomato Delivery Predictor](https://zomato-delivery-prediction-eriel.streamlit.app/)**

## 📂 Dataset
Dataset yang digunakan adalah **Zomato Delivery Operations Dataset** yang mencakup 45.000+ baris data transaksi pengiriman.
* **Cleaning:** Penanganan 1.700+ missing values pada jam pemesanan.
* **Feature Engineering:** Pembuatan fitur baru `distance_km` (Jarak Fisik) dan `prep_time_min` (Waktu Masak).

## 🛠️ Tech Stack
* **Python** (Bahasa Pemrograman Utama)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Machine Learning Modeling)
* **Matplotlib & Seaborn** (Data Visualization)
* **Streamlit** (Web App Deployment)

## 📊 Model Performance
Setelah membandingkan Linear Regression, Gradient Boosting, dan Random Forest, model terbaik yang dipilih adalah:

* **Champion Model:** Random Forest Regressor
* **R2 Score:** ~75%
* **MAE (Mean Absolute Error):** ~4 Menit

### 🔍 Key Insights (Feature Importance):
1.  **Rating Kurir** adalah faktor #1 yang mempengaruhi kecepatan pengiriman.
2.  **Jarak Fisik** tetap menjadi batasan utama.
3.  **Prep Time** (Efisiensi Restoran) memiliki dampak signifikan yang sering terabaikan.

## 💻 How to Run Locally (Instalasi Lokal)
Jika Anda ingin menjalankan proyek ini di laptop Anda:

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/USERNAME_ANDA/Zomato-Delivery-Prediction.git](https://github.com/USERNAME_ANDA/Zomato-Delivery-Prediction.git)
    cd Zomato-Delivery-Prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

## 👤 Author
**Eriel Setiawan Dewantoro**
* [LinkedIn](https://www.linkedin.com/in/eriel-setiawan-dewantoro)

---
