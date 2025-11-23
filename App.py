import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Zomato Delivery Predictor",
    page_icon="🛵",
    layout="wide"
)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Pastikan file .pkl ada di satu folder dengan app.py
        model = joblib.load('zomato_model_final.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- 3. UI DESIGN (SIDEBAR) ---
st.sidebar.header("🔧 Masukkan Parameter Pesanan")
st.sidebar.write("Ubah nilai di bawah untuk simulasi:")

def user_input_features():
    # Input Numerik
    age = st.sidebar.slider('Umur Kurir (Tahun)', 18, 65, 30)
    ratings = st.sidebar.slider('Rating Kurir (1-5)', 1.0, 5.0, 4.5, 0.1)
    distance = st.sidebar.number_input('Jarak Pengiriman (km)', 1.0, 30.0, 5.0)
    prep_time = st.sidebar.number_input('Waktu Masak (Menit)', 5, 60, 15)
    
    # Input Kategori (Dropdown)
    weather = st.sidebar.selectbox('Kondisi Cuaca', ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog', 'Cloudy'])
    traffic = st.sidebar.selectbox('Kepadatan Lalu Lintas', ['Low', 'Medium', 'High', 'Jam'])
    vehicle = st.sidebar.selectbox('Jenis Kendaraan', ['motorcycle', 'scooter', 'electric_scooter']) # Optional display
    city = st.sidebar.selectbox('Tipe Kota', ['Metropolitian', 'Urban', 'Semi-Urban'])
    festival = st.sidebar.selectbox('Sedang Festival?', ['No', 'Yes'])
    multiple = st.sidebar.selectbox('Jumlah Orderan Dibawa', [0, 1, 2, 3])

    # Buat DataFrame sesuai format training model (PENTING!)
    data = {
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': ratings,
        'distance_km': distance,
        'prep_time_min': prep_time,
        'multiple_deliveries': multiple,
        'Weather_conditions': weather,
        'Road_traffic_density': traffic,
        'City': city,
        'Festival': festival
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- 4. MAIN PAGE ---
st.title("🛵 Zomato Delivery Time Prediction")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk memprediksi estimasi waktu kedatangan makanan 
berdasarkan faktor kurir, cuaca, dan lalu lintas.
""")

st.write("---")

# Tampilkan Input User
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Detail Pesanan")
    st.write(input_df)

# --- 5. PREDICTION ENGINE ---
if st.button('🚀 Hitung Estimasi Waktu'):
    if model is not None:
        try:
            # Prediksi
            prediction = model.predict(input_df)
            result = round(prediction[0], 2)
            
            # Tampilkan Hasil
            with col2:
                st.success("✅ Prediksi Selesai!")
                st.metric(label="Estimasi Waktu Tiba", value=f"{result} Menit")
                
                # Logika Pesan Tambahan
                if result > 45:
                    st.warning("⚠️ Pengiriman mungkin terlambat karena kondisi sulit.")
                elif result < 25:
                    st.info("⚡ Pengiriman sangat cepat!")
                else:
                    st.info("👍 Waktu pengiriman normal.")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
            st.write("Pastikan format input sesuai dengan data training.")
    else:
        st.error("Model belum dimuat. Cek file .pkl Anda.")

st.write("---")
st.caption("Created by Eriel Setiawan Dewantoro | Powered by Streamlit & Scikit-Learn")