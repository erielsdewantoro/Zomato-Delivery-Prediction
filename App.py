import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Zomato AI Delivery Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan lebih modern
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF5252;
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('zomato_model_final.pkl')
        return model
    except Exception as e:
        st.error(f"⚠️ Model tidak ditemukan: {e}")
        return None

model = load_model()

# --- 3. SIDEBAR INPUT (CONTROLS) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Zomato_Logo.svg", width=150)
    st.header("🎛️ Panel Kontrol Pesanan")
    
    with st.expander("👤 Profil Kurir", expanded=True):
        age = st.slider('Umur Kurir (Tahun)', 18, 65, 30)
        ratings = st.slider('Rating Kurir (⭐)', 1.0, 5.0, 4.5, 0.1)
        
    with st.expander("📍 Detail Pengiriman", expanded=True):
        distance = st.number_input('Jarak Pengiriman (km)', 1.0, 50.0, 5.0, step=0.5)
        prep_time = st.number_input('Waktu Masak (Menit)', 5, 120, 15, step=5)
        multiple = st.selectbox('Jumlah Orderan Dibawa', [0, 1, 2, 3])
        
    with st.expander("🌤️ Kondisi Eksternal", expanded=False):
        weather = st.selectbox('Cuaca', ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Fog', 'Cloudy'])
        traffic = st.selectbox('Kepadatan Lalu Lintas', ['Low', 'Medium', 'High', 'Jam'])
        city = st.selectbox('Tipe Kota', ['Metropolitian', 'Urban', 'Semi-Urban'])
        festival = st.selectbox('Sedang Festival?', ['No', 'Yes'])

    st.markdown("---")
    st.caption("© 2025 Eriel Setiawan | Powered by Random Forest")

# Buat DataFrame Input
input_data = {
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
input_df = pd.DataFrame([input_data])

# --- 4. MAIN DASHBOARD ---
st.title("🛵 Zomato AI Delivery Time Estimator")
st.markdown("Sistem cerdas untuk memprediksi estimasi waktu tiba (**ETA**) makanan berdasarkan kondisi real-time.")

# TAB SYSTEM
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Live", "📊 Analisis Model", "📂 Tentang Data"])

# --- TAB 1: PREDIKSI ---
with tab1:
    # Display Key Metrics (Input Overview)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jarak", f"{distance} km")
    col2.metric("Rating Kurir", f"{ratings} ⭐")
    col3.metric("Traffic", traffic)
    col4.metric("Cuaca", weather)
    
    st.write("---")
    
    # Tombol Prediksi Besar
    if st.button('🚀 Analisis & Prediksi Sekarang'):
        if model:
            with st.spinner('Sedang mengkalkulasi rute terbaik...'):
                prediction = model.predict(input_df)[0]
                result = round(prediction, 2)
                
            # Tampilan Hasil (Hero Section)
            st.success("✅ Kalkulasi Selesai!")
            
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.subheader("⏱️ Estimasi Waktu Tiba:")
                st.markdown(f"<h1 style='color:#FF5252; font-size: 60px;'>{result} Menit</h1>", unsafe_allow_html=True)
                
                # Logika Insight Bisnis
                if result > 45:
                    st.error("⚠️ **Peringatan:** Pengiriman berisiko terlambat. Sarankan pelanggan memesan lebih awal.")
                elif result < 25:
                    st.balloons()
                    st.success("⚡ **Flash Delivery:** Kondisi sangat optimal untuk pengiriman cepat!")
                else:
                    st.info("👍 **Normal:** Waktu pengiriman standar sesuai ekspektasi.")
            
            with res_col2:
                st.subheader("Faktor Utama:")
                st.progress(min(result/60, 1.0))
                st.caption(f"Beban Waktu: {int((result/60)*100)}% dari batas 1 jam")

# --- TAB 2: ANALISIS MODEL ---
with tab2:
    st.header("🧠 Di Balik Layar AI")
    st.markdown("Model ini menggunakan **Random Forest Regressor** dengan akurasi R2 Score ~75%.")
    
    # Feature Importance Plot (Static Image or Interactive)
    # Karena kita tidak bisa load training data di sini, kita buat dummy plot atau hardcode nilai penting
    # Ini contoh data Feature Importance dari notebook Anda
    features = ['Rating Kurir', 'Jarak (km)', 'Umur Kurir', 'Traffic Jam', 'Prep Time']
    importance = [0.24, 0.18, 0.13, 0.11, 0.08]
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
    ax.set_title('Top 5 Faktor Penentu Kecepatan')
    ax.set_xlabel('Tingkat Pengaruh')
    st.pyplot(fig)
    
    st.info("💡 **Insight:** Rating kurir dan Jarak adalah dua faktor terbesar yang mempengaruhi kecepatan pengiriman.")

# --- TAB 3: DATASET ---
with tab3:
    st.header("📂 Sampel Data Zomato")
    st.markdown("Cuplikan data yang digunakan untuk melatih model ini.")
    
    # Load raw data sampel (pastikan file csv ada di repo jika mau fitur ini jalan)
    # Jika tidak mau upload csv besar, hapus bagian try-except ini
    try:
        df_sample = pd.read_csv("Zomato Dataset.csv").head(10)
        st.dataframe(df_sample)
    except:
        st.warning("File dataset tidak ditemukan di repository. Upload 'Zomato Dataset.csv' untuk melihat sampel.")
        st.markdown("""
        **Struktur Data:**
        - **ID:** Identitas Order
        - **Delivery_person_Age:** Umur Kurir
        - **Delivery_person_Ratings:** Rating Kinerja
        - **Time_taken:** Target Prediksi (Waktu)
        """)
