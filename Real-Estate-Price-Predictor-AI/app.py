import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تحميل الموديل والسكيلر اللي حفظناهم قبل كدة
model = joblib.load('housing_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏠 Real Estate Price Predictor")
st.write("Enter the house details to get an instant AI price estimation.")

# إنشاء واجهة المستخدم (Sliders & Inputs)
med_inc = st.slider("Median Income (in $10k)", 0.5, 15.0, 3.0)
house_age = st.slider("House Age", 1, 52, 20)
ave_rooms = st.slider("Average Rooms", 1, 10, 5)
ave_occup = st.slider("Average Occupancy", 1, 6, 3)
lat = st.number_input("Latitude", value=34.0)
long = st.number_input("Longitude", value=-118.0)

# تجهيز البيانات للتنبؤ
input_data = np.array([[med_inc, house_age, ave_rooms, 1.0, 500.0, ave_occup, lat, long]])
input_scaled = scaler.transform(input_data)

if st.button("Predict House Price"):
    prediction = model.predict(input_scaled)
    st.success(f"### 🔮 Estimated Price: ${prediction[0]*100000:,.2f}")
    st.info("This prediction is based on the California Housing AI model.")
