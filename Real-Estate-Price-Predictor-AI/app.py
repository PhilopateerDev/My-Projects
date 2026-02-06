import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- إعدادات الصفحة ---
st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Prediction Web App")

# --- خطوة قراءة البيانات ---
# بنقرأ ملف البيانات ونعرض منه جزء بسيط للمستخدم 
df = pd.read_csv('https://raw.githubusercontent.com/PhilopateerDev/My-Projects/main/Real-Estate-Price-Predictor-AI/housing_data.csv')
st.subheader("Dataset Preview")
st.write(df.head())

# --- معالجة عمود المكان ليكون أرقام قبل أي عملية حسابية ---
# بنحول الكلمات لأرقام عشان الـ scaler والموديل يقبلوا البيانات
if df['location'].dtype == 'object':
    locations_list = df['location'].unique().tolist()
    location_mapping = {name: i for i, name in enumerate(locations_list)}
    df['location'] = df['location'].map(location_mapping)
    display_locations = locations_list
else:
    display_locations = None

# --- تقسيم البيانات إلى X و y ---
# X هي كل المميزات و y هو السعر
X = df.drop('price', axis=1)
y = df['price']

# --- تقسيم البيانات لتدريب واختبار ---
# بنستخدم 80% للتدريب و 20% للاختبار مع تثبيت العشوائية
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- تجهيز البيانات (Scaling) ---
# الموديل بيحتاج البيانات تكون بمقياس موحد عشان النتائج تطلع دقيقة
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- تدريب الموديل ---
# بناء موديل الـ Linear Regression وتدريبه على البيانات المحجمة
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- عرض دقة الموديل ---
y_pred = model.predict(X_test_scaled)
score = r2_score(y_test, y_pred)
st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy (R2 Score): {score:.2f}")

# --- الرسم البياني لدقة التوقع فقط ---
# الرسمة دي بتبين السعر الحقيقي ضد المتوقع (مفيش علاقة بالمكان هنا)
st.subheader("Prediction Accuracy Chart")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
st.pyplot(fig)

# --- مدخلات المستخدم ---
st.divider()
st.subheader("Enter House Details for Prediction:")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area in Square Meters", min_value=0.0)
    # اختيار المكان بالاسم لراحة المستخدم والتحويل بيتم داخلياً
    if display_locations:
        selected_loc = st.selectbox("Location", options=display_locations)
        location_val = location_mapping[selected_loc]
    else:
        location_val = st.number_input("Location Code", min_value=0)

with col2:
    bathrooms = st.number_input("Number of Bathrooms", min_value=0)
    rooms = st.number_input("Number of Rooms", min_value=0)

# --- زر التوقع النهائي ---
if st.button("Predict Price Now"):
    # تحويل المدخلات لمصفوفة وعمل Scaling لها بنفس مقياس التدريب
    user_input = np.array([[area, location_val, bathrooms, rooms]])
    user_input_scaled = scaler.transform(user_input)
    
    # تنفيذ التوقع وعرض النتيجة
    final_prediction = model.predict(user_input_scaled)
    st.success(f"Estimated Market Price: ${final_prediction[0]:,.2f}")
    
