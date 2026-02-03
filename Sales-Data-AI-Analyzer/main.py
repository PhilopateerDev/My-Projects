import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import os

# إعدادات عامة لتحسين مظهر النتائج
warnings.filterwarnings('ignore')
plt.style.use('ggplot') 

# ========================================================
# المرحلة 0: نظام توليد البيانات الذكي (Data Generator)
# الهدف: ضمان عمل الكود فوراً عند التحميل دون الحاجة لملف خارجي
# ========================================================
def generate_sample_data():
    file_name = "raw_sales_data.csv"
    if not os.path.exists(file_name):
        print("📊 Creating synthetic sales data for demonstration...")
        data = {
            'Order_Date': pd.date_range(start='2025-01-01', periods=24, freq='M'),
            'Product_Name': ['Laptop', 'Phone', 'Chair', 'Table', 'Headphones'] * 4 + ['Laptop', 'Phone', 'Chair', 'Table'],
            'Category': ['Electronics', 'Electronics', 'Furniture', 'Furniture', 'Elec'] * 4 + ['Electronics', 'Electronics', 'Furniture', 'Furniture'],
            'Quantity': [10, 20, 15, 5, np.nan, 12, 25, 10, 8, 30, 15, 22, 10, 18, 14, 6, 9, 21, 24, 11, 7, 28, 13, 19],
            'Unit_Price': [1000, 500, 200, 500, 100] * 4 + [1000, 500, 200, 500],
            'Total_Sales': [10000, 10000, 3000, 2500, 500, 12000, np.nan, 2000, 4000, 3000, 15000, 11000, 10000, 9000, 2800, 3000, 9000, 10500, 12000, 5500, 7000, 14000, 2600, 9500]
        }
        pd.DataFrame(data).to_csv(file_name, index=False)
        print(f"✅ File '{file_name}' generated successfully!\n")

# ========================================================
# المرحلة 1: معالجة البيانات (Level 3: Data Mastery)
# ========================================================
generate_sample_data()
df = pd.read_csv("raw_sales_data.csv")

# تنظيف البيانات بذكاء (Data Cleaning & Imputation)
df["Product_Name"] = df["Product_Name"].str.strip().str.title()
df["Category"] = df["Category"].str.strip().str.title().replace({"Elec": "Electronics", "Furn": "Furniture"})

# ملء الفراغات بناءً على معادلات رياضية (Logic-Based Imputation)
df["Quantity"] = df["Quantity"].fillna(df["Total_Sales"] // df["Unit_Price"])
df["Total_Sales"] = df["Total_Sales"].fillna(df["Quantity"] * df["Unit_Price"])

# تحويل التواريخ لاستخراج الأنماط الزمنية
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month_Num"] = range(1, len(df) + 1) # ترتيب الشهور للتنبؤ
df["Month_Name"] = df["Order_Date"].dt.month_name()

# ========================================================
# المرحلة 2: التحليل البصري (Level 3: Analytics)
# ========================================================
# مبيعات كل فئة
category_sales = df.groupby('Category')['Total_Sales'].sum()
category_sales.plot(kind='pie', autopct='%1.1f%%', title='Revenue Distribution by Category', figsize=(8,8))
plt.show()

# ========================================================
# المرحلة 3: الذكاء الاصطناعي (Level 4: Machine Learning)
# ========================================================
# تجهيز الموديل
X = df[['Month_Num']] # المتغير المستقل (الزمن)
y = df['Total_Sales'] # المتغير التابع (المبيعات)

model = LinearRegression()
model.fit(X, y)

# التنبؤ بالشهر القادم
next_step = [[len(df) + 1]]
prediction = model.predict(next_step)

print(f"🔮 AI Future Sales Prediction: ${prediction[0]:,.2f}")

# رسم التنبؤ والواقع (Advanced Visualization)
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Actual Sales Data')
plt.plot(X, model.predict(X), color='red', label='Trend Line (Regression)')
plt.scatter(next_step, prediction, color='green', marker='*', s=250, label='AI Forecasted Point')
plt.title("Sales Growth & Future Prediction Pipeline")
plt.xlabel("Month Step")
plt.ylabel("Revenue ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n🚀 Full Data Pipeline executed successfully!")


