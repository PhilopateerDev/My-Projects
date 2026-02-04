import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. تحميل البيانات
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target 

# 2. استكشاف البيانات (Preview)
print("Data Preview:")
print(df.head())

# 3. رسم خريطة الارتباط (Heatmap)
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. فصل الميزات (X) عن الهدف (y)
X = df.drop('Price', axis=1) # تم تصحيح الخطأ من drophna إلى drop
y = df['Price']

# 5. تقسيم البيانات (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. توحيد مقاييس البيانات (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. بناء وتدريب الموديل
model = LinearRegression()
model.fit(X_train_scaled, y_train) # التدريب يتم على البيانات الـ scaled

# 8. التنبؤ بالأسعار لبيانات الاختبار
y_pred = model.predict(X_test_scaled)

# 9. تقييم الموديل (الأرقام)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"متوسط الخطأ (MAE): {mae:.4f}")
print(f"نسبة الدقة (R2 Score): {r2:.4f}")

# 10. رسم العلاقة بين السعر الحقيقي والمتوقع
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

# 11. حساب ورسم أهم العوامل المؤثرة (Feature Importance)
importance = model.coef_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Impact of Each Feature on House Price')
plt.show()

# 12. تجربة الموديل على بيت "جديد" من خيالك
# ترتيب البيانات: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long
new_house = np.array([[8.0, 15.0, 6.0, 1.0, 500.0, 3.0, 34.0, -118.0]])
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print(f"السعر المتوقع لهذا البيت الافتراضي هو: {predicted_price[0]:.2f}")

# 13. حفظ الموديل والسكيلر للاستخدام لاحقاً
joblib.dump(model, 'housing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("تم حفظ الموديل والسكيلر بنجاح!")
