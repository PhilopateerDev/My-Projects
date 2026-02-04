import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# تحسين شكل الرسومات البيانية وجعلها تدعم الأحجام الكبيرة
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = [12, 6]

def generate_professional_data(num_rows=5000):
    """دالة لتوليد بيانات مالية بمنطق أمني قوي لمحاكاة الواقع."""
    print("\n[1/4] 🔄 جاري إنشاء بيانات العمليات المالية الوهمية...")
    np.random.seed(42) # تثبيت العشوائية لضمان تكرار النتائج
    
    data = {
        'Amount': np.random.uniform(10, 10000, num_rows),          # مبالغ من 10 إلى 10 آلاف دولار
        'Hour_of_Day': np.random.randint(0, 24, num_rows)          # توقيت العملية خلال الـ 24 ساعة
        'Is_International': np.random.choice([0, 1], num_rows, p=[0.85, 0.15]), # 15% عمليات دولية
        'Login_Attempts': np.random.randint(1, 6, num_rows),       # محاولات الدخول من 1 لـ 5
        'Account_Age_Days': np.random.randint(1, 3650, num_rows)   # عمر الحساب بالأيام (حتى 10 سنوات)
    }
    
    df = pd.DataFrame(data)
    
    # --- منطق تحديد الاحتيال (Business Logic) ---
    # الحالة 1: مبلغ ضخم (> 4000) في وقت الفجر (قبل الساعة 5 صباحاً)
    # الحالة 2: محاولات دخول كثيرة (> 3) مع كون العملية دولية
    fraud_condition = (
        ((df['Amount'] > 4000) & (df['Hour_of_Day'] < 5)) | 
        ((df['Login_Attempts'] > 3) & (df['Is_International'] == 1))
    )
    df['Is_Fraud'] = fraud_condition.astype(int) # تحويل True/False إلى 1/0
    
    print(f"✅ تم إنشاء {num_rows} عملية | عدد حالات الاحتيال المكتشفة: {df['Is_Fraud'].sum()}")
    return df

def train_smart_model(df):
    """تدريب نموذج 'الغابة العشوائية' لاتخاذ قرارات أمنية دقيقة."""
    print("\n[2/4] 🧠 جاري تدريب محرك الذكاء الاصطناعي (Random Forest)...")
    
    # فصل الميزات (X) عن النتيجة المطلوبة (y)
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']
    
    # تقسيم البيانات: 80% للتدريب و 20% لاختبار دقة الموديل
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # إنشاء الموديل مع موازنة البيانات (لأن حالات الاحتيال دائماً أقل)
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train) # عملية التعلم
    
    print("✅ اكتمل تدريب الموديل بنجاح.")
    return model, X_test, y_test

def evaluate_and_visualize(model, X_test, y_test):
    """تقييم أداء الموديل ورسم لوحة النتائج التوضيحية."""
    print("\n[3/4] 📊 جاري تحليل الأداء ورسم النتائج البيانية...")
    
    y_pred = model.predict(X_test) # التوقع بناءً على ما تعلمه
    
    # طباعة تقرير الدقة التفصيلي في الكونسول
    print("-" * 50)
    print("      تقرير أداء كاشف الاحتيال الذكي (AI Report)")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    # إنشاء لوحة رسومات مكونة من شكلين
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 1. مصفوفة الارتباك (Confusion Matrix) لتوضيح الصح والخطأ
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1, cbar=False)
    ax1.set_title('مصفوفة الارتباك: عمليات سليمة vs احتيال')
    ax1.set_xlabel('توقع الذكاء الاصطناعي')
    ax1.set_ylabel('الحالة الحقيقية للعملية')

    # 2. أهمية الميزات (Feature Importance) لمعرفة سبب القرار
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    importances.sort_values().plot(kind='barh', color='darkred', ax=ax2)
    ax2.set_title('أهم العوامل المؤثرة في كشف الاحتيال')
    
    plt.tight_layout()
    plt.show() # عرض الرسومات

def deploy_model(model):
    """حفظ الموديل النهائي في ملف ليكون جاهزاً للاستخدام الفوري."""
    print("\n[4/4] 🚀 جاري تجهيز الموديل للنشر (Deployment)...")
    folder = "AI-Financial-Fraud-Detector"
    
    # التأكد من وجود المجلد
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    path = os.path.join(folder, 'fraud_detector_v1.pkl')
    joblib.dump(model, path) # حفظ الموديل كملف
    print(f"✅ الموديل جاهز للعمل! تم الحفظ في: {path}")

# --- نقطة انطلاق البرنامج ---
if __name__ == "__main__":
    print("🚀 نظام كشف الاحتيال المالي الذكي يبدأ العمل الآن...")
    
    # تنفيذ الخطوات بالترتيب
    dataframe = generate_professional_data()
    ai_model, x_val, y_val = train_smart_model(dataframe)
    evaluate_and_visualize(ai_model, x_val, y_val)
    deploy_model(ai_model)
    
    print("\n🎯 تمت المهمة بنجاح: حارس أمنك الرقمي الآن في الخدمة!")
