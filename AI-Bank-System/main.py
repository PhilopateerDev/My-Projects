import sqlite3
import random
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# مشروع: نظام ATM متقدم مع تنبؤ بالذكاء الاصطناعي
# الوصف: نظام بنكي يستخدم SQLite لإدارة البيانات
#        و Linear Regression للتوقعات المالية
# ==========================================

# 1. إنشاء الاتصال بقاعدة البيانات
# --------------------------------
conn = sqlite3.connect("bank.db")
cursor = conn.cursor()

def clear_screen():
    """دالة لمسح شاشة الكونسول لتحسين شكل الواجهة"""
    os.system('cls' if os.name == 'nt' else 'clear')

# 2. إعداد التاريخ والوقت
# -----------------------
date_for_ai = datetime.datetime.now()
date_for_human = date_for_ai.strftime("%Y-%m-%d %H:%M:%S")

# 3. إنشاء الجداول (هيكل قاعدة البيانات)
# -------------------------------------

# جدول المستخدمين: يخزن بيانات الحساب
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    password INTEGER,
    name TEXT,
    balance REAL
)
""")

# جدول السجل: يخزن كل العمليات للتحليل بالذكاء الاصطناعي
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    user_id INTEGER,
    amount REAL,
    date TEXT
)
""")
conn.commit()

# متغير لتتبع المستخدم الحالي
current_user_id = None

# ==========================================
# الجزء الأول: تسجيل الدخول / إنشاء حساب
# ==========================================
try:
    print("Welcome to the Advanced ATM System")
    choice = int(input("Choose: \n1_ Login \n2_ Sign Up\n> "))

    if choice == 1:
        # ---- تسجيل الدخول ----
        clear_screen()
        user_id = int(input("Enter your Account ID: "))

        # التحقق من وجود المستخدم
        cursor.execute("SELECT name FROM users WHERE id=?", (user_id,))
        row = cursor.fetchone()

        if row:
            print(f"Welcome back, {row[0]}!")
            current_user_id = user_id
        else:
            print("Error: Account ID not found.")
            sys.exit()

    else:
        # ---- إنشاء حساب جديد ----
        clear_screen()
        user_name = input("Enter your Name: ")
        user_password = int(input("Create a Password (numbers): "))
        user_balance = float(input("Enter Initial Deposit: "))

        while True:
            # توليد رقم حساب عشوائي وفريد
            new_id = random.randint(1000, 99999)

            # التأكد أن الرقم غير مستخدم
            cursor.execute("SELECT name FROM users WHERE id=?", (new_id,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?)",
                    (new_id, user_password, user_name, user_balance)
                )
                conn.commit()
                print(f"Account Created Successfully! Your ID is: {new_id}")
                current_user_id = new_id
                break

    # ==========================================
    # الجزء الثاني: العمليات الأساسية
    # ==========================================
    while True:
        print("\n" + "=" * 30)
        print("1_ Deposit")
        print("2_ Withdraw")
        print("3_ Inquiry (Check Balance)")
        print("4_ AI Predict (Forecast & Plot)")
        print("5_ Exit")
        print("=" * 30)

        op_choice = int(input("Select Operation: "))

        # ---- إيداع ----
        if op_choice == 1:
            money = float(input("Enter amount to deposit: "))
            cursor.execute(
                "UPDATE users SET balance = balance + ? WHERE id = ?",
                (money, current_user_id)
            )
            cursor.execute(
                "INSERT INTO history VALUES (?, ?, ?)",
                (current_user_id, money, date_for_human)
            )
            conn.commit()
            print(">> Deposit Successful.")

        # ---- سحب ----
        elif op_choice == 2:
            money = float(input("Enter amount to withdraw: "))
            cursor.execute(
                "UPDATE users SET balance = balance - ? WHERE id = ?",
                (money, current_user_id)
            )
            cursor.execute(
                "INSERT INTO history VALUES (?, ?, ?)",
                (current_user_id, -money, date_for_human)
            )
            conn.commit()
            print(">> Withdrawal Successful.")

        # ---- استعلام عن الرصيد ----
        elif op_choice == 3:
            cursor.execute(
                "SELECT balance FROM users WHERE id = ?",
                (current_user_id,)
            )
            bal = cursor.fetchone()
            print(f">> Your Current Balance: ${bal[0]:.2f}")

        # ---- التنبؤ بالذكاء الاصطناعي ----
        elif op_choice == 4:
            cursor.execute(
                "SELECT amount FROM history WHERE user_id = ?",
                (current_user_id,)
            )
            data = cursor.fetchall()

            if len(data) > 1:
                # تجهيز البيانات للنموذج
                Y = np.array([r[0] for r in data]).reshape(-1, 1)
                X = np.array(range(len(Y))).reshape(-1, 1)

                # تدريب الموديل
                model = LinearRegression()
                model.fit(X, Y)

                # التنبؤ بقيمة مستقبلية
                future_step = int(input("Enter future transaction step: "))
                prediction = model.predict([[future_step]])

                print(f">> AI Prediction: ${prediction[0][0]:.2f}")

                # رسم البيانات
                plt.figure(figsize=(10, 5))
                plt.plot(X, Y, marker='o', label='Transaction History')
                plt.plot(future_step, prediction[0][0], 'rx', markersize=10, label='Prediction')
                plt.title("AI Financial Forecast")
                plt.xlabel("Transaction Order")
                plt.ylabel("Amount")
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print(">> Not enough data for prediction.")

        # ---- خروج ----
        elif op_choice == 5:
            print("Thank you for using our bank.")
            break

except ValueError:
    print(">> Error: Please enter numbers only.")
except Exception as e:
    print(f">> Unexpected Error: {e}")
finally:
    if conn:
        conn.close()
