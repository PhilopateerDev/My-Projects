import time
import os

# قائمة رئيسية لتخزين البيانات (اسم المهمة ودرجة الأولوية)
big_list = []

# وظيفة برمجية لتنظيف شاشة الكونسول حسب نوع نظام التشغيل
def clear():
    os.system("cls" if os.name == "nt" else "clear")

while True:
    print("\n--- Smart To-Do System ---")
    c = input("1. Add Task\n2. View Tasks\n3. Delete Task\n4. Exit\n\nChoose an option: ")

    if c == "1":
        # استقبال البيانات من المستخدم
        work = input("Enter task name: ")
        priority = input("Priority (1:High, 2:Medium, 3:Normal): ")
        
        # التأكد من أن المستخدم أدخل قيمة صحيحة للأولوية
        if priority not in ["1", "2", "3"]:
            print("Invalid input! Please enter 1, 2, or 3.")
        else:
            # تخزين المهمة كقائمة فرعية داخل القائمة الكبيرة
            small_list = [work, int(priority)]
            big_list.append(small_list)
            print("Task added successfully ✅")
        
        # الانتظار لمدة ثانية ثم تنظيف الشاشة
        time.sleep(1)
        clear()

    elif c == "2":
        # التحقق إذا كانت القائمة تحتوي على بيانات أم لا
        if not big_list:
            print("The list is empty!")
        else:
            # ترتيب القائمة تصاعدياً بناءً على قيمة الأولوية (رقم 1 يظهر أولاً)
            big_list.sort(key=lambda x: x[1])
            print("\nTasks sorted by priority:")
            for item in big_list:
                # تحويل أرقام الأولوية إلى نصوص توضيحية للمستخدم
                label = "High" if item[1] == 1 else "Medium" if item[1] == 2 else "Normal"
                print(f"[{label}] - {item[0]}")
        
        input("\nPress Enter to return to main menu...")
        clear()

    elif c == "3":
        # البحث عن مهمة معينة وحذفها
        delete_work = input("Enter the task name to delete: ")
        found = False
        for item in big_list:
            if item[0] == delete_work:
                big_list.remove(item) # حذف العنصر من القائمة
                found = True
                print("Task deleted successfully 🗑️")
                break 
        
        # تنبيه في حال عدم وجود الاسم المدخل في القائمة
        if not found:
            print("Task not found!")
        
        time.sleep(1)
        clear()

    elif c == "4":
        # إنهاء حلقة التكرار والخروج من البرنامج
        print("Exiting program...")
        break
