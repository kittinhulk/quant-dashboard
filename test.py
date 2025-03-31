import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV
df = pd.read_csv("XAU_1h_data.csv")

# ดูตัวอย่าง 5 แถวแรก
print(df.head())
