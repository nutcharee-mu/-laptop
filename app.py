import streamlit as st
import pandas as pd
from groq import Groq
import json

# เชื่อมต่อกับ Groq API
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_data
def load_data():
    # ตรวจสอบว่าชื่อไฟล์ csv ตรงกับของคุณ
    return pd.read_csv('laptop_price.csv')

df = load_data()

st.title("💻 AI CHATBOT: ระบบแนะนำโน้ตบุ๊ก")

user_input = st.text_input("ระบุความต้องการของคุณ:", placeholder="เช่น หาโน้ตบุ๊กทำงาน งบ 25,000")

if user_input:
    with st.spinner('กำลังคิดให้ครับ...'):
        # 1. ให้ AI สรุป Intent (ดัดแปลงจากโค้ดเดิมของคุณ)
        prompt = f"""วิเคราะห์ความต้องการ: "{user_input}" 
        แล้วตอบเป็น JSON เท่านั้น {{ "budget": ตัวเลข, "usage": "เรียน/เล่นเกม/ทำงาน" }}"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen-2.5-32b", # ใช้ Qwen 2.5 ตัวแรง
            response_format={"type": "json_object"}
        )
        
        res = json.loads(chat_completion.choices[0].message.content)
        
        # 2. กรองข้อมูลใน Pandas (Logic จาก Notebook)
        filtered_df = df[df['Price_baht'] <= res.get('budget', 1000000)].head(3)
        
        # 3. แสดงผล
        st.write(f"วิเคราะห์งบประมาณ: {res.get('budget')} บาท")
        st.table(filtered_df[['Company', 'Product', 'Price_baht']])