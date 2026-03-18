import streamlit as st
import pandas as pd
from groq import Groq
import json
import re

# --- การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="AI CHATBOT - Laptop Advisor", layout="wide")

# --- ส่วนของการจัดการข้อมูล (Logic จาก Notebook) ---
@st.cache_data
def load_and_clean_data():
    # โหลดไฟล์ (ต้องชื่อ laptop_price.csv)
    df = pd.read_csv('laptop_price.csv')
    
    # ลบช่องว่างในชื่อคอลัมน์
    df.columns = df.columns.str.strip()
    
    # แปลง RAM เป็นตัวเลข (เช่น '8GB' -> 8)
    df['Ram_num'] = df['Ram'].str.extract('(\d+)').fillna(0).astype(int)
    
    # แปลง Weight เป็นตัวเลข (เช่น '1.37kg' -> 1.37)
    df['Weight_num'] = df['Weight'].str.replace('kg', '', case=False).str.strip()
    df['Weight_num'] = pd.to_numeric(df['Weight_num'], errors='coerce').fillna(0)
    
    # แปลงราคายูโรเป็นบาท (เรทประมาณ 38 บาท)
    df['Price_baht'] = df['Price_euros'] * 38
    
    # คำนวณ Value Score (ความคุ้มค่า) ตาม Logic ใน Notebook
    # สูตร: (Ram / Price) * 100000
    df['Value_Score'] = (df['Ram_num'] / df['Price_baht']) * 100000
    
    return df

# --- ส่วนการเรียก AI (Groq API) ---
def extract_intent(user_query, client):
    # Prompt ที่บังคับให้ AI ตอบเป็น JSON (แก้ BadRequestError)
    prompt = f"""
    คุณคือผู้ช่วยวิเคราะห์สเปคโน้ตบุ๊ก "เอไอกัส"
    จงสกัดข้อมูลจากประโยค: "{user_query}"
    ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่นปน โดยใช้โครงสร้างนี้:
    {{
      "max_price": ตัวเลขงบสูงสุด (ถ้าไม่ระบุให้ใส่ 100000),
      "min_ram": ตัวเลขแรมที่ต้องการ (ถ้าไม่ระบุให้ใส่ 8),
      "category": "ระบุประเภท เช่น Gaming, Ultrabook, Work"
    }}
    """
    
    completion = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that always outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

# --- ส่วนประกอบหลักของแอป ---
def main():
    st.title("💻 AI CHATBOT: ระบบแนะนำโน้ตบุ๊กอัจฉริยะ")
    st.markdown("ถามได้เลย เช่น *'หาโน้ตบุ๊กไว้เรียน งบ 25,000'* หรือ *'อยากได้เครื่องแรงๆ งบ 5 หมื่น'*")

    # ตรวจสอบ API Key ใน Secrets
    if "GROQ_API_KEY" not in st.secrets:
        st.error("กรุณาใส่ GROQ_API_KEY ใน Streamlit Secrets ก่อนใช้งาน")
        st.stop()
    
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # โหลดข้อมูล
    df = load_and_clean_data()

    # รับ Input
    user_input = st.text_input("ความต้องการของคุณ:", placeholder="พิมพ์ที่นี่...")

    if user_input:
        with st.spinner('กัสกำลังวิเคราะห์ข้อมูลให้ครับ...'):
            try:
                # 1. AI วิเคราะห์เจตนา
                intent = extract_intent(user_input, client)
                
                # 2. กรองข้อมูลตามที่ AI บอก
                results = df[
                    (df['Price_baht'] <= intent['max_price']) & 
                    (df['Ram_num'] >= intent['min_ram'])
                ]
                
                # 3. เรียงลำดับตามความคุ้มค่า (Value Score)
                results = results.sort_values('Value_Score', ascending=False)

                # 4. แสดงผล
                st.subheader(f"🔍 ผลวิเคราะห์: งบไม่เกิน {intent['max_price']:,} บาท | RAM {intent['min_ram']}GB ขึ้นไป")
                
                if not results.empty:
                    top_3 = results.head(3)
                    for _, row in top_3.iterrows():
                        with st.expander(f"⭐ {row['Company']} {row['Product']} - {row['Price_baht']:,.0f} บาท"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**CPU:** {row['Cpu']}")
                                st.write(f"**RAM:** {row['Ram']}")
                            with col2:
                                st.write(f"**หน้าจอ:** {row['Inches']} นิ้ว")
                                st.write(f"**น้ำหนัก:** {row['Weight']}")
                            st.progress(min(row['Value_Score']/10, 1.0), text=f"คะแนนความคุ้มค่า: {row['Value_Score']:.2f}")
                else:
                    st.warning("กัสหาเครื่องที่ตรงสเปคในงบนี้ไม่เจอเลยครับ ลองปรับงบเพิ่มดูไหม?")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()