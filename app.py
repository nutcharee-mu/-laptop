import streamlit as st
import pandas as pd
from groq import Groq
import json

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Gus - Laptop Advisor", layout="wide")

# 2. ฟังก์ชันจัดการข้อมูล (ยกมาจาก Notebook ของคุณ)
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('laptop_price.csv')
    df.columns = df.columns.str.strip()
    
    # แปลง RAM เป็นตัวเลข (เช่น '8GB' -> 8)
    df['Ram_num'] = df['Ram'].str.extract('(\d+)').fillna(0).astype(int)
    
    # แปลงราคายูโรเป็นบาท
    df['Price_baht'] = df['Price_euros'] * 38 
    
    # คำนวณความคุ้มค่า (Value Score)
    df['Value_Score'] = (df['Ram_num'] / df['Price_baht']) * 100000
    return df

# 3. ฟังก์ชันคุยกับ AI (ใช้ Llama 3.3 แทนตัวเก่าที่โดนลบ)
def extract_intent(user_query, client):
    prompt = f"""
    คุณคือ "เอไอกัส" ผู้ช่วยเลือกโน้ตบุ๊ก วิเคราะห์ประโยค: "{user_query}"
    แล้วตอบกลับเป็น JSON เท่านั้น โครงสร้าง:
    {{
      "max_price": ตัวเลขงบสูงสุด,
      "min_ram": ตัวเลขแรมที่ต้องการ,
      "usage": "สรุปสั้นๆ"
    }}
    (หากไม่บอกงบ ให้ใส่ 100000, แรมใส่ 8)
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "Output JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

# 4. หน้าจอหลัก (UI)
def main():
    st.title("💻 AI CHATBOT: ระบบแนะนำโน้ตบุ๊กออนไลน์")
    
    # ดึง API Key จาก Secrets
    if "GROQ_API_KEY" not in st.secrets:
        st.error("กรุณาใส่ API Key ในหน้า Settings > Secrets")
        st.stop()

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    df = load_and_prep_data()

    user_input = st.text_input("อยากได้โน้ตบุ๊กแบบไหน บอกกัสได้เลย:")

    if user_input:
        with st.spinner('AI กำลังประมวลผล...'):
            try:
                intent = extract_intent(user_input, client)
                
                # กรองข้อมูลตามที่ AI วิเคราะห์
                results = df[
                    (df['Price_baht'] <= intent['max_price']) & 
                    (df['Ram_num'] >= intent['min_ram'])
                ].sort_values('Value_Score', ascending=False)

                st.subheader(f"🔍 สรุปงบ: {intent['max_price']:,} บาท | แรม {intent['min_ram']}GB ขึ้นไป")
                
                if not results.empty:
                    cols = st.columns(3)
                    for i, row in enumerate(results.head(3).itertuples()):
                        with cols[i]:
                            st.info(f"**{row.Company} {row.Product}**")
                            st.write(f"💰 ราคา: {row.Price_baht:,.0f}.-")
                            st.write(f"🧠 RAM: {row.Ram}")
                            st.write(f"⚙️ CPU: {row.Cpu}")
                else:
                    st.warning("กัสหาไม่เจอเลย ลองปรับงบเพิ่มดูไหม?")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()