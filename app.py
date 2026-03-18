import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Laptop Recommender", layout="wide")

# โหลด data
@st.cache_data
def load_data():
    df = pd.read_csv("laptop_price.csv", encoding="latin-1")
    return df

df = load_data()

st.title("💻 AI Laptop Recommendation")

user_input = st.text_input("พิมพ์ความต้องการ")

if user_input:
    st.write(f"คุณพิมพ์: {user_input}")

    # 👉 ตรงนี้เรียก function ของคุณ
    # table, review = get_recommendation(user_input)

    # demo
    st.write("🤖 AI: กำลังแนะนำ...")
    st.dataframe(df.head(3))