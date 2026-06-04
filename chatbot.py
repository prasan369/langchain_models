from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

st.title("Chat")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Type a message")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    result = model.invoke(user_input)
    st.session_state.history.append({"role": "assistant", "content": result.content})
    st.chat_message("assistant").write(result.content)