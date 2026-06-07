import streamlit as st
from rag_chat import ask

st.set_page_config(
    page_title="Chatbot Sekolah"
)

st.title("🎓 Chatbot Sekolah")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input(
    "Tanyakan sesuatu..."
)

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    answer = ask(question)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )