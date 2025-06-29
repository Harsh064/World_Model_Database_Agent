import streamlit as st
from main import run_crew_agent
from utils.vector_store import build_vectorstore
# from crewai import Task, Crew

vectorstore = build_vectorstore()

st.set_page_config(page_title="Conversational DB Agent", page_icon="🧠")
st.title("🧠 Conversational DB Agent (CrewAI World Model Style)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about accounts, transactions, or customers")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        similar_docs = vectorstore.similarity_search(user_input, k=1)
        if similar_docs:
            st.toast(f"🔍 Closest intent: {similar_docs[0].page_content}")
    except Exception as e:
        st.toast("❌ Similarity check failed.")

    try:
        response = run_crew_agent(user_input)
    except Exception as e:
        response = f"❌ CrewAgent error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
