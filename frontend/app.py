import streamlit as st
import requests
import base64

st.set_page_config(layout="centered", page_title="🚢 Titanic Chatbot")

st.title("🚢 Titanic Chatbot")  
st.write("Ask me anything about the Titanic dataset, and I'll provide insights!")
question = st.text_input("Ask a question:")
if st.button("Ask"):
    if question:
        with st.spinner("Thinking... 🤔"):
            response = requests.post("https://titanic-chat-bot-production.up.railway.app/ask", json={"question": question}).json()
        st.subheader("📄 Answer:")
        print(response)
        if 'chart' in response:
            chart = base64.b64decode(response["chart"])
            st.image(chart, caption="Generated Chart")
        if 'answer' in response:
            st.write(response["answer"])
        else:
            st.error("An error occured! Please try again.")
    else:
        st.warning("Please enter a question to ask the chatbot.")