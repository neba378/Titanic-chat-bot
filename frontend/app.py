import os
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
            BACKEND_URL = "https://cooperative-exploration-production.up.railway.app"
            response = requests.post(f"{BACKEND_URL}/ask", json={"question": question}).json()

        st.subheader("📄 Answer:")
        if response["chart"]:
            chart = base64.b64decode(response["chart"])
            st.image(chart, caption="Generated Chart")
        st.write(response["answer"])
    else:
        st.warning("Please enter a question to ask the chatbot.")
PORT = int(os.environ.get("PORT", 8501))  # Get port from Railway

if __name__ == "__main__":
    st.run("app.py", host="0.0.0.0", port=PORT)