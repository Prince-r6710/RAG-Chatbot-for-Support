import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

# --- Page Setup ---
st.set_page_config(page_title="AngelOne Support Chatbot", page_icon="🤖")
st.title("🤖 AngelOne Support Chatbot")
st.markdown("Ask a question about AngelOne services:")

# --- API Key Input ---
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# --- Question Input ---
question = st.text_input("💬 Your question:")

# --- Process ---
if openai_key and question:
    try:
        # 1. Load embeddings with user's key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

        # 2. Dynamically get full path to vectorstore directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vectorstore_path = os.path.join(current_dir, "vectorstore")

        # 3. Load the FAISS vectorstore from disk
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )

        # 4. Create retriever and QA chain
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # 5. Run the chain and show answer
        with st.spinner("Searching..."):
            result = qa_chain.run(question)
        st.success("✅ Answer:")
        st.write(result)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

elif not openai_key:
    st.info("Please enter your OpenAI API key to begin.")
elif not question:
    st.info("Type your question to get started.")
