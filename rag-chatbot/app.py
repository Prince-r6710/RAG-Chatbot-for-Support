import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- Page Setup ---
st.set_page_config(page_title="AngelOne Support Chatbot", page_icon="🤖")
st.title("🤖 AngelOne Support Chatbot")
st.markdown("Ask a question about AngelOne services:")

# --- API Key Input ---
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# --- Question Input ---
question = st.text_input("💬 Your question:")

# --- Define vectorstore path relative to this script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore")

if openai_key and question:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

        # Debug info: show current dir and vectorstore contents
        st.write("Current working directory:", os.getcwd())
        st.write("Vectorstore directory contents:", os.listdir(VECTORSTORE_PATH))

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        with st.spinner("Searching..."):
            response = qa_chain({"query": question})

        answer = response["result"]
        docs = response["source_documents"]

        if "i don't know" in answer.lower() or len(answer.strip()) < 10:
            st.warning("I Don't know")
        else:
            st.success("✅ Answer:")
            st.write(answer)
            st.markdown("### Sources:")
            for doc in docs:
                st.markdown(f"- {doc.metadata.get('source', 'unknown source')}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

elif not openai_key:
    st.info("Please enter your OpenAI API key to begin.")
elif not question:
    st.info("Type your question to get started.")
