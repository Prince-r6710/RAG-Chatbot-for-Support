import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --- Page Setup ---
st.set_page_config(page_title="AngelOne Support Chatbot", page_icon="🤖")
st.title("🤖 AngelOne Support Chatbot")
st.markdown("Ask a question about AngelOne services:")

# --- API Key Input ---
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# --- Question Input ---
question = st.text_input("💬 Your question:")

# Helper: customize answer with metadata info if available
def format_answer(answer_docs, raw_answer):
    if not answer_docs:
        return "I Don't know"

    # Collect unique plan/source metadata from docs used in answer
    metadata_info = []
    for doc in answer_docs:
        meta = doc.metadata
        if "plan_name" in meta:
            metadata_info.append(meta["plan_name"])
        elif "source" in meta:
            metadata_info.append(meta["source"])

    metadata_info = list(set(metadata_info))  # unique values

    # Format metadata info in answer
    if metadata_info:
        meta_text = ", ".join(metadata_info)
        return f"**Based on:** {meta_text}\n\n{raw_answer}"
    else:
        return raw_answer


if openai_key and question:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vectorstore = FAISS.load_local(
            "vectorstore", embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(openai_api_key=openai_key, temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )
        
        with st.spinner("Searching..."):
            response = qa_chain({"query": question})  # use __call__ here
            
        answer = response["result"]
        docs = response["source_documents"]

        if "I don't know" in answer.lower() or len(answer.strip()) < 10:
            st.warning("I Don't know")
        else:
            # Example: show answer and source document metadata
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
