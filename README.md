# 🤖 AngelOne Support RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot trained on AngelOne's customer support documents and insurance PDFs. It allows users to ask queries and get precise answers using OpenAI's LLMs and FAISS vector search.

---

## 🚀 Live Demo

Try the hosted chatbot here: [AngelOne Support Chatbot](https://rag-chatbot-for-support-angle.streamlit.app/)

---

## 📦 Features

- Accepts custom OpenAI API key at runtime
- Uses FAISS vector store built from support documents
- Clean and interactive Streamlit interface
- Fast and accurate responses using LangChain

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

<pre> ```bash git clone https://github.com/Prince-r6710/RAG-Chatbot-for-Support.git cd rag-chatbot-for-support/rag-chatbot ``` </pre>


### 2. Clone the Repo
<pre> ```bash  python -m venv venv source venv/bin/activate `` </pre>

### 3. Install Dependencies
<pre> ```bash  pip install -r requirements.txt `` </pre>

### 4.Ingest Documents & Create Vector Store (Run Once)
<pre> ```bash  python ingest.py `` </pre>

### 5. Run the Chatbot
<pre> ```bash streamlit run app.py `` </pre>



📝 Notes
You need to provide your own OpenAI API key in the app at runtime.

The vectorstore files are already included for quick startup
