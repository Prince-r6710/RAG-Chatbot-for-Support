import os
import glob
from dotenv import load_dotenv
load_dotenv()

import os
import glob
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,              # Added for fallback
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_documents():
    docs = []

    # Define your plans and PDFs
    plans = {
        "2500_plan_gold.pdf": "2500 Plan Gold",
        "5000_plan_bronze.pdf": "5000 Plan Bronze",
        "5000_plan_hsa.pdf": "5000 Plan HSA",
        "7350_plan_copper.pdf": "7350 Plan Copper",
    }

    for filename, plan_name in plans.items():
        pdf_path = os.path.join("/home/princekumargupta/RAG-Chatbot-for-Support/rag-chatbot/data", filename)
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                loaded_docs = loader.load()
            except Exception as e:
                print(f"❌ PyPDFLoader failed for {filename}: {e}")
                print("👉 Trying PDFPlumberLoader instead...")
                try:
                    loader = PDFPlumberLoader(pdf_path)
                    loaded_docs = loader.load()
                except Exception as e2:
                    print(f"❌ PDFPlumberLoader also failed for {filename}: {e2}")
                    continue

            for doc in loaded_docs:
                doc.metadata["plan_name"] = plan_name
            docs.extend(loaded_docs)

    # Load HTML files with fixed metadata key
    html_files = glob.glob("/home/princekumargupta/RAG-Chatbot-for-Support/rag-chatbot/data/html/*.html")
    for html in html_files:
        loader = UnstructuredHTMLLoader(html)
        doc = loader.load()
        for file in doc:
            file.metadata["source"] = "angel one"
        docs.extend(doc)

    # Load DOCX file
    docx_path = os.path.join("/home/princekumargupta/RAG-Chatbot-for-Support/rag-chatbot/data", "america_choice_questions.docx")
    if os.path.exists(docx_path):
        loader = UnstructuredWordDocumentLoader(docx_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = "America Choice Medical Questions"
        docs.extend(loaded_docs)

    return docs


def split_and_store():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vectorstore saved with documents and metadata!")


if __name__ == "__main__":
    split_and_store()
