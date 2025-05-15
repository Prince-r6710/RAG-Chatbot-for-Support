import os
import nltk
nltk.download("all")
import glob
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_documents():
    docs = []

    # Load PDFs
    pdf_files = glob.glob("data/*.pdf")
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    # Load HTML pages (skip if not available)
    html_files = glob.glob("data/html/*.html")
    for html in html_files:
        loader = UnstructuredHTMLLoader(html)
        docs.extend(loader.load())

    # Load Word DOCX file (your case)
    docx_path = "data/America's_Choice_Medical_Questions_-_Modified_(3) (1).docx"
    if os.path.exists(docx_path):
        loader = UnstructuredWordDocumentLoader(docx_path)
        docs.extend(loader.load())

    return docs

def split_and_store():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")

if __name__ == "__main__":
    split_and_store()
