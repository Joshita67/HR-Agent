from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

def create_faiss_index(pdf_path: str, index_path: str):
    print("🔹 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("🔹 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("🔹 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("🔹 Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("🔹 Saving index to disk...")
    vectorstore.save_local(index_path)

    print("✅ FAISS index created successfully.")

def load_faiss_index(index_path: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
