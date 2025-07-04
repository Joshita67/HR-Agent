import streamlit as st
from agent import simple_agent
import os
from utils import create_faiss_index

PDF_PATH = "data/HR Policy Manual.pdf"
INDEX_PATH = "embeddings"

st.title("🤖 Agentic HR Policy Assistant")
query = st.text_input("Ask your HR query")

if not os.path.exists(f"{INDEX_PATH}/index.faiss"):
    with st.spinner("Creating FAISS index..."):
        create_faiss_index(PDF_PATH, INDEX_PATH)

if query:
    with st.spinner("Thinking..."):
        answer = simple_agent(query)
        st.success(answer)
