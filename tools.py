import pandas as pd
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from utils import load_faiss_index

# Load employee data
employee_df = pd.read_csv("data/employee_data.csv")

# Load local text generation pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Query HR Policy using FAISS RAG
def query_policy(query: str):
    db = load_faiss_index("embeddings")
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Handle structured employee data queries
def employee_data_query(query: str):
    if "joining date" in query.lower():
        for _, row in employee_df.iterrows():
            if row["Name"].lower() in query.lower():
                return f"{row['Name']} joined on {row['JoiningDate']}"
        return "Employee not found"
    return "Please ask a date-related query (e.g. joining date)."
