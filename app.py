import os
import streamlit as st

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Free Resume Chatbot", layout="centered")
st.title("⚡ Resume Q&A Chatbot (Free & Fast)")

# -----------------------------
# Load Groq API Key
# -----------------------------
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("Groq API key not found. Set GROQ_API_KEY.")
    st.stop()

# -----------------------------
# Local Embeddings (FAST & FREE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS Index
# -----------------------------
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 1})

# -----------------------------
# FREE & FAST LLM (Groq)
# -----------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)


# -----------------------------
# Retrieval QA
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# -----------------------------
# Query
# -----------------------------
query = st.text_input("Ask something about the resume:")

if query:
    with st.spinner("⚡ Thinking..."):
        result = qa.invoke({"query": query + " Answer briefly."})
        st.subheader("Answer")
        st.write(result["result"])





