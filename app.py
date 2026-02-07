import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA

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



# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains.retrieval_qa import RetrievalQA
# from langchain.chains import RetrievalQA
# import os

# st.set_page_config(page_title="Enterprise RAG System")

# st.title("📄 Enterprise Document Q&A System")

# # Load embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Load FAISS index
# db = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# # Load LLM (Gemini)
# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     convert_system_message_to_human=True,
#     temperature=0.2
# )



# # RAG chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=db.as_retriever(search_kwargs={"k": 2}),
#     return_source_documents=True
# )

# query = st.text_input("Ask a question from the document")

# if query:
#     with st.spinner("Thinking..."):
#         result = qa(query)
#         st.subheader("Answer")
#         st.write(result["result"])


