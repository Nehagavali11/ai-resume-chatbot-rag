import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Resume Chatbot", page_icon="🤖")
st.title("🤖 AI Resume Chatbot (RAG)")
st.write("Upload any PDF and chat with it")

# -------------------------------
# API Key
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


    st.success("✅ PDF processed successfully")

    # -------------------------------
    # LLM
    # -------------------------------
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512
    )


    # -------------------------------
    # Prompt
    # -------------------------------
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question ONLY using the context below.
    If the answer is not present, say "Not found in document".
    Keep the answer under 4 lines.

    Context:
    {context}

    Question:
    {question}
    """
    )


    # -------------------------------
    # RAG Chain (NO RetrievalQA)
    # -------------------------------
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------------------------------
    # Chat
    # -------------------------------
    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)
        st.markdown("### ✅ Answer")
        st.write(response)













