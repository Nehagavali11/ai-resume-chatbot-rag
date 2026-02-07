# 🧠 AI Resume Chatbot (RAG-based)

An **AI-powered Resume Question–Answering Chatbot** built using **Retrieval-Augmented Generation (RAG)**.  
This application allows users or recruiters to ask natural language questions about a resume PDF and receive **accurate, contextual answers** in real time.

---

## 🚀 Features

- 📄 PDF resume ingestion  
- ✂️ Intelligent text chunking  
- 🔍 Semantic search using FAISS  
- 🧠 Context-aware answers using LLM  
- ⚡ Fast responses (~1–2 seconds)  
- 💻 Works on CPU-only systems (8 GB RAM)  
- 🌐 Simple and interactive Streamlit UI  

---

## 🏗️ System Architecture (RAG Pipeline)

1. **Document Loader** – Loads resume PDF  
2. **Text Splitter** – Splits content into overlapping chunks  
3. **Embedding Model** – HuggingFace `all-MiniLM-L6-v2`  
4. **Vector Store** – FAISS for similarity search  
5. **LLM** – LLaMA-3.1 via Groq API  
6. **Frontend** – Streamlit web interface  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Framework:** LangChain  
- **LLM:** LLaMA-3.1 (Groq API)  
- **Embeddings:** HuggingFace Sentence Transformers  
- **Vector Database:** FAISS  
- **Frontend:** Streamlit  

---
## SnapShot
<img width="1744" height="863" alt="Screenshot (807)" src="https://github.com/user-attachments/assets/b7e4521d-7ab8-47f2-b120-603ac2c69b41" />


<img width="1871" height="792" alt="Screenshot (805)" src="https://github.com/user-attachments/assets/7b87a9fe-e923-4d5c-ad9b-94ec0514df7d" />

---
## 📦 Installation

```bash
pip install langchain langchain-core langchain-community langchain-groq faiss-cpu sentence-transformers streamlit


