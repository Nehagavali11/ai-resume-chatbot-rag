from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("✅ Ingestion complete")
