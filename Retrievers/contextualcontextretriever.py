from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

load_dotenv()

# -----------------------------
# Groq LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
)

# -----------------------------
# Documents
# -----------------------------
docs = [
    Document(page_content="AI is used in healthcare for diagnosis and medical imaging."),
    Document(page_content="Machine learning improves predictions using data patterns."),
    Document(page_content="AI powers chatbots, assistants, and automation tools."),
    Document(page_content="Blockchain provides secure decentralized transactions."),
]

# -----------------------------
# HuggingFace Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Vector Store (FAISS)
# -----------------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# -----------------------------
# Base Retriever
# -----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Compressor (LLM extracts only relevant parts)
# -----------------------------
compressor = LLMChainExtractor.from_llm(llm)

# -----------------------------
# Contextual Compression Retriever
# -----------------------------
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

# -----------------------------
# Query
# -----------------------------
query = "how is AI used in healthcare?"

results = compression_retriever.invoke(query)

# -----------------------------
# Output
# -----------------------------
for i, doc in enumerate(results, 1):
    print(f"\nRESULT {i}")
    print("-" * 40)
    print(doc.page_content)