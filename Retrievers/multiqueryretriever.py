from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ FIXED MODEL
llm = ChatGroq(
    model="openai/gpt-oss-120b",
)

docs = [
    Document(page_content="AI is used in healthcare for diagnosis."),
    Document(page_content="Machine learning improves predictions using data."),
    Document(page_content="AI powers chatbots and virtual assistants."),
    Document(page_content="Blockchain is used for secure transactions."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

query = "applications of AI in real life"

results = retriever.invoke(query)

for doc in results:
    print("\n---")
    print(doc.page_content)