from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq



# Load API keys

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
)



# YouTube Transcript

video_id = "759f-CHW-S8"

transcript = YouTubeTranscriptApi().fetch(video_id)

text = " ".join(chunk.text for chunk in transcript)



# Convert to Document

docs = [Document(page_content=text)]



# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)



# Embeddings (HuggingFace)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# FAISS Vector Store

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})



# Chat Loop (RAG)

print("\nYouTube RAG chatbot ready! Type 'exit' to stop.\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant. Answer using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\n--- ANSWER ---")
    print(response.content)