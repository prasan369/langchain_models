from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Artificial intelligence is transforming healthcare and education."),
    Document(page_content="AI is used in self-driving cars and robotics."),
    Document(page_content="Machine learning improves systems by learning from data."),
    Document(page_content="Deep learning is a subset of machine learning using neural networks."),
    Document(page_content="Blockchain provides secure and decentralized transactions."),
    Document(page_content="Quantum computing uses qubits to solve complex problems."),
    Document(page_content="AI is also widely used in customer service chatbots."),
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

query = "applications of AI in real life"

results = vectorstore.max_marginal_relevance_search(
    query=query,
    k=3,        # final results
    fetch_k=10  # candidates to choose from
)


print("\nTOP RESULTS USING MMR:\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}")
    print("-" * 40)
    print(doc.page_content)