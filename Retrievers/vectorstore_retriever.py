from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -----------------------------
# 1. Sample knowledge base
# -----------------------------
docs = [
    Document(page_content="Artificial intelligence is transforming industries like healthcare, finance, and education."),
    Document(page_content="Machine learning models improve their performance by learning from data patterns."),
    Document(page_content="Photosynthesis is the process by which plants convert sunlight into energy."),
    Document(page_content="Black holes are regions in space where gravity is so strong that nothing can escape."),
    Document(page_content="Quantum computing uses qubits instead of classical bits to perform complex calculations."),
    Document(page_content="Blockchain technology is used to create secure and decentralized digital transactions."),
]

# -----------------------------
# 2. Embeddings model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 3. Create vector store
# -----------------------------
vectorstore = FAISS.from_documents(docs, embeddings)

# -----------------------------
# 4. Query (NEW INTERESTING QUERY)
# -----------------------------
query = "How does AI learn from data?"

results = vectorstore.similarity_search(query, k=2)

# -----------------------------
# 5. Output results
# -----------------------------
for i, doc in enumerate(results):
    print(f"\nResult {i+1}")
    print("-" * 40)
    print(doc.page_content)