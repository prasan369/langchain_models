from langchain_community.retrievers import WikipediaRetriever

# Create retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Query
query = "Bhagwat gita best verses"

try:
    docs = retriever.invoke(query)

    print(f"Found {len(docs)} documents\n")

    for i, doc in enumerate(docs, 1):
        print(f"Document {i}")
        print("-" * 50)
        print(doc.page_content[:1000])  # First 1000 characters
        print("\n")

except Exception as e:
    print("Error:", e)