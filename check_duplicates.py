from collections import Counter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

vector_store = Chroma(
    collection_name="university_docs",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retrieve stored documents and metadata
data = vector_store.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]
metadatas = data["metadatas"]

# Count exact duplicate chunks
counts = Counter(documents)

print("\nCHROMADB DUPLICATE CHECK")
print("=" * 50)

print("Total stored chunks:", len(documents))
print("Unique chunks:", len(counts))
print("Duplicate chunk groups:",
      sum(1 for count in counts.values() if count > 1))

print("\nDUPLICATE DETAILS")
print("-" * 50)

shown = 0

for document, count in counts.items():

    if count > 1:
        print(f"\nOccurrences: {count}")
        print("Content:", document[:250])

        for i, stored_document in enumerate(documents):
            if stored_document == document:
                print(
                    "Source:", metadatas[i].get("source"),
                    "| Page:", metadatas[i].get("page")
                )

        shown += 1

        if shown == 5:
            break

print("\nCheck completed. No database changes made.")