from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

vector_store = Chroma(
    collection_name="university_docs_clean",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

data = vector_store.get(
    include=["documents", "metadatas"]
)

# Pages containing programme descriptions and key features
target_pages = [4, 5, 8, 11, 14, 17]

print("\n========== FULL PROGRAMME INFORMATION ==========")

chunks = []

for document, metadata in zip(
    data["documents"],
    data["metadatas"]
):
    page = metadata.get("page", -1)

    if page in target_pages:
        chunks.append((page, document))

# Sort by page number
chunks.sort(key=lambda x: x[0])

for page, content in chunks:

    print(f"\n{'=' * 60}")
    print(f"PDF PAGE INDEX: {page}")
    print("=" * 60)

    print(content)