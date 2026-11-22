from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Connect to the existing clean collection
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

vector_store = Chroma(
    collection_name="university_docs_clean",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retrieve stored documents and metadata
data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\n========== DOCUMENT COLLECTION ==========")
print("Total indexed chunks:", len(data["documents"]))

# Identify unique PDF sources and pages
sources = {}

for document, metadata in zip(
    data["documents"],
    data["metadatas"]
):
    source = metadata.get("source", "Unknown")
    page = metadata.get("page", "Unknown")

    if source not in sources:
        sources[source] = {}

    sources[source][page] = document

# Display available content
for source, pages in sources.items():

    print(f"\n\nPDF: {source}")
    print("Unique pages:", len(pages))

    for page, content in sorted(
        pages.items(),
        key=lambda x: str(x[0])
    ):
        print(f"\n--- Page {page} ---")
        print(content[:350].replace("\n", " "))