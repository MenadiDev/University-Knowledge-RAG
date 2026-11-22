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

print("Clean collection count:", vector_store._collection.count())