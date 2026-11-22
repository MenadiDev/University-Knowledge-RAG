from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Load the existing embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

# Connect to the existing ChromaDB collection
vector_store = Chroma(
    collection_name="university_docs_clean",
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Test questions in three languages
questions = {
    "English": "How long is the Computer Science degree?",

    "Sinhala": "පරිගණක විද්‍යා උපාධිය සඳහා කොපමණ කාලයක් ගතවේද?",

    "Tamil": "கணினி அறிவியல் பட்டப்படிப்பு எத்தனை ஆண்டுகள்?"
}

print("\nCROSS-LINGUAL RETRIEVAL TEST")
print("=" * 50)

for language, question in questions.items():

    print(f"\n{language} QUESTION:")
    print(question)

    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    print("\nTOP 3 RETRIEVED DOCUMENT CHUNKS:")

    for i, (doc, score) in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Source:", doc.metadata.get("source", "Unknown"))
        print("Page:", doc.metadata.get("page", "Unknown"))
        print("Distance:", round(score, 4))
        print("Content:", doc.page_content[:400])
        print("-" * 40)
