
from langchain_ollama import OllamaEmbeddings

# Load your existing embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

# Test questions in three languages
questions = [
    "What are the examination registration requirements?",
    "විභාගයට ලියාපදිංචි වීමේ අවශ්‍යතා මොනවාද?",
    "தேர்வுக்கான பதிவு தேவைகள் என்ன?"
]

print("\nMULTILINGUAL EMBEDDING TEST")
print("-" * 40)

for question in questions:

    vector = embeddings.embed_query(question)

    print("\nQuestion:", question)
    print("Embedding dimensions:", len(vector))
    print("Embedding generated:", bool(vector))
