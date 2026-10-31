from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Define folder paths
DOCUMENTS_FOLDER = Path("documents")
CHROMA_FOLDER = "chroma_db"


# 2. Load all PDF documents
documents = []

pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in the documents folder.")
    exit()

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file.name}")

    loader = PyMuPDFLoader(str(pdf_file))
    documents.extend(loader.load())


# 3. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total PDF files: {len(pdf_files)}")
print(f"Total document chunks: {len(chunks)}")


# 4. Load the embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# 5. Store embeddings in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_FOLDER,
    collection_name="university_docs"
)

print("Documents successfully stored in ChromaDB!")