"""Models, vector store and question answering, shared by the app and CLI."""

import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K,
)

PROMPT_TEMPLATE = """You are a university knowledge assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
clearly say that the information was not found
in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

NOT_FOUND_MESSAGE = "I couldn't find relevant information in the documents."


def get_vector_store() -> Chroma:
    """Connect to (or create) the ChromaDB collection."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def add_chunks(vector_store: Chroma, chunks: list[Document]) -> int:
    """Embed chunks and store them. Returns how many were added."""
    if not chunks:
        return 0
    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def answer_question(
    question: str, vector_store: Chroma, llm: ChatOllama
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and ask the LLM.

    Returns (answer_text, retrieved_documents).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return NOT_FOUND_MESSAGE, []

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content, docs


def format_sources(docs: list[Document]) -> list[str]:
    """Unique "file (Page N)" labels, in retrieval order.

    PyMuPDF page numbers start at 0, so we add 1 for display.
    """
    seen = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        label = f"{source} (Page {page + 1})" if isinstance(page, int) else source
        if label not in seen:
            seen.append(label)
    return seen
