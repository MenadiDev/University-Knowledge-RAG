"""Loading PDFs and splitting them into chunks."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(pdf_path: Path, source_name: str | None = None) -> list[Document]:
    """Load one PDF. Each page becomes a Document.

    source_name overrides the "source" metadata (useful for uploads saved
    to a temporary path, so sources show the real file name).
    """
    documents = PyMuPDFLoader(str(pdf_path)).load()
    name = source_name or Path(pdf_path).name
    for doc in documents:
        doc.metadata["source"] = name
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)
