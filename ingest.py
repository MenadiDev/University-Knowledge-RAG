"""Load every PDF in documents/ into ChromaDB.

Run:  python ingest.py
Note: running it twice adds the same chunks twice. To start fresh,
delete the chroma_db/ folder first.
"""

from rag.chain import add_chunks, get_vector_store
from rag.config import DOCUMENTS_DIR
from rag.loader import load_pdf, split_documents


def main() -> None:
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {DOCUMENTS_DIR}")
        return

    documents = []
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        documents.extend(load_pdf(pdf_file))

    chunks = split_documents(documents)
    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Total document chunks: {len(chunks)}")

    added = add_chunks(get_vector_store(), chunks)
    print(f"Stored {added} chunks in ChromaDB.")


if __name__ == "__main__":
    main()
