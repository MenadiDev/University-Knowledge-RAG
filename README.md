# 🎓 University Knowledge RAG

A local question-answering assistant for university documents. It reads PDFs, stores them in a ChromaDB vector database, and answers questions using only what's in those documents, with page-level sources. Everything runs locally through [Ollama](https://ollama.com).

## Features

- Chat web app (Streamlit) with PDF upload
- Terminal chat for quick testing
- Answers grounded in your documents, with source file and page shown
- Fully local: no API keys needed

## Project structure

```
├── app.py            # Streamlit web app
├── cli.py            # Terminal chat
├── ingest.py         # Load all PDFs in documents/ into the database
├── rag/
│   ├── config.py     # Model names, paths, chunk size, top-k
│   ├── loader.py     # PDF loading and chunking
│   └── chain.py      # Vector store, LLM and question answering
├── assets/
│   └── style.css     # App theme
├── documents/        # Source PDFs
└── chroma_db/        # Vector database (generated, not in git)
```

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama pull llama3.2:3b
   ```

2. Create a virtual environment and install dependencies (Python 3.10+):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Put your PDFs in `documents/` and build the database:

   ```bash
   python ingest.py
   ```

## Usage

Web app:

```bash
streamlit run app.py
```

Terminal:

```bash
python cli.py
```

You can also upload PDFs from the app's sidebar instead of running `ingest.py`.

## Configuration

Edit `rag/config.py` to change the models, chunk size or how many chunks are retrieved.

To rebuild the database from scratch, delete `chroma_db/` and run `python ingest.py` again.
