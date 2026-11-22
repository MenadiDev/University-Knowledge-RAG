"""Streamlit web app for the University Knowledge Assistant.

Run:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from rag.chain import (
    add_chunks,
    answer_question,
    format_sources,
    get_llm,
    get_vector_store,
)
from rag.config import STYLE_FILE
from rag.loader import load_pdf, split_documents


# 1. Page setup
st.set_page_config(
    page_title="Uni AI Assistant",
    page_icon="🎓",
    layout="wide",
)

st.markdown(f"<style>{STYLE_FILE.read_text()}</style>", unsafe_allow_html=True)

st.title("🎓 Uni AI Assistant")
st.caption("Ask questions about your university documents.")


# 2. Models and database (created once, then cached)
@st.cache_resource
def initialize_rag():
    return get_vector_store(), get_llm()


vector_store, llm = initialize_rag()


# 3. Sidebar: upload and process PDFs
st.sidebar.header("Manage Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.sidebar.button("Process Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        all_documents = []
        with st.spinner("Reading and processing PDFs..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    pdf_path = Path(temp_dir) / uploaded_file.name
                    pdf_path.write_bytes(uploaded_file.getvalue())
                    all_documents.extend(
                        load_pdf(pdf_path, source_name=uploaded_file.name)
                    )

            chunks = split_documents(all_documents)
            added = add_chunks(vector_store, chunks)

        st.sidebar.success(f"Successfully processed {len(uploaded_files)} PDF(s)!")
        st.sidebar.write(f"Total chunks added: {added}")


# 4. Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 5. Chat input
question = st.chat_input("Ask your university question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            answer, docs = answer_question(question, vector_store, llm)

        st.markdown(answer)

        if docs:
            with st.expander("📄 View Sources"):
                for label in format_sources(docs):
                    st.write(f"- {label}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
