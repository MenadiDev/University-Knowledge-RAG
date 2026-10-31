
import streamlit as st
import tempfile
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# 1. Page configuration
st.set_page_config(
    page_title="Uni AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# Pastel light theme
st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #FFFAFD;
    color: #49364D;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F9EFF6;
    border-right: 1px solid #F0DFEA;
}

/* Main headings */
h1, h2, h3 {
    color: #49364D;
    font-family: 'Segoe UI', sans-serif;
}

/* Paragraphs */
p, label {
    color: #705E73;
}

/* Buttons */
.stButton > button {
    background-color: #EBD9EC;
    color: #65466C;
    border: 1px solid #E2CBE2;
    border-radius: 12px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #DFC4E0;
    border-color: #D4B4D7;
    color: #49364D;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #D9B9D5;
    border-radius: 14px;
    padding: 12px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #FFFFFF;
    border: 1px solid #EBD9EC;
    border-radius: 16px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid #F0E5EF;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 12px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #F0DFEA;
    border-radius: 14px;
}

/* Success messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}
/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #49364D !important;
}

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #705E73;
}

/* PDF uploader */
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 1px dashed #D9B9D5 !important;
    border-radius: 12px;
}

[data-testid="stFileUploader"] section button {
    background-color: #EBD9EC !important;
    color: #65466C !important;
    border: 1px solid #E2CBE2 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD9EC !important;
    border-radius: 16px !important;
}

[data-testid="stChatInput"] textarea {
    color: #49364D !important;
    background-color: #FFFFFF !important;
}

/* Main toolbar */
[data-testid="stToolbar"] {
    background-color: #FFFAFD !important;
}

/* Bottom chat area */
[data-testid="stBottom"] {
    background-color: #FFFAFD !important;
}

/* Remove the dark bottom background */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background-color: #FFFAFD !important;
}

/* Chat input outer container */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EBD9EC !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 8px rgba(100, 70, 108, 0.06);
}

/* Actual text input area */
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #49364D !important;
    caret-color: #49364D !important;
}

/* Placeholder text */
[data-testid="stChatInput"] textarea::placeholder {
    color: #A18BA5 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


st.title("🎓 Uni AI Assistant")
st.caption("Ask questions about your university documents.")


# 2. Initialize models and database
@st.cache_resource
def initialize_rag():

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    vector_store = Chroma(
        collection_name="university_docs",
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return vector_store, llm


vector_store, llm = initialize_rag()


# 3. Sidebar - PDF uploads
st.sidebar.header("Manage Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# 4. Process uploaded PDFs
if st.sidebar.button("Process Documents"):

    if uploaded_files:

        all_documents = []

        with st.spinner("Reading and processing PDFs..."):

            for uploaded_file in uploaded_files:

                with tempfile.TemporaryDirectory() as temp_dir:

                    pdf_path = Path(temp_dir) / uploaded_file.name
                    pdf_path.write_bytes(uploaded_file.getvalue())

                    loader = PyMuPDFLoader(str(pdf_path))
                    pdf_documents = loader.load()

                    for doc in pdf_documents:
                        doc.metadata["source"] = uploaded_file.name

                    all_documents.extend(pdf_documents)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_documents(all_documents)

            # Generate embeddings and store in ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]

            vector_store.add_documents(
                documents=chunks,
                ids=ids
            )

        st.sidebar.success(
            f"Successfully processed {len(uploaded_files)} PDF(s)!"
        )

        st.sidebar.write(f"Total chunks added: {len(chunks)}")

    else:
        st.sidebar.warning("Please upload at least one PDF.")


# 5. Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# 6. Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 7. Chat input
question = st.chat_input("Ask your university question...")


if question:

    # Display user question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant documents
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            retriever = vector_store.as_retriever(
                search_kwargs={"k": 4}
            )

            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:

                answer = "I couldn't find relevant information in the uploaded documents."

            else:

                context = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )

                prompt = f"""
                You are a university knowledge assistant.

                Answer using ONLY the information in the context.

                If the answer is not available in the context,
                clearly state that it was not found in the documents.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """

                response = llm.invoke(prompt)
                answer = response.content

            st.markdown(answer)

            # Display sources
            if retrieved_docs:

                unique_sources = {}

                for doc in retrieved_docs:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")

                    unique_sources[(source, page)] = True

                with st.expander("📄 View Sources"):
                    for source, page in unique_sources:
                        st.write(f"- {source} (Page {page})")

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
