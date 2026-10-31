
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# 1. Load the embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# 2. Connect to the existing ChromaDB database
vector_store = Chroma(
    collection_name="university_docs",
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# 3. Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# 4. Load the Llama language model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 5. Ask questions
while True:

    question = input("\nAsk a university question (or exit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("\nNo relevant information found.")
        continue

    # Combine retrieved text
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Create the prompt
    prompt = f"""
    You are a university knowledge assistant.

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

    # Generate an answer
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    # Display sources
    print("\nSources:")

    for doc in retrieved_docs:
        print(doc.metadata.get("source", "Unknown document"))
