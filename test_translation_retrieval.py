
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# 1. Initialize embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# 2. Connect to clean Chroma collection
vector_store = Chroma(
    collection_name="university_docs_clean",
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# 3. Initialize translation LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 4. Test questions
questions = {
    "English": "How long is the Computer Science degree?",

    "Sinhala": "පරිගණක විද්‍යා උපාධිය සඳහා කොපමණ කාලයක් ගතවේද?",

    "Tamil": "கணினி அறிவியல் பட்டப்படிப்பு எத்தனை ஆண்டுகள்?"
}


print("\nTRANSLATION-BASED RETRIEVAL TEST")
print("=" * 55)


# 5. Translate and retrieve
for language, question in questions.items():

    print(f"\n{language} QUESTION:")
    print(question)

    # Translate non-English questions
    if language == "English":
        english_query = question

    else:
        translation_prompt = f"""
        You are a professional translator working with university academic documents.

        Translate the following question from {language} into English.

        IMPORTANT:
        - Preserve the original meaning exactly.
        - This question is about a university degree programme.
        - Interpret words about time or duration in the context of the length of academic study.
        - Do not interpret the question as referring to a game,countdown, event, or timer
        - Do not answer the question.
        - Do not add explanations.
        - Return only the English translation.
       
        Original question:
        {question}

        English translation:
        """

        response = llm.invoke(translation_prompt)
        english_query = response.content.strip()

    print("\nTRANSLATED QUERY:")
    print(english_query)

    # Retrieve using English query
    results = vector_store.similarity_search_with_score(
        english_query,
        k=5
    )

    print("\nTOP 3 RETRIEVED CHUNKS:")
    print("-" * 55)

    for i, (doc, score) in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Distance:", round(score, 4))
        print("Content:", doc.page_content[:500])
        print("-" * 40)
