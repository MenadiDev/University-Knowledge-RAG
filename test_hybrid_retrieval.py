
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import hashlib


# 1. Initialize embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# 2. Connect to existing ChromaDB collection
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


# 5. Translation function
def translate_query(question, language):

    if language == "English":
        return question

    prompt = f"""
    Translate the following {language} question into English.

    Preserve the exact meaning.
    Do not answer the question.
    Return only the English translation.

    Question:
    {question}

    English translation:
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# 6. Reciprocal Rank Fusion
def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_n=5
):

    fused_scores = {}
    documents = {}

    for results in result_lists:

        for rank, (doc, distance) in enumerate(
            results, start=1
        ):

            # Create a stable document identifier
            source_id = (
                doc.metadata.get("file_hash")
                or doc.metadata.get("source", "unknown")
            )

            doc_id = hashlib.sha256(
                (
                    str(source_id)
                    + "_"
                    + str(doc.metadata.get("page", ""))
                    + "_"
                    + doc.page_content
                ).encode("utf-8")
            ).hexdigest()
            
            # Store document
            documents[doc_id] = doc

            # RRF scoring
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0

            fused_scores[doc_id] += 1 / (k + rank)

    # Sort documents by fused score
    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    return [
        (documents[doc_id], fused_scores[doc_id])
        for doc_id in ranked_ids[:top_n]
    ]


# 7. Run hybrid retrieval
print("\nHYBRID MULTILINGUAL RAG TEST")
print("=" * 60)


for language, question in questions.items():

    print(f"\n{language} QUESTION:")
    print(question)

    # Direct multilingual retrieval
    direct_results = (
        vector_store.similarity_search_with_score(
            question,
            k=5
        )
    )

    # Translation-based retrieval
    english_query = translate_query(
        question,
        language
    )

    print("\nTRANSLATED QUERY:")
    print(english_query)

    translation_results = (
        vector_store.similarity_search_with_score(
            english_query,
            k=5
        )
    )

    # Combine both retrieval result lists
    hybrid_results = reciprocal_rank_fusion(
        [direct_results, translation_results],
        k=60,
        top_n=5
    )

    # Display hybrid results
    print("\nTOP 5 HYBRID RETRIEVED CHUNKS")
    print("-" * 60)

    for rank, (doc, score) in enumerate(
        hybrid_results,
        start=1
    ):

        print(f"\nResult {rank}")

        print(
            "Source:",
            doc.metadata.get("source", "Unknown")
        )

        print(
            "Page:",
            doc.metadata.get("page", "Unknown")
        )

        print(
            "RRF Score:",
            round(score, 6)
        )

        print(
            "Content:",
            doc.page_content[:500]
        )

        print("-" * 40)
