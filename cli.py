"""Ask questions about the documents from the terminal.

Run:  python cli.py
"""

from rag.chain import answer_question, format_sources, get_llm, get_vector_store


def main() -> None:
    vector_store = get_vector_store()
    llm = get_llm()

    while True:
        question = input("\nAsk a university question (or exit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        answer, docs = answer_question(question, vector_store, llm)

        print("\nAnswer:")
        print(answer)

        if docs:
            print("\nSources:")
            for label in format_sources(docs):
                print(f"- {label}")


if __name__ == "__main__":
    main()
