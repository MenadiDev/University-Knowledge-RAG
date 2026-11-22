from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

questions = [
    "පරිගණක විද්‍යා උපාධිය සඳහා කොපමණ කාලයක් ගතවේද?",
    "පරිගණක විද්‍යා උපාධිය වසර කීයක උපාධියක්ද?",
    "පරිගණක විද්‍යාව හැදෑරීමට කොපමණ කාලයක් ගත වෙනවාද?"
]

for question in questions:

    prompt = f"""
    Translate this Sinhala question into English.

    Preserve the exact meaning.
    Do not answer the question.
    Do not add explanations.

    Sinhala:
    {question}

    English:
    """

    response = llm.invoke(prompt)

    print("\nSinhala:")
    print(question)

    print("English:")
    print(response.content.strip())

    print("-" * 50)