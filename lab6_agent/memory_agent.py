from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import ast
import operator
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from calendar_test import get_calendar_service, create_calendar_event

# Initialize the Ollama language model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Ollama model initialized successfully!")

# Initialize the embedding model
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)

# Connect to the existing ChromaDB database
vector_store = Chroma(
    collection_name="university_docs",
    persist_directory=str(Path(__file__).parent.parent / "chroma_db"),
    embedding_function=embeddings
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

print("University document database connected successfully!")

# RAG Search Tool
@tool
def rag_search_tool(question: str) -> str:
    """Search university documents and answer questions using their content."""

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "No relevant information found in the university documents."

    # Combine retrieved text
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Prepare the prompt
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

    # Collect source information
    sources = list(set(
        doc.metadata.get("source", "Unknown document")
        for doc in retrieved_docs
    ))

    return (
        f"{response.content}\n\n"
        f"Sources: {', '.join(sources)}"
    )

# Database location
DB_PATH = Path(__file__).parent / "agent_memory.db"


# Initialize the SQLite database
def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            information TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

    print("Memory database initialized successfully!")


# Run database initialization
initialize_database()


from langchain_core.tools import tool


# 1. Save Memory Tool
@tool
def save_memory(information: str) -> str:
    """Save useful information about the user into long-term memory."""

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "INSERT INTO memories (information) VALUES (?)",
        (information,)
    )

    connection.commit()
    connection.close()

    return f"Successfully saved to memory: {information}"


# 2. Memory Search Tool
@tool
def search_memory(query: str) -> str:
    """Search previously stored user information using keywords."""

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT information, created_at
        FROM memories
        WHERE information LIKE ?
        ORDER BY id DESC
        LIMIT 10
        """,
        (f"%{query}%",)
    )

    results = cursor.fetchall()
    connection.close()

    if not results:
        return "No matching information found in memory."

    return "\n".join(
        [f"Memory: {information} (Saved: {created_at})"
         for information, created_at in results]
    )

# Calculator Tool
@tool
def calculator_tool(expression: str) -> str:
    """Calculate basic mathematical expressions."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def calculate(node):

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            left = calculate(node.left)
            right = calculate(node.right)
            return allowed_operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](calculate(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        result = calculate(tree.body)
        return f"Result: {result}"

    except Exception as error:
        return f"Calculation error: {error}"

# Google Calendar - Check Today's Schedule
@tool
def check_calendar() -> str:
    """Check today's events in the user's Google Calendar."""

    try:
        service = get_calendar_service()

        now = datetime.now().astimezone()

        start_of_day = now.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        end_of_day = start_of_day + timedelta(days=1)

        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            maxResults=50,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])

        if not events:
            return "There are no events scheduled for today."

        result = ["Today's calendar events:"]

        for event in events:
            start = event["start"].get(
                "dateTime", event["start"].get("date")
            )

            result.append(
                f"- {event.get('summary', 'Untitled event')} | {start}"
            )

        return "\n".join(result)

    except Exception as e:
        return f"Calendar error: {str(e)}"

# Google Calendar - Create Event
@tool
def create_event(
    title: str,
    start_time: str,
    end_time: str,
    description: str = ""
) -> str:
    """
    Create a Google Calendar event.
    Use ISO format for start_time and end_time,
    including the timezone offset, e.g. 2026-11-16T10:00:00+05:30.
    """

    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        if start.tzinfo is None or end.tzinfo is None:
            return "Please provide both times with a timezone offset, such as +05:30."

        if end <= start:
            return "The event end time must be after its start time."

        event_link = create_calendar_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
            description=description
        )

        return f"Event created successfully! Calendar link: {event_link}"

    except Exception as e:
        return f"Calendar event creation failed: {str(e)}"    

# Register all available tools
tools = [
    calculator_tool,
    save_memory,
    search_memory,
    rag_search_tool,
    check_calendar,
    create_event
]

# Create a dictionary to access tools by name
tools_by_name = {
    tool.name: tool for tool in tools
}

# Bind tools to the language model
llm_with_tools = llm.bind_tools(tools)

print("AI tools registered successfully!")

# System instructions for the AI agent
system_message = SystemMessage(content="""
You are a helpful AI assistant with access to six tools.

MEMORY RULES:

1. When the user shares useful personal information,
   use save_memory to store it.

2. Whenever the user asks about previously shared
   information, ALWAYS use search_memory.

3. For questions about the user's name, use "name"
   as the search keyword.

4. Never invent memories.

CALCULATOR RULES:

Use calculator_tool whenever the user asks
a mathematical question.

UNIVERSITY KNOWLEDGE RULES:

1. Whenever the user asks a question about university
   documents, courses, academic regulations, degree
   programmes, or handbook information, ALWAYS use
   rag_search_tool.

2. Answer university questions using the information
   returned by rag_search_tool.

3. Do not invent university information.

4. If the information is unavailable in the documents,
   clearly explain that it was not found.

GOOGLE CALENDAR RULES:

1. When the user asks about today's schedule or
   today's calendar events, ALWAYS use check_calendar.

2. When the user explicitly asks to create a calendar
   event, use create_event.

3. Never invent event dates or times. If the title,
   date, start time, or end time is missing, ask
   the user for the missing information.

4. Use the Asia/Colombo timezone for Sri Lankan events.
   Include the +05:30 timezone offset in date/time values.

5. Only report an event as created when the tool
   confirms successful creation.

6. After calling check_calendar, your final answer
   MUST be based on the exact result returned by the tool.
   If the tool says there are no events scheduled,
   tell the user exactly that. Never replace the result
   with an invitation to create an event.    

GENERAL RULE:

Always select the appropriate tool before answering
questions that require stored information,
calculations, or university document retrieval.
""")

# Store conversation messages
messages = [system_message]


# Agent execution function
def run_agent(user_input):

    messages.append(HumanMessage(content=user_input))

    # Ask the LLM to process the request
    response = llm_with_tools.invoke(messages)

    # Execute requested tools
    while response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            selected_tool = tools_by_name[tool_call["name"]]

            result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        # Send tool results back to the LLM
        response = llm_with_tools.invoke(messages)

    messages.append(response)

    return response.content

# Main chat interface
if __name__ == "__main__":

    print("\n===== AI MEMORY AGENT =====")
    print("Ask me questions, share information, or request calculations.")
    print("Type 'exit' to stop the agent.\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        try:
            response = run_agent(user_input)
            print("Agent:", response)

        except Exception as error:
            print("Error:", error)


