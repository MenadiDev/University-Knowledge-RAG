
import ast
import operator
import subprocess
import time

import pyautogui
import pygetwindow as gw
import os
import platform
from datetime import datetime
from pathlib import Path

import psutil

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# 1. Safe mathematical operations
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def safe_calculate(expression):
    """Evaluate basic arithmetic without using Python eval()."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            return OPERATORS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            if isinstance(node.op, ast.UAdd):
                return value

        raise ValueError("Unsupported mathematical expression.")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


# 2. Create the calculator tool

@tool
def calculator_tool(expression: str) -> str:
    """Calculate a mathematical expression and return the result."""

    try:
        expression = (
            expression.replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("=", "")
            .strip()
        )

        result = safe_calculate(expression)

        return f"The result of {expression} is {result}."

    except Exception as error:
        return f"Calculation failed: {error}"


# 1. Date and Time Tool
@tool
def get_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 2. Notepad Tool
@tool
def open_notepad(text: str) -> str:
    """Open Windows Notepad and display the user-provided text."""

    try:
        import subprocess
        import tempfile

        # Create a temporary text file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8"
        ) as file:
            file.write(text)
            file_path = file.name

        # Open the text file in Notepad
        subprocess.Popen(["notepad.exe", file_path])

        return f"Notepad opened with the following text:\n{text}"

    except Exception as error:
        return f"Could not open Notepad: {error}"



# 3. File Search Tool
@tool
def search_files(directory: str, extension: str) -> str:
    """Search a directory for files with a given extension."""

    try:
        folder = Path(directory).expanduser()

        if not folder.is_dir():
            return "The specified directory does not exist."

        extension = extension if extension.startswith(".") else "." + extension

        results = list(folder.rglob(f"*{extension}"))

        if not results:
            return "No matching files found."

        return "\n".join(str(file) for file in results[:100])

    except Exception as error:
        return f"File search failed: {error}"


# 4. File Creation Tool
@tool
def create_text_file(filename: str, content: str) -> str:
    """Create a text file in the lab6_agent folder."""

    try:
        safe_name = Path(filename).name

        if not safe_name.endswith(".txt"):
            safe_name += ".txt"

        output_folder = Path("created_files")
        output_folder.mkdir(exist_ok=True)

        file_path = output_folder / safe_name
        file_path.write_text(content, encoding="utf-8")

        return f"File created successfully: {file_path.resolve()}"

    except Exception as error:
        return f"File creation failed: {error}"


# 5. System Information Tool
@tool
def get_system_info() -> str:
    """Get operating system, processor and memory information."""

    try:
        os_name = platform.system()
        os_version = platform.version()
        processor = platform.processor()

        total_memory = psutil.virtual_memory().total / (1024 ** 3)
        available_memory = psutil.virtual_memory().available / (1024 ** 3)

        return (
            f"Operating System: {os_name}\n"
            f"OS Version: {os_version}\n"
            f"Processor: {processor}\n"
            f"Total RAM: {total_memory:.2f} GB\n"
            f"Available RAM: {available_memory:.2f} GB"
        )

    except Exception as error:
        return f"Could not retrieve system information: {error}"
    



# 3. Initialize the local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 4. Register the calculator tool with the LLM
tools = [
    calculator_tool,
    get_datetime,
    open_notepad,
    search_files,
    create_text_file,
    get_system_info
]

llm_with_tools = llm.bind_tools(tools)

tools_by_name = {
    tool.name: tool for tool in tools
}



# 5. Create the agent execution loop

messages = [
    SystemMessage(
        content="""
        You are a helpful AI assistant with access to six tools.

        1. calculator_tool: Performs mathematical calculations.
        2. get_datetime: Gets the current date and time.
        3. open_notepad: Opens Windows Notepad.
        4. search_files: Searches for files in a directory.
        5. create_text_file: Creates a text file.
        6. get_system_info: Retrieves computer specifications.

        Use the appropriate tool whenever the user requests
        an action that one of these tools can perform.

        Do not invent tool results.
        Explain the result clearly to the user.
        """
    )
]

print("Multi-Tool AI Agent is ready!")
print("Available tools: Calculator, Date & Time, Notepad,")
print("File Search, File Creation, System Information")
print("Type 'exit' to stop.\n")

while True:

    question = input("You: ")

    if question.lower().strip() == "exit":
        print("Agent: Goodbye!")
        break

    messages.append(HumanMessage(content=question))

    while True:

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check whether the LLM requested a tool
        if not response.tool_calls:
            print("\nAgent:", response.content)
            print()
            break

        # Execute each requested tool
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\n[Executing tool: {tool_name}]")

            if tool_name in tools_by_name:

                try:
                    result = tools_by_name[tool_name].invoke(tool_args)

                except Exception as error:
                    result = f"Tool execution error: {error}"

            else:
                result = f"Unknown tool: {tool_name}"

            print("[Tool result]:", result)

            # Send the result back to the LLM
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

