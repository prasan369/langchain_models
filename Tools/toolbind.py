from langchain_groq import ChatGroq
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int):
    """Adds two numbers."""
    return a + b

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

llm_with_tools = llm.bind_tools([add])

response = llm_with_tools.invoke("What is 5 + 3?")
print(response)