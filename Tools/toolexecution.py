from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int):
    """Adds two numbers."""
    return a + b

llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools([add])

query = HumanMessage("can you add 4 by 10")
messages = [query]

result = llm_with_tools.invoke(messages)
messages.append(result)

# Fix: wrap tool output in ToolMessage
tool_call = result.tool_calls[0]
tool_output = add.invoke(tool_call["args"])

messages.append(ToolMessage(
    content=str(tool_output),
    tool_call_id=tool_call["id"]   # links result back to the specific tool call
))

response = llm_with_tools.invoke(messages)
print(response.content)