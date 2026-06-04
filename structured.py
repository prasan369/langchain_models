from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model=ChatGroq(model="openai/gpt-oss-120b")

class Review(TypedDict):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(Review)
result=structured_model.invoke("The movie was very short and sweet overall a very lovable movie")

print(result)