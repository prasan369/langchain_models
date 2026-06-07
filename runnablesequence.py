from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt=PromptTemplate(
    template="Write a joke about a {topic}",
    input_variables=['topic']
)
model=ChatGroq(model="openai/gpt-oss-120b")
parser=StrOutputParser()
chain=RunnableSequence(prompt,model,parser)
print(chain.invoke({'topic':'prasan'}))