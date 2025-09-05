from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

#Memory in LLMs 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
prompt = PromptTemplate.from_template("What is the name of the ecommerce store that sells {product} in india?")
chain1=LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory())
output = chain1.run("fruits")
output = chain1.run("Electronics")
output = chain1.run("furniture")
output = chain1.run("PC parts")
print(chain1.memory.buffer)
print(output)