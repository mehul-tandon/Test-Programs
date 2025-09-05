from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

#Agent Demo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
output =agent.run("How old is will Gal Gadot be in 2026?")
print(output)
