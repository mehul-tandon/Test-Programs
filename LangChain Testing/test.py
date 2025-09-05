from dotenv import load_dotenv
load_dotenv()

import langchain
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

prompt = PromptTemplate.from_template("What is the name of the e commerce store that sells {product}?")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
chain1 = LLMChain(llm=llm, prompt=prompt)

prompt = PromptTemplate.from_template("What are the names of the products at {store}?")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
chain2 = LLMChain(llm=llm, prompt=prompt)

chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True )
chain.run("candles")