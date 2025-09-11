from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import BearlyInterpreterTool
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Bearly Code Interpreter

> Bearly Code Interpreter allows for remote execution of code. This makes it perfect for a code sandbox for agents, to allow for safe implementation of things like Code Interpreter

Get your api key here: https://bearly.ai/dashboard/developers
"""
logger.info("# Bearly Code Interpreter")

# %pip install --upgrade --quiet langchain-community

"""
In this notebook, we will create an example of an agent that uses Bearly to interact with data
"""
logger.info("In this notebook, we will create an example of an agent that uses Bearly to interact with data")



"""
Initialize the interpreter
"""
logger.info("Initialize the interpreter")

bearly_tool = BearlyInterpreterTool()

"""
Let's add some files to the sandbox
"""
logger.info("Let's add some files to the sandbox")

bearly_tool.add_file(
    source_path="sample_data/Bristol.pdf", target_path="Bristol.pdf", description=""
)
bearly_tool.add_file(
    source_path="sample_data/US_GDP.csv", target_path="US_GDP.csv", description=""
)

"""
Create a `Tool` object now. This is necessary, because we added the files, and we want the tool description to reflect that
"""
logger.info("Create a `Tool` object now. This is necessary, because we added the files, and we want the tool description to reflect that")

tools = [bearly_tool.as_tool()]

tools[0].name

logger.debug(tools[0].description)

"""
Initialize an agent
"""
logger.info("Initialize an agent")

llm = ChatOllama(model="llama3.2")
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

agent.run("What is the text on page 3 of the pdf?")

agent.run("What was the US GDP in 2019?")

agent.run("What would the GDP be in 2030 if the latest GDP number grew by 50%?")

agent.run("Create a nice and labeled chart of the GDP growth over time")

logger.info("\n\n[DONE]", bright=True)