from jet.adapters.langchain.chat_ollama import ChatOllama, Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.tools import HumanInputRun
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
# Human as a tool

Human are AGI so they can certainly be used as a tool to help out AI agent 
when it is confused.
"""
logger.info("# Human as a tool")

# %pip install --upgrade --quiet  langchain-community


llm = ChatOllama(model="llama3.2")
math_llm = ChatOllama(temperature=0.0)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

"""
In the above code you can see the tool takes input directly from command line.
You can customize `prompt_func` and `input_func` according to your need (as shown below).
"""
logger.info(
    "In the above code you can see the tool takes input directly from command line.")

agent_chain.run("What's my friend Eric's surname?")

"""
## Configuring the Input Function

By default, the `HumanInputRun` tool uses the python `input` function to get input from the user.
You can customize the input_func to be anything you'd like.
For instance, if you want to accept multi-line input, you could do the following:
"""
logger.info("## Configuring the Input Function")


def get_input() -> str:
    logger.debug(
        "Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


tools = load_tools(["human", "ddg-search"], llm=math_llm, input_func=get_input)


tool = HumanInputRun(input_func=get_input)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run("I need help attributing a quote")

logger.info("\n\n[DONE]", bright=True)
