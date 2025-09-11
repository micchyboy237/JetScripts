from dotenv import find_dotenv, load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import HumanMessage
from langchain_prolog import PrologConfig, PrologRunnable, PrologTool
from langgraph.prebuilt import create_react_agent
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
# Prolog

LangChain tools that use Prolog rules to generate answers.

## Overview

The PrologTool class allows the generation of langchain tools that use Prolog rules to generate answers.

## Setup

Let's use the following Prolog rules in the file family.pl:

parent(john, bianca, mary).\
parent(john, bianca, michael).\
parent(peter, patricia, jennifer).\
partner(X, Y) :- parent(X, Y, _).
"""
logger.info("# Prolog")


TEST_SCRIPT = "family.pl"

"""
## Instantiation

First create the Prolog tool:
"""
logger.info("## Instantiation")

schema = PrologRunnable.create_schema("parent", ["men", "women", "child"])
config = PrologConfig(
    rules_file=TEST_SCRIPT,
    query_schema=schema,
)
prolog_tool = PrologTool(
    prolog_config=config,
    name="family_query",
    description="""
        Query family relationships using Prolog.
        parent(X, Y, Z) implies only that Z is a child of X and Y.
        Input can be a query string like 'parent(john, X, Y)' or 'john, X, Y'"
        You have to specify 3 parameters: men, woman, child. Do not use quotes.
    """,
)

"""
## Invocation

### Using a Prolog tool with an LLM and function calling
"""
logger.info("## Invocation")


load_dotenv(find_dotenv(), override=True)



"""
To use the tool, bind it to the LLM model:
"""
logger.info("To use the tool, bind it to the LLM model:")

llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools([prolog_tool])

"""
and then query the model:
"""
logger.info("and then query the model:")

query = "Who are John's children?"
messages = [HumanMessage(query)]
response = llm_with_tools.invoke(messages)

"""
The LLM will respond with a tool call request:
"""
logger.info("The LLM will respond with a tool call request:")

messages.append(response)
response.tool_calls[0]

"""
The tool takes this request and queries the Prolog database:
"""
logger.info("The tool takes this request and queries the Prolog database:")

tool_msg = prolog_tool.invoke(response.tool_calls[0])

"""
The tool returns a list with all the solutions for the query:
"""
logger.info("The tool returns a list with all the solutions for the query:")

messages.append(tool_msg)
tool_msg

"""
That we then pass to the LLM, and the LLM answers the original query using the tool response:
"""
logger.info("That we then pass to the LLM, and the LLM answers the original query using the tool response:")

answer = llm_with_tools.invoke(messages)
logger.debug(answer.content)

"""
## Chaining

### Using a Prolog Tool with an agent

To use the prolog tool with an agent, pass it to the agent's constructor:
"""
logger.info("## Chaining")


agent_executor = create_react_agent(llm, [prolog_tool])

"""
The agent takes the query and use the Prolog tool if needed:
"""
logger.info("The agent takes the query and use the Prolog tool if needed:")

messages = agent_executor.invoke({"messages": [("human", query)]})

"""
Then the agent receives​ the tool response and generates the answer:
"""
logger.info("Then the agent receives​ the tool response and generates the answer:")

messages["messages"][-1].pretty_logger.debug()

"""
## API reference

See https://langchain-prolog.readthedocs.io/en/latest/modules.html for detail.
"""
logger.info("## API reference")


logger.info("\n\n[DONE]", bright=True)