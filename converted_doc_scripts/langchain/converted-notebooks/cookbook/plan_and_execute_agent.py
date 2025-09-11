from jet.adapters.langchain.chat_ollama import ChatOllama, Ollama
from jet.logger import logger
from langchain.chains import LLMMathChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
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
# Plan-and-execute

Plan-and-execute agents accomplish an objective by first planning what to do, then executing the sub tasks. This idea is largely inspired by [BabyAGI](https://github.com/yoheinakajima/babyagi) and then the ["Plan-and-Solve" paper](https://arxiv.org/abs/2305.04091).

The planning is almost always done by an LLM.

The execution is usually done by a separate agent (equipped with tools).

## Imports
"""
logger.info("# Plan-and-execute")


"""
## Tools
"""
logger.info("## Tools")

search = DuckDuckGoSearchAPIWrapper()
llm = ChatOllama(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]

"""
## Planner, Executor, and Agent
"""
logger.info("## Planner, Executor, and Agent")

model = ChatOllama(model="llama3.2")
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor)

"""
## Run example
"""
logger.info("## Run example")

agent.run(
    "Who is the current prime minister of the UK? What is their current age raised to the 0.43 power?"
)

logger.info("\n\n[DONE]", bright=True)
