from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.llms.fake import FakeListLLM
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
# Fake LLM
LangChain provides a fake LLM class that can be used for testing. This allows you to mock out calls to the LLM and simulate what would happen if the LLM responded in a certain way.

In this notebook we go over how to use this.

We start this with using the FakeLLM in an agent.
"""
logger.info("# Fake LLM")



tools = load_tools(["python_repl"])

responses = ["Action: Python REPL\nAction Input: logger.debug(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.invoke("whats 2 + 2")

logger.info("\n\n[DONE]", bright=True)