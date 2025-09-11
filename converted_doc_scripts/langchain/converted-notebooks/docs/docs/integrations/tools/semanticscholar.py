from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
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
# Semantic Scholar API Tool

This notebook demos how to use the semantic scholar tool with an agent.
"""
logger.info("# Semantic Scholar API Tool")

# %pip install --upgrade --quiet  semanticscholar


instructions = """You are an expert researcher."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOllama(model="llama3.2")


tools = [SemanticScholarQueryRun()]

agent = create_ollama_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke(
    {
        "input": "What are some biases in the large language models? How have people tried to mitigate them? "
        "show me a list of papers and techniques. Based on your findings write new research questions "
        "to work on. Break down the task into subtasks for search. Use the search tool"
    }
)

logger.info("\n\n[DONE]", bright=True)