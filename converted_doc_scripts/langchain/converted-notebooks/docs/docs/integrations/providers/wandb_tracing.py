from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.callbacks import wandb_tracing_enabled
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
# Weights & Biases tracing

There are two recommended ways to trace your LangChains:

1. Setting the `LANGCHAIN_WANDB_TRACING` environment variable to "true".
1. Using a context manager with tracing_enabled() to trace a particular block of code.

**Note** if the environment variable is set, all code will be traced, regardless of whether or not it's within the context manager.
"""
logger.info("# Weights & Biases tracing")



os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

os.environ["WANDB_PROJECT"] = "langchain-tracing"


llm = Ollama(temperature=0)
tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What is 2 raised to .123243 power?")  # this should be traced

if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

with wandb_tracing_enabled():
    agent.run("What is 5 raised to .123243 power?")  # this should be traced

agent.run("What is 2 raised to .123243 power?")  # this should not be traced

logger.info("\n\n[DONE]", bright=True)