from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.callbacks.tracers.comet import CometTracer
import comet_llm
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
# Comet Tracing

There are two ways to trace your LangChains executions with Comet:

1. Setting the `LANGCHAIN_COMET_TRACING` environment variable to "true". This is the recommended way.
2. Import the `CometTracer` manually and pass it explicitely.
"""
logger.info("# Comet Tracing")



os.environ["LANGCHAIN_COMET_TRACING"] = "true"

comet_llm.init()

os.environ["COMET_PROJECT_NAME"] = "comet-example-langchain-tracing"


llm = Ollama(temperature=0)
tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What is 2 raised to .123243 power?")  # this should be traced

if "LANGCHAIN_COMET_TRACING" in os.environ:
    del os.environ["LANGCHAIN_COMET_TRACING"]


tracer = CometTracer()

llm = Ollama(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What is 2 raised to .123243 power?", callbacks=[tracer]
)  # this should be traced

logger.info("\n\n[DONE]", bright=True)