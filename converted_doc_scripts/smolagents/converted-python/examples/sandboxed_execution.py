from jet.logger import logger
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool
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



model = InferenceClientModel()

# Docker executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="docker") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
logger.debug("Docker executor result:", output)

# E2B executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="e2b") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
logger.debug("E2B executor result:", output)

# Modal executor example
with CodeAgent(tools=[WebSearchTool()], model=model, executor_type="modal") as agent:
    output = agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
logger.debug("Modal executor result:", output)

# WebAssembly executor example
with CodeAgent(tools=[], model=model, executor_type="wasm") as agent:
    output = agent.run("Calculate the square root of 125.")
logger.debug("Wasm executor result:", output)
# TODO: Support tools
# with CodeAgent(tools=[VisitWebpageTool()], model=model, executor_type="wasm") as agent:
#     output = agent.run("What is the content of the Wikipedia page at https://en.wikipedia.org/wiki/Intelligent_agent?")

logger.info("\n\n[DONE]", bright=True)