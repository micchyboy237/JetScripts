from jet.logger import logger
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
from smolagents import (
CodeAgent,
InferenceClientModel,
ToolCallingAgent,
VisitWebpageTool,
WebSearchTool,
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



register()
SmolagentsInstrumentor().instrument(skip_dep_check=True)




# Then we run the agentic part!
model = InferenceClientModel(provider="nebius")

search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="search_agent",
    description="This is an agent that can do web search.",
    return_full_result=True,
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
    return_full_result=True,
)
run_result = manager_agent.run(
    "If the US keeps it 2024 growth rate, how many years would it take for the GDP to double?"
)
logger.debug("Here is the token usage for the manager agent", run_result.token_usage)
logger.debug("Here are the timing informations for the manager agent:", run_result.timing)

logger.info("\n\n[DONE]", bright=True)