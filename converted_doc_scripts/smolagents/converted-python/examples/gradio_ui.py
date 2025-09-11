from jet.logger import logger
from smolagents import CodeAgent, GradioUI, InferenceClientModel, WebSearchTool
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



agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct", provider="fireworks-ai"),
    verbosity_level=1,
    planning_interval=3,
    name="example_agent",
    description="This is an example agent.",
    step_callbacks=[],
    stream_outputs=True,
    # use_structured_outputs_internally=True,
)

GradioUI(agent, file_upload_folder="./data").launch()

logger.info("\n\n[DONE]", bright=True)