from jet.logger import logger
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
# Tracing

<span data-heading-keywords="trace,tracing"></span>

A trace is essentially a series of steps that your application takes to go from input to output.
Traces contain individual steps called `runs`. These can be individual calls from a model, retriever,
tool, or sub-chains.
Tracing gives you observability inside your chains and agents, and is vital in diagnosing issues.

For a deeper dive, check out [this LangSmith conceptual guide](https://docs.langchain.com/langsmith/observability-quickstart).
"""
logger.info("# Tracing")

logger.info("\n\n[DONE]", bright=True)