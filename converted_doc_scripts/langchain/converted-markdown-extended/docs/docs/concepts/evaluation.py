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
# Evaluation
<span data-heading-keywords="evaluation,evaluate"></span>

Evaluation is the process of assessing the performance and effectiveness of your LLM-powered applications.
It involves testing the model's responses against a set of predefined criteria or benchmarks to ensure it meets the desired quality standards and fulfills the intended purpose.
This process is vital for building reliable applications.

![](/img/langsmith_evaluate.png)

[LangSmith](https://docs.smith.langchain.com/) helps with this process in a few ways:

- It makes it easier to create and curate datasets via its tracing and annotation features
- It provides an evaluation framework that helps you define metrics and run your app against your dataset
- It allows you to track results over time and automatically run your evaluators on a schedule or as part of CI/Code

To learn more, check out [this LangSmith guide](https://docs.smith.langchain.com/concepts/evaluation).
"""
logger.info("# Evaluation")

logger.info("\n\n[DONE]", bright=True)