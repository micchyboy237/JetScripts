from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '📝 evaluate'
---

`evaluate()` method is used to evaluate the performance of a RAG app. You can find the signature below:

### Parameters

<ParamField path="question" type="Union[str, list[str]]">
    A question or a list of questions to evaluate your app on.
</ParamField>
<ParamField path="metrics" type="Optional[list[Union[BaseMetric, str]]]" optional>
    The metrics to evaluate your app on. Defaults to all metrics: `["context_relevancy", "answer_relevancy", "groundedness"]`
</ParamField>
<ParamField path="num_workers" type="int" optional>
    Specify the number of threads to use for parallel processing.
</ParamField>

### Returns

<ResponseField name="metrics" type="dict">
    Returns the metrics you have chosen to evaluate your app on as a dictionary.
</ResponseField>

## Usage
"""
logger.info("### Parameters")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

app.evaluate("what is the net worth of Elon Musk?")

logger.info("\n\n[DONE]", bright=True)