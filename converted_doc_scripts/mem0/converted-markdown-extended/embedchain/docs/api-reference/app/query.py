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
title: '❓ query'
---

`.query()` method empowers developers to ask questions and receive relevant answers through a user-friendly query API. Function signature is given below:

### Parameters

<ParamField path="input_query" type="str">
    Question to ask
</ParamField>
<ParamField path="config" type="BaseLlmConfig" optional>
    Configure different llm settings such as prompt, temprature, number_documents etc.
</ParamField>
<ParamField path="dry_run" type="bool" optional>
    The purpose is to test the prompt structure without actually running LLM inference. Defaults to `False`
</ParamField>
<ParamField path="where" type="dict" optional>
    A dictionary of key-value pairs to filter the chunks from the vector database. Defaults to `None`
</ParamField>
<ParamField path="citations" type="bool" optional>
    Return citations along with the LLM answer. Defaults to `False`
</ParamField>

### Returns

<ResponseField name="answer" type="str | tuple">
  If `citations=False`, return a stringified answer to the question asked. <br />
  If `citations=True`, returns a tuple with answer and citations respectively.
</ResponseField>

## Usage

### With citations

If you want to get the answer to question and return both answer and citations, use the following code snippet:
"""
logger.info("### Parameters")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

answer, sources = app.query("What is the net worth of Elon?", citations=True)
logger.debug(answer)

logger.debug(sources)

"""
<Note>
When `citations=True`, note that the returned `sources` are a list of tuples where each tuple has two elements (in the following order):
1. source chunk
2. dictionary with metadata about the source chunk
    - `url`: url of the source
    - `doc_id`: document id (used for book keeping purposes)
    - `score`: score of the source chunk with respect to the question
    - other metadata you might have added at the time of adding the source
</Note>

### Without citations

If you just want to return answers and don't want to return citations, you can use the following example:
"""
logger.info("### Without citations")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

answer = app.query("What is the net worth of Elon?")
logger.debug(answer)

logger.info("\n\n[DONE]", bright=True)