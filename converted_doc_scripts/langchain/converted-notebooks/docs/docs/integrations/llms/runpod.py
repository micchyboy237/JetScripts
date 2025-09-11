from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_runpod import RunPod
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
# RunPod LLM

Get started with RunPod LLMs.

## Overview

This guide covers how to use the LangChain `RunPod` LLM class to interact with text generation models hosted on [RunPod Serverless](https://www.runpod.io/serverless-gpu).

## Setup

1. **Install the package:**
   ```bash
   pip install -qU langchain-runpod
   ```
2. **Deploy an LLM Endpoint:** Follow the setup steps in the [RunPod Provider Guide](/docs/integrations/providers/runpod#setup) to deploy a compatible text generation endpoint on RunPod Serverless and get its Endpoint ID.
3. **Set Environment Variables:** Make sure `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` are set.
"""
logger.info("# RunPod LLM")

# import getpass

if "RUNPOD_API_KEY" not in os.environ:
#     os.environ["RUNPOD_API_KEY"] = getpass.getpass("Enter your RunPod API Key: ")
if "RUNPOD_ENDPOINT_ID" not in os.environ:
    os.environ["RUNPOD_ENDPOINT_ID"] = input("Enter your RunPod Endpoint ID: ")

"""
## Instantiation

Initialize the `RunPod` class. You can pass model-specific parameters via `model_kwargs` and configure polling behavior.
"""
logger.info("## Instantiation")


llm = RunPod(
    model_kwargs={
        "max_new_tokens": 256,
        "temperature": 0.6,
        "top_k": 50,
    },
)

"""
## Invocation

Use the standard LangChain `.invoke()` and `.ainvoke()` methods to call the model. Streaming is also supported via `.stream()` and `.astream()` (simulated by polling the RunPod `/stream` endpoint).
"""
logger.info("## Invocation")

prompt = "Write a tagline for an ice cream shop on the moon."

try:
    response = llm.invoke(prompt)
    logger.debug("--- Sync Invoke Response ---")
    logger.debug(response)
except Exception as e:
    logger.debug(
        f"Error invoking LLM: {e}. Ensure endpoint ID/API key are correct and endpoint is active/compatible."
    )

logger.debug("\n--- Sync Stream Response ---")
try:
    for chunk in llm.stream(prompt):
        logger.debug(chunk, end="", flush=True)
    logger.debug()  # Newline
except Exception as e:
    logger.debug(
        f"\nError streaming LLM: {e}. Ensure endpoint handler supports streaming output format."
    )

"""
### Async Usage
"""
logger.info("### Async Usage")

try:
    async_response = await llm.ainvoke(prompt)
    logger.success(format_json(async_response))
    logger.debug("--- Async Invoke Response ---")
    logger.debug(async_response)
except Exception as e:
    logger.debug(f"Error invoking LLM asynchronously: {e}.")

logger.debug("\n--- Async Stream Response ---")
try:
    for chunk in llm.stream(prompt):
        logger.debug(chunk, end="", flush=True)
    logger.debug()  # Newline
except Exception as e:
    logger.debug(
        f"\nError streaming LLM asynchronously: {e}. Ensure endpoint handler supports streaming output format."
    )

"""
## Chaining

The LLM integrates seamlessly with LangChain Expression Language (LCEL) chains.
"""
logger.info("## Chaining")


prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()

chain = prompt_template | llm | parser

try:
    chain_response = chain.invoke({"topic": "bears"})
    logger.debug("--- Chain Response ---")
    logger.debug(chain_response)
except Exception as e:
    logger.debug(f"Error running chain: {e}")

try:
    async_chain_response = await chain.ainvoke({"topic": "robots"})
    logger.success(format_json(async_chain_response))
    logger.debug("--- Async Chain Response ---")
    logger.debug(async_chain_response)
except Exception as e:
    logger.debug(f"Error running async chain: {e}")

"""
## Endpoint Considerations

- **Input:** The endpoint handler should expect the prompt string within `{"input": {"prompt": "...", ...}}`.
- **Output:** The handler should return the generated text within the `"output"` key of the final status response (e.g., `{"output": "Generated text..."}` or `{"output": {"text": "..."}}`).
- **Streaming:** For simulated streaming via the `/stream` endpoint, the handler must populate the `"stream"` key in the status response with a list of chunk dictionaries, like `[{"output": "token1"}, {"output": "token2"}]`.

## API reference

For detailed documentation of the `RunPod` LLM class, parameters, and methods, refer to the source code or the generated API reference (if available).

Link to source code: [https://github.com/runpod/langchain-runpod/blob/main/langchain_runpod/llms.py](https://github.com/runpod/langchain-runpod/blob/main/langchain_runpod/llms.py)
"""
logger.info("## Endpoint Considerations")

logger.info("\n\n[DONE]", bright=True)