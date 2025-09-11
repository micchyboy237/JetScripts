from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_runpod import ChatRunPod
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
# RunPod Chat Model

Get started with RunPod chat models.

## Overview

This guide covers how to use the LangChain `ChatRunPod` class to interact with chat models hosted on [RunPod Serverless](https://www.runpod.io/serverless-gpu).

## Setup

1.  **Install the package:**
    ```bash
    pip install -qU langchain-runpod
    ```
2.  **Deploy a Chat Model Endpoint:** Follow the setup steps in the [RunPod Provider Guide](/docs/integrations/providers/runpod#setup) to deploy a compatible chat model endpoint on RunPod Serverless and get its Endpoint ID.
3.  **Set Environment Variables:** Make sure `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` (or a specific `RUNPOD_CHAT_ENDPOINT_ID`) are set.
"""
logger.info("# RunPod Chat Model")

# import getpass

if "RUNPOD_API_KEY" not in os.environ:
#     os.environ["RUNPOD_API_KEY"] = getpass.getpass("Enter your RunPod API Key: ")

if "RUNPOD_ENDPOINT_ID" not in os.environ:
    os.environ["RUNPOD_ENDPOINT_ID"] = input(
        "Enter your RunPod Endpoint ID (used if RUNPOD_CHAT_ENDPOINT_ID is not set): "
    )


chat_endpoint_id = os.environ.get(
    "RUNPOD_CHAT_ENDPOINT_ID", os.environ.get("RUNPOD_ENDPOINT_ID")
)
if not chat_endpoint_id:
    raise ValueError(
        "No RunPod Endpoint ID found. Please set RUNPOD_ENDPOINT_ID or RUNPOD_CHAT_ENDPOINT_ID."
    )

"""
## Instantiation

Initialize the `ChatRunPod` class. You can pass model-specific parameters via `model_kwargs` and configure polling behavior.
"""
logger.info("## Instantiation")


chat = ChatRunPod(
    runpod_endpoint_id=chat_endpoint_id,  # Specify the correct endpoint ID
    model_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    },
)

"""
## Invocation

Use the standard LangChain `.invoke()` and `.ainvoke()` methods to call the model. Streaming is also supported via `.stream()` and `.astream()` (simulated by polling the RunPod `/stream` endpoint).
"""
logger.info("## Invocation")


messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is the RunPod Serverless API flow?"),
]

try:
    response = chat.invoke(messages)
    logger.debug("--- Sync Invoke Response ---")
    logger.debug(response.content)
except Exception as e:
    logger.debug(
        f"Error invoking Chat Model: {e}. Ensure endpoint ID/API key are correct and endpoint is active/compatible."
    )

logger.debug("\n--- Sync Stream Response ---")
try:
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    logger.debug()  # Newline
except Exception as e:
    logger.debug(
        f"\nError streaming Chat Model: {e}. Ensure endpoint handler supports streaming output format."
    )


try:
    async_response = await chat.ainvoke(messages)
    logger.success(format_json(async_response))
    logger.debug("--- Async Invoke Response ---")
    logger.debug(async_response.content)
except Exception as e:
    logger.debug(f"Error invoking Chat Model asynchronously: {e}.")

logger.debug("\n--- Async Stream Response ---")
try:
    for chunk in chat.stream(messages):
        logger.debug(chunk.content, end="", flush=True)
    logger.debug()  # Newline
except Exception as e:
    logger.debug(
        f"\nError streaming Chat Model asynchronously: {e}. Ensure endpoint handler supports streaming output format.\n"
    )

"""
## Chaining

The chat model integrates seamlessly with LangChain Expression Language (LCEL) chains.
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)

parser = StrOutputParser()

chain = prompt | chat | parser

try:
    chain_response = chain.invoke(
        {"input": "Explain the concept of serverless computing in simple terms."}
    )
    logger.debug("--- Chain Response ---")
    logger.debug(chain_response)
except Exception as e:
    logger.debug(f"Error running chain: {e}")


try:
    async_chain_response = await chain.ainvoke(
            {"input": "What are the benefits of using RunPod for AI/ML workloads?"}
        )
    logger.success(format_json(async_chain_response))
    logger.debug("--- Async Chain Response ---")
    logger.debug(async_chain_response)
except Exception as e:
    logger.debug(f"Error running async chain: {e}")

"""
## Model Features (Endpoint Dependent)

The availability of advanced features depends **heavily** on the specific implementation of your RunPod endpoint handler. The `ChatRunPod` integration provides the basic framework, but the handler must support the underlying functionality.

| Feature                                                    | Integration Support | Endpoint Dependent? | Notes                                                                                                                                                                      |
| :--------------------------------------------------------- | :-----------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Tool calling](/docs/how_to/tool_calling)                | ❌                  | ✅                  | Requires handler to process tool definitions and return tool calls (e.g., Ollama format). Integration needs parsing logic.                                                 |
| [Structured output](/docs/how_to/structured_output)        | ❌                  | ✅                  | Requires handler support for forcing structured output (JSON mode, function calling). Integration needs parsing logic.                                                   |
| JSON mode                                                  | ❌                  | ✅                  | Requires handler to accept a `json_mode` parameter (or similar) and guarantee JSON output.                                                                               |
| [Image input](/docs/how_to/multimodal_inputs)            | ❌                  | ✅                  | Requires multimodal handler accepting image data (e.g., base64). Integration does not support multimodal messages.                                                       |
| Audio input                                                | ❌                  | ✅                  | Requires handler accepting audio data. Integration does not support audio messages.                                                                                        |
| Video input                                                | ❌                  | ✅                  | Requires handler accepting video data. Integration does not support video messages.                                                                                        |
| [Token-level streaming](/docs/how_to/chat_streaming)       | ✅ (Simulated)      | ✅                  | Polls `/stream`. Requires handler to populate `stream` list in status response with token chunks (e.g., `[{"output": "token"}]`). True low-latency streaming not built-in. |
| Native async                                               | ✅                  | ✅                  | Core `ainvoke`/`astream` implemented. Relies on endpoint handler performance.                                                                                              |
| [Token usage](/docs/how_to/chat_token_usage_tracking)    | ❌                  | ✅                  | Requires handler to return `prompt_tokens`, `completion_tokens` in the final response. Integration currently does not parse this.                                           |
| [Logprobs](/docs/how_to/logprobs)                          | ❌                  | ✅                  | Requires handler to return log probabilities. Integration currently does not parse this.                                                                                  |

**Key Takeaway:** Standard chat invocation and simulated streaming work if the endpoint follows basic RunPod API conventions. Advanced features require specific handler implementations and potentially extending or customizing this integration package.

## API reference

For detailed documentation of the `ChatRunPod` class, parameters, and methods, refer to the source code or the generated API reference (if available).

Link to source code: [https://github.com/runpod/langchain-runpod/blob/main/langchain_runpod/chat_models.py](https://github.com/runpod/langchain-runpod/blob/main/langchain_runpod/chat_models.py)
"""
logger.info("## Model Features (Endpoint Dependent)")

logger.info("\n\n[DONE]", bright=True)