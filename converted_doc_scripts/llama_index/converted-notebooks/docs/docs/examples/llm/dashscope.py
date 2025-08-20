from jet.logger import CustomLogger
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/dashscope.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DashScope LLMS

In this notebook, we show how to use the DashScope LLM models in LlamaIndex. Check out the [DashScope site](https://dashscope.aliyun.com/) or the [documents](https://help.aliyun.com/zh/dashscope/developer-reference/api-details).

If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the DashScope Python SDK.
"""
logger.info("# DashScope LLMS")

# !pip install llama-index-llms-dashscope

"""
## Basic Usage

You will need to login [DashScope](https://dashscope.aliyun.com/) an create a API. Once you have one, you can either pass it explicitly to the API, or use the `DASHSCOPE_API_KEY` environment variable.
"""
logger.info("## Basic Usage")

# %env DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY


os.environ["DASHSCOPE_API_KEY"] = "YOUR_DASHSCOPE_API_KEY"

"""
#### Initialize `DashScope` Object
"""
logger.info("#### Initialize `DashScope` Object")


dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")

resp = dashscope_llm.complete("How to make cake?")
logger.debug(resp)

"""
#### Call `stream_complete`` with a prompt
"""
logger.info("#### Call `stream_complete`` with a prompt")

responses = dashscope_llm.stream_complete("How to make cake?")
for response in responses:
    logger.debug(response.delta, end="")

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content="How to make cake?"),
]
resp = dashscope_llm.chat(messages)
logger.debug(resp)

"""
#### Using `stream_chat`
"""
logger.info("#### Using `stream_chat`")

responses = dashscope_llm.stream_chat(messages)
for response in responses:
    logger.debug(response.delta, end="")

"""
#### Multiple rounds conversation.
"""
logger.info("#### Multiple rounds conversation.")

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a helpful assistant."
    ),
    ChatMessage(role=MessageRole.USER, content="How to make cake?"),
]
resp = dashscope_llm.chat(messages)
logger.debug(resp)

messages.append(
    ChatMessage(role=MessageRole.ASSISTANT, content=resp.message.content)
)

messages.append(
    ChatMessage(role=MessageRole.USER, content="How to make it without sugar")
)
resp = dashscope_llm.chat(messages)
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)