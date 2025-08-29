from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.rungpt import RunGptLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/rungpt.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# RunGPT
RunGPT is an open-source cloud-native large-scale multimodal models (LMMs) serving framework. It is designed to simplify the deployment and management of large language models, on a distributed cluster of GPUs. RunGPT aim to make it a one-stop solution for a centralized and accessible place to gather techniques for optimizing large-scale multimodal models and make them easy to use for everyone. In RunGPT, we have supported a number of LLMs such as LLaMA, Pythia, StableLM, Vicuna, MOSS, and Large Multi-modal Model(LMMs) like MiniGPT-4 and OpenFlamingo additionally.

# Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# RunGPT")

# %pip install llama-index-llms-rungpt

# !pip install llama-index

"""
You need to install rungpt package in your python environment with `pip install`
"""
logger.info("You need to install rungpt package in your python environment with `pip install`")

# !pip install rungpt

"""
After installing successfully, models supported by RunGPT can be deployed with an one-line command. This option will download target language model from open source platform and deploy it as a service at a localhost port, which can be accessed by http or grpc requests. I suppose you not run this command in jupyter book, but in command line instead.
"""
logger.info("After installing successfully, models supported by RunGPT can be deployed with an one-line command. This option will download target language model from open source platform and deploy it as a service at a localhost port, which can be accessed by http or grpc requests. I suppose you not run this command in jupyter book, but in command line instead.")

# !rungpt serve decapoda-research/llama-7b-hf --precision fp16 --device_map balanced

"""
## Basic Usage
#### Call `complete` with a prompt
"""
logger.info("## Basic Usage")


llm = RunGptLLM()
promot = "What public transportation might be available in a city?"
response = llm.complete(promot)

logger.debug(response)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role=MessageRole.USER,
        content="Now, I want you to do some math for me.",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT, content="Sure, I would like to help you."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="How many points determine a straight line?",
    ),
]
llm = RunGptLLM()
response = llm.chat(messages=messages, temperature=0.8, max_tokens=15)

logger.debug(response)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

promot = "What public transportation might be available in a city?"
response = RunGptLLM().stream_complete(promot)
for item in response:
    logger.debug(item.text)

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role=MessageRole.USER,
        content="Now, I want you to do some math for me.",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT, content="Sure, I would like to help you."
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="How many points determine a straight line?",
    ),
]
response = RunGptLLM().stream_chat(messages=messages)

for item in response:
    logger.debug(item.message)

logger.info("\n\n[DONE]", bright=True)