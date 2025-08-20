from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/groq.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Groq

Welcome to Groq! 🚀 At Groq, we've developed the world's first Language Processing Unit™, or LPU. The Groq LPU has a deterministic, single core streaming architecture that sets the standard for GenAI inference speed with predictable and repeatable performance for any given workload.

Beyond the architecture, our software is designed to empower developers like you with the tools you need to create innovative, powerful AI applications. With Groq as your engine, you can:

* Achieve uncompromised low latency and performance for real-time AI and HPC inferences 🔥
* Know the exact performance and compute time for any given workload 🔮
* Take advantage of our cutting-edge technology to stay ahead of the competition 💪

Want more Groq? Check out our [website](https://groq.com) for more resources and join our [Discord community](https://discord.gg/JvNsBDKeCG) to connect with our developers!

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Groq")

# % pip install llama-index-llms-groq

# !pip install llama-index


"""
Create an API key at the [Groq console](https://console.groq.com/keys), then set it to the environment variable `GROQ_API_KEY`.

```bash
export GROQ_API_KEY=<your api key>
```

Alternatively, you can pass your API key to the LLM when you init it:
"""
logger.info("Create an API key at the [Groq console](https://console.groq.com/keys), then set it to the environment variable `GROQ_API_KEY`.")

llm = Groq(model="llama3-70b-8192", api_key="your_api_key")

"""
A list of available LLM models can be found [here](https://console.groq.com/docs/models).
"""
logger.info("A list of available LLM models can be found [here](https://console.groq.com/docs/models).")

response = llm.complete("Explain the importance of low latency LLMs")

logger.debug(response)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Explain the importance of low latency LLMs")

for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)