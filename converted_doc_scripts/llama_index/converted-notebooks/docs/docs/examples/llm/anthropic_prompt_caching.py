from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import (
ChatMessage,
TextBlock,
CachePoint,
CacheControl,
)
from llama_index.llms.anthropic import Anthropic
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/anthropic_prompt_caching.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Anthropic Prompt Caching

In this Notebook, we will demonstrate the usage of [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) with LlamaIndex abstractions.

Prompt Caching is enabled by marking `cache_control` in the messages request.


## How Prompt Caching works

When you send a request with Prompt Caching enabled:

1. The system checks if the prompt prefix is already cached from a recent query.
2. If found, it uses the cached version, reducing processing time and costs.
3. Otherwise, it processes the full prompt and caches the prefix for future use.


**Note:**

A. Prompt caching works with `Claude 4 Opus`, `Claude 4 Sonnet`, `Claude 3.7 Sonnet`, `Claude 3.5 Sonnet`, `Claude 3.5 Haiku`, `Claude 3 Haiku` and `Claude 3 Opus` models.

B. The minimum cacheable prompt length is:

    1. 2048 tokens for Claude 3.5 Haiku and Claude 3 Haiku
    2. 1024 for all the other models.

C. Shorter prompts cannot be cached, even if marked with `cache_control`.

### Setup API Keys
"""
logger.info("# Anthropic Prompt Caching")


os.environ[
#     "ANTHROPIC_API_KEY"
] = "sk-ant-..."  # replace with your Anthropic API key

"""
### Setup LLM
"""
logger.info("### Setup LLM")


llm = Anthropic(model="claude-3-5-sonnet-20240620")

"""
### Download Data

In this demonstration, we will use the text from the `Paul Graham Essay`. We will cache the text and run some queries based on it.
"""
logger.info("### Download Data")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'

"""
### Load Data
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader(
    input_files=["./paul_graham_essay.txt"],
).load_data()

document_text = documents[0].text

"""
### Prompt Caching

To enable prompt caching, you can just use the `CachePoint` block within LlamaIndex: everything previous to that block will be cached.

We can verify if the text is cached by checking the following parameters:

`cache_creation_input_tokens:` Number of tokens written to the cache when creating a new entry.

`cache_read_input_tokens:` Number of tokens retrieved from the cache for this request.

`input_tokens:` Number of input tokens which were not read from or used to create a cache.
"""
logger.info("### Prompt Caching")


messages = [
    ChatMessage(role="system", content="You are helpful AI Assitant."),
    ChatMessage(
        role="user",
        content=[
            TextBlock(
                text=f"{document_text}",
                type="text",
            ),
            TextBlock(
                text="\n\nWhy did Paul Graham start YC?",
                type="text",
            ),
            CachePoint(cache_control=CacheControl(type="ephemeral")),
        ],
    ),
]

resp = llm.chat(messages)

"""
Let's examine the raw response.
"""
logger.info("Let's examine the raw response.")

resp.raw

"""
As you can see, since I've ran this a few different times, `cache_creation_input_tokens` and `cache_read_input_tokens` are both higher than zero, indicating that the text was cached properly.

Now, let’s run another query on the same document. It should retrieve the document text from the cache, which will be reflected in `cache_read_input_tokens`.
"""
logger.info("As you can see, since I've ran this a few different times, `cache_creation_input_tokens` and `cache_read_input_tokens` are both higher than zero, indicating that the text was cached properly.")

messages = [
    ChatMessage(role="system", content="You are helpful AI Assitant."),
    ChatMessage(
        role="user",
        content=[
            TextBlock(
                text=f"{document_text}",
                type="text",
            ),
            TextBlock(
                text="\n\nWhat did Paul Graham do growing up?",
                type="text",
            ),
            CachePoint(cache_control=CacheControl(type="ephemeral")),
        ],
    ),
]

resp = llm.chat(messages)

resp.raw

"""
As you can see, the response was generated using cached text, as indicated by `cache_read_input_tokens`.

With Anthropic, the default cache lasts 5 minutes. You can also have longer lasting caches, for instance 1 hour, you just have to specify that under the `ttl` argument in `CachControl`.
"""
logger.info("As you can see, the response was generated using cached text, as indicated by `cache_read_input_tokens`.")

messages = [
    ChatMessage(role="system", content="You are helpful AI Assitant."),
    ChatMessage(
        role="user",
        content=[
            TextBlock(
                text=f"{document_text}",
                type="text",
            ),
            TextBlock(
                text="\n\nWhat did Paul Graham do growing up?",
                type="text",
            ),
            CachePoint(
                cache_control=CacheControl(type="ephemeral", ttl="1h"),
            ),
        ],
    ),
]

resp = llm.chat(messages)

logger.info("\n\n[DONE]", bright=True)