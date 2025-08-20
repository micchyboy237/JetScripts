from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.konko import Konko
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/Konko.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Konko

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

>[Konko](https://www.konko.ai/) API is a fully managed Web API designed to help application developers:

Konko API is a fully managed API designed to help application developers:

1. Select the right LLM(s) for their application
2. Prototype with various open-source and proprietary LLMs
3. Access Fine Tuning for open-source LLMs to get industry-leading performance at a fraction of the cost
4. Setup low-cost production APIs according to security, privacy, throughput, latency SLAs without infrastructure set-up or administration using Konko AI's SOC 2 compliant, multi-cloud infrastructure

### Steps to Access Models
1. **Explore Available Models:** Start by browsing through the [available models](https://docs.konko.ai/docs/list-of-models) on Konko. Each model caters to different use cases and capabilities.

2. **Identify Suitable Endpoints:** Determine which [endpoint](https://docs.konko.ai/docs/list-of-models#list-of-available-models) (ChatCompletion or Completion) supports your selected model.

3. **Selecting a Model:** [Choose a model](https://docs.konko.ai/docs/list-of-models#list-of-available-models) based on its metadata and how well it fits your use case.

4. **Prompting Guidelines:** Once a model is selected, refer to the [prompting guidelines](https://docs.konko.ai/docs/prompting) to effectively communicate with it.

5. **Using the API:** Finally, use the appropriate Konko [API endpoint](https://docs.konko.ai/docs/quickstart-for-completion-and-chat-completion-endpoint) to call the model and receive responses.

To run this notebook, you'll need Konko API key. You can create one by signing up on [Konko](https://www.konko.ai/).

This example goes over how to use LlamaIndex to interact with `Konko` ChatCompletion [models](https://docs.konko.ai/docs/list-of-models#konko-hosted-models-for-chatcompletion) and Completion [models](https://docs.konko.ai/docs/list-of-models#konko-hosted-models-for-completion)
"""
logger.info("# Konko")

# %pip install llama-index-llms-konko

# !pip install llama-index

"""
## Call `chat` with ChatMessage List
You need to set env var `KONKO_API_KEY`
"""
logger.info("## Call `chat` with ChatMessage List")


os.environ["KONKO_API_KEY"] = "<your-api-key>"


llm = Konko(model="meta-llama/llama-2-13b-chat")
messages = ChatMessage(role="user", content="Explain Big Bang Theory briefly")

resp = llm.chat([messages])
logger.debug(resp)

"""
## Call `chat` with MLX Models
# You need to either set env var `OPENAI_API_KEY`
"""
logger.info("## Call `chat` with MLX Models")


# os.environ["OPENAI_API_KEY"] = "<your-api-key>"

llm = Konko(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

message = ChatMessage(role="user", content="Explain Big Bang Theory briefly")
resp = llm.chat([message])
logger.debug(resp)

"""
### Streaming
"""
logger.info("### Streaming")

message = ChatMessage(role="user", content="Tell me a story in 250 words")
resp = llm.stream_chat([message], max_tokens=1000)
for r in resp:
    logger.debug(r.delta, end="")

"""
## Call `complete` with Prompt
"""
logger.info("## Call `complete` with Prompt")

llm = Konko(model="numbersstation/nsql-llama-2-7b", max_tokens=100)
text = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
    highest number,
    lowest number,
    average number
)

CREATE TABLE singer (
    singer_id number,
    name text,
    country text,
    song_name text,
    song_release_year text,
    age number,
    is_male others
)

CREATE TABLE concert (
    concert_id number,
    concert_name text,
    theme text,
    stadium_id text,
    year text
)

CREATE TABLE singer_in_concert (
    concert_id number,
    singer_id text
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- What is the maximum capacity of stadiums ?

SELECT"""
response = llm.complete(text)
logger.debug(response)

llm = Konko(model="phind/phind-codellama-34b-v2", max_tokens=100)
text = """### System Prompt
You are an intelligent programming assistant.

Implement a linked list in C++

..."""

resp = llm.stream_complete(text, max_tokens=1000)
for r in resp:
    logger.debug(r.delta, end="")

"""
## Model Configuration
"""
logger.info("## Model Configuration")

llm = Konko(model="meta-llama/llama-2-13b-chat")

resp = llm.stream_complete(
    "Show me the c++ code to send requests to HTTP Server", max_tokens=1000
)
for r in resp:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)