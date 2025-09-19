from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import os
import shutil
import tiktoken


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/TokenCountingHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Token Counting Handler

This notebook walks through how to use the TokenCountingHandler and how it can be used to track your prompt, completion, and embedding token usage over time.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Token Counting Handler")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Setup

Here, we setup the callback and the serivce context. We set global settings so that we don't have to worry about passing it into indexes and queries.
"""
logger.info("## Setup")


# os.environ["OPENAI_API_KEY"] = "sk-..."



token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

Settings.llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.2)
Settings.callback_manager = CallbackManager([token_counter])

"""
## Token Counting

The token counter will track embedding, prompt, and completion token usage. The token counts are __cummulative__ and are only reset when you choose to do so, with `token_counter.reset_counts()`.

### Embedding Token Usage

Now that the settings is setup, let's track our embedding token usage.

## Download Data
"""
logger.info("## Token Counting")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


index = VectorStoreIndex.from_documents(documents)

logger.debug(token_counter.total_embedding_token_count)

"""
That looks right! Before we go any further, lets reset the counts
"""
logger.info("That looks right! Before we go any further, lets reset the counts")

token_counter.reset_counts()

"""
### LLM + Embedding Token Usage

Next, let's test a query and see what the counts look like.
"""
logger.info("### LLM + Embedding Token Usage")

query_engine = index.as_query_engine(similarity_top_k=4)
response = query_engine.query("What did the author do growing up?")

logger.debug(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

"""
### Token Counting + Streaming!

The token counting handler also handles token counting during streaming.

Here, token counting will only happen once the stream is completed.
"""
logger.info("### Token Counting + Streaming!")

token_counter.reset_counts()

query_engine = index.as_query_engine(similarity_top_k=4, streaming=True)
response = query_engine.query("What happened at Interleaf?")

for token in response.response_gen:
    continue

logger.debug(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
)

"""
## Advanced Usage

The token counter tracks each token usage event in an object called a `TokenCountingEvent`. This object has the following attributes:

- prompt -> The prompt string sent to the LLM or Embedding model
- prompt_token_count -> The token count of the LLM prompt
- completion -> The string completion received from the LLM (not used for embeddings)
- completion_token_count -> The token count of the LLM completion (not used for embeddings)
- total_token_count -> The total prompt + completion tokens for the event
- event_id -> A string ID for the event, which aligns with other callback handlers

These events are tracked on the token counter in two lists:

- llm_token_counts
- embedding_token_counts

Let's explore what these look like!
"""
logger.info("## Advanced Usage")

logger.debug("Num LLM token count events: ", len(token_counter.llm_token_counts))
logger.debug(
    "Num Embedding token count events: ",
    len(token_counter.embedding_token_counts),
)

"""
This makes sense! The previous query embedded the query text, and then made 2 LLM calls (since the top k was 4, and the default chunk size is 1024, two separate calls need to be made so the LLM can read all the retrieved text).

Next, let's quickly see what these events look like for a single event.
"""
logger.info("This makes sense! The previous query embedded the query text, and then made 2 LLM calls (since the top k was 4, and the default chunk size is 1024, two separate calls need to be made so the LLM can read all the retrieved text).")

logger.debug("prompt: ", token_counter.llm_token_counts[0].prompt[:100], "...\n")
logger.debug(
    "prompt token count: ",
    token_counter.llm_token_counts[0].prompt_token_count,
    "\n",
)

logger.debug(
    "completion: ", token_counter.llm_token_counts[0].completion[:100], "...\n"
)
logger.debug(
    "completion token count: ",
    token_counter.llm_token_counts[0].completion_token_count,
    "\n",
)

logger.debug("total token count", token_counter.llm_token_counts[0].total_token_count)

logger.info("\n\n[DONE]", bright=True)