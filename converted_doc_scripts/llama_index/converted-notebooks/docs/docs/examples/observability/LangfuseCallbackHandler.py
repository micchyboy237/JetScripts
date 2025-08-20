from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import global_handler, set_global_handler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/LangfuseCallbackHandler.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Langfuse Callback Handler

⚠️ This integration is deprecated. We recommend using the new instrumentation-based integration with Langfuse as described [here](https://langfuse.com/docs/integrations/llama-index/get-started).

This cookbook shows you how to use the Langfuse callback handler to monitor LlamaIndex applications.

## What is Langfuse?

[Langfuse](https://langfuse.com/docs) is an open source LLM engineering platform to help teams collaboratively debug, analyze and iterate on their LLM Applications. Langfuse offers a simple integration for automatic capture of [traces](https://langfuse.com/docs/tracing) and metrics generated in LlamaIndex applications. 

## How does it work?

The `LangfuseCallbackHandler` is integrated with Langfuse and empowers you to seamlessly track and monitor performance, traces, and metrics of your LlamaIndex application. Detailed traces of the LlamaIndex context augmentation and the LLM querying processes are captured and can be inspected directly in the Langfuse UI.

![langfuse-tracing](https://static.langfuse.com/llamaindex-langfuse-docs.gif)

## Setup

### Install packages
"""
logger.info("# Langfuse Callback Handler")

# %pip install llama-index llama-index-callbacks-langfuse

"""
### Configure environment

If you haven't done yet, [sign up on Langfuse](https://cloud.langfuse.com/auth/sign-up) and obtain your API keys from the project settings.
"""
logger.info("### Configure environment")


os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # 🇪🇺 EU region

# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
### Register the Langfuse callback handler

#### Option 1: Set global LlamaIndex handler
"""
logger.info("### Register the Langfuse callback handler")


set_global_handler("langfuse")
langfuse_callback_handler = global_handler

"""
#### Option 2: Use Langfuse callback directly
"""
logger.info("#### Option 2: Use Langfuse callback directly")


langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

"""
### Flush events to Langfuse

The Langfuse SDKs queue and batches events in the background to reduce the number of network requests and improve overall performance. Before exiting your application, make sure all queued events have been flushed to Langfuse servers.
"""
logger.info("### Flush events to Langfuse")

langfuse_callback_handler.flush()

"""
Done!✨ Traces and metrics from your LlamaIndex application are now automatically tracked in Langfuse. If you construct a new index or query an LLM with your documents in context, your traces and metrics are immediately visible in the Langfuse UI. Next, let's take a look at how traces will look in Langfuse.

## Example

Fetch and save example data.
"""
logger.info("## Example")

# !mkdir -p 'data/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham_essay.txt'

"""
Run an example index construction, query, and chat.
"""
logger.info("Run an example index construction, query, and chat.")


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
query_response = query_engine.query("What did the author do growing up?")
logger.debug(query_response)

chat_engine = index.as_chat_engine()
chat_response = chat_engine.chat("What did the author do growing up?")
logger.debug(chat_response)

langfuse_callback_handler.flush()

"""
Done!✨ You will now see traces of your index and query in your Langfuse project.

Example traces (public links):
1. [Query](https://cloud.langfuse.com/project/cltipxbkn0000cdd7sbfbpovm/traces/f2e7f721-0940-4139-9b3a-e5cc9b0cb2d3)
2. [Query (chat)](https://cloud.langfuse.com/project/cltipxbkn0000cdd7sbfbpovm/traces/89c62a4d-e992-4923-a6b7-e2f27ae4cff3)
3. [Session](https://cloud.langfuse.com/project/cltipxbkn0000cdd7sbfbpovm/sessions/notebook-session-2)

## 📚 More details

Check out the full [Langfuse documentation](https://langfuse.com/docs) for more details on Langfuse's tracing and analytics capabilities and how to make most of this integration.

## Feedback

If you have any feedback or requests, please create a GitHub [Issue](https://github.com/orgs/langfuse/discussions) or share your idea with the community on [Discord](https://discord.langfuse.com/).
"""
logger.info("## 📚 More details")

logger.info("\n\n[DONE]", bright=True)