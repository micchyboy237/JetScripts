from jet.adapters.langchain.chat_ollama.chat_models import ChatOllama
from jet.adapters.langchain.chat_ollama.llms import OllamaLLM
from jet.logger import logger
from langchain_community.embeddings import OllamaEmbeddings
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
# Ollama

>[Ollama](https://ollama.com/) allows you to run open-source large language models,
> such as [gpt-oss](https://ollama.com/library/gpt-oss), locally.
>
>`Ollama` bundles model weights, configuration, and data into a single package, defined by a Modelfile.
>It optimizes setup and configuration details, including GPU usage.
>For a complete list of supported models and model variants, see the [Ollama model library](https://ollama.ai/library).

See [this guide](/docs/how_to/local_llms#ollama) for more details
on how to use `ollama` with LangChain.

## Installation and Setup
### Ollama installation
Follow [these instructions](https://github.com/ollama/ollama?tab=readme-ov-file#ollama)
to set up and run a local Ollama instance.

Ollama will start as a background service automatically, if this is disabled, run:
"""
logger.info("# Ollama")

ollama serve

"""
After starting ollama, run `ollama pull <name-of-model>` to download a model from the [Ollama model library](https://ollama.ai/library):
"""
logger.info("After starting ollama, run `ollama pull <name-of-model>` to download a model from the [Ollama model library](https://ollama.ai/library):")

ollama pull gpt-oss:20b

"""
- This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model.
- To view all pulled (downloaded) models, use `ollama list`

We're now ready to install the `langchain-ollama` partner package and run a model.

### Ollama LangChain partner package install
Install the integration package with:
"""
logger.info("### Ollama LangChain partner package install")

pip install langchain-ollama

"""
## LLM
"""
logger.info("## LLM")


"""
See the notebook example [here](/docs/integrations/llms/ollama).

## Chat Models

### Chat Ollama
"""
logger.info("## Chat Models")


"""
See the notebook example [here](/docs/integrations/chat/ollama).

### Ollama tool calling
[Ollama tool calling](https://ollama.com/blog/tool-support) uses the
Ollama compatible web server specification, and can be used with
the default `BaseChatModel.bind_tools()` methods
as described [here](/docs/how_to/tool_calling/).
Make sure to select an ollama model that supports [tool calling](https://ollama.com/search?&c=tools).

## Embedding models
"""
logger.info("### Ollama tool calling")


"""
See the notebook example [here](/docs/integrations/text_embedding/ollama).
"""
logger.info("See the notebook example [here](/docs/integrations/text_embedding/ollama).")

logger.info("\n\n[DONE]", bright=True)