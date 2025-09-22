from jet.logger import logger
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
---
sidebar:
  order: 3
---

# Installation and Setup

The LlamaIndex ecosystem is structured using a collection of namespaced python packages.

What this means for users is that `pip install llama-index` comes with a core starter bundle of packages, and additional integrations can be installed as needed.

A complete list of packages and available integrations is available on [LlamaHub](https://llamahub.ai/).

## Quickstart Installation from Pip

To get started quickly, you can install with:
"""
logger.info("# Installation and Setup")

pip install llama-index

"""
This is a starter bundle of packages, containing

- `llama-index-core`
- `llama-index-llms-ollama`
- `llama-index-embeddings-huggingface`
- `llama-index-readers-file`

**NOTE:** LlamaIndex may download and store local files for various packages (NLTK, HuggingFace, ...). Use the environment variable "LLAMA_INDEX_CACHE_DIR" to control where these files are saved.

### Important: Ollama Environment Setup

# By default, we use the Ollama `gpt-3.5-turbo` model for text generation and `text-embedding-ada-002` for retrieval and embeddings. In order to use this, you must have an OPENAI_API_KEY set up as an environment variable.
You can obtain an API key by logging into your Ollama account and [creating a new API key](https://platform.ollama.com/account/api-keys).

<Aside type="tip">
You can also [use one of many other available LLMs](/python/framework/module_guides/models/llms/usage_custom). You may need additional environment keys + tokens setup depending on the LLM provider.
</Aside>

[Check out our Ollama Starter Example](/python/framework/getting_started/starter_example)

## Custom Installation from Pip

If you aren't using Ollama, or want a more selective installation, you can install individual packages as needed.

For example, for a local setup with Ollama and HuggingFace embeddings, the installation might look like:
"""
logger.info("### Important: Ollama Environment Setup")

pip install llama-index-core llama-index-readers-file llama-index-llms-ollama llama-index-embeddings-huggingface

"""
[Check out our Starter Example with Local Models](/python/framework/getting_started/starter_example_local)

A full guide to using and configuring LLMs is available [here](/python/framework/module_guides/models/llms).

A full guide to using and configuring embedding models is available [here](/python/framework/module_guides/models/embeddings).

## Installation from Source

Git clone this repository: `git clone https://github.com/run-llama/llama_index.git`. Then do the following:

- [Install poetry](https://python-poetry.org/docs/#installation) - this will help you manage package dependencies
- If you need to run shell commands using Poetry but the shell plugin is not installed, add the plugin by running:
  ```
  poetry self add poetry-plugin-shell
  ```
- `poetry shell` - this command creates a virtual environment, which keeps installed packages contained to this project
- `pip install -e llama-index-core` - this will install the core package
- (Optional) `poetry install --with dev,docs` - this will install all dependencies needed for most local development

From there, you can install integrations as needed with `pip`, For example:
"""
logger.info("## Installation from Source")

pip install -e llama-index-integrations/readers/llama-index-readers-file
pip install -e llama-index-integrations/llms/llama-index-llms-ollama

logger.info("\n\n[DONE]", bright=True)