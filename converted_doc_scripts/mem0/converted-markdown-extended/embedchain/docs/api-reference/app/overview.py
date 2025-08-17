from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: "App"
---

Create a RAG app object on Embedchain. This is the main entrypoint for a developer to interact with Embedchain APIs. An app configures the llm, vector database, embedding model, and retrieval strategy of your choice.

### Attributes

<ParamField path="local_id" type="str">
    App ID
</ParamField>
<ParamField path="name" type="str" optional>
    Name of the app
</ParamField>
<ParamField path="config" type="BaseConfig">
    Configuration of the app
</ParamField>
<ParamField path="llm" type="BaseLlm">
    Configured LLM for the RAG app
</ParamField>
<ParamField path="db" type="BaseVectorDB">
    Configured vector database for the RAG app
</ParamField>
<ParamField path="embedding_model" type="BaseEmbedder">
    Configured embedding model for the RAG app
</ParamField>
<ParamField path="chunker" type="ChunkerConfig">
    Chunker configuration
</ParamField>
<ParamField path="client" type="Client" optional>
    Client object (used to deploy an app to Embedchain platform)
</ParamField>
<ParamField path="logger" type="logging.Logger">
    Logger object
</ParamField>

## Usage

You can create an app instance using the following methods:

### Default setting
"""
logger.info("### Attributes")

app = App()

"""
### Python Dict
"""
logger.info("### Python Dict")


config_dict = {
  'llm': {
    'provider': 'gpt4all',
    'config': {
      'model': 'orca-mini-3b-gguf2-q4_0.gguf',
      'temperature': 0.5,
      'max_tokens': 1000,
      'top_p': 1,
      'stream': False
    }
  },
  'embedder': {
    'provider': 'gpt4all'
  }
}

app = App.from_config(config=config_dict)

"""
### YAML Config

<CodeGroup>
"""
logger.info("### YAML Config")


app = App.from_config(config_path="config.yaml")

"""

"""

llm:
  provider: gpt4all
  config:
    model: 'orca-mini-3b-gguf2-q4_0.gguf'
    temperature: 0.5
    max_tokens: 1000
    top_p: 1
    stream: false

embedder:
  provider: gpt4all

"""
</CodeGroup>

### JSON Config

<CodeGroup>
"""
logger.info("### JSON Config")


app = App.from_config(config_path="config.json")

"""

"""

{
  "llm": {
    "provider": "gpt4all",
    "config": {
      "model": "orca-mini-3b-gguf2-q4_0.gguf",
      "temperature": 0.5,
      "max_tokens": 1000,
      "top_p": 1,
      "stream": false
    }
  },
  "embedder": {
    "provider": "gpt4all"
  }
}

"""
</CodeGroup>
"""

logger.info("\n\n[DONE]", bright=True)