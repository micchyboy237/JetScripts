from jet.llm.ollama.base_langchain import MLXEmbeddings
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain_huggingface import HuggingFaceEmbeddings
from mem0 import Memory
import os
import shutil
import { MLXEmbeddings } from "@langchain/openai"
import { Memory } from "mem0ai"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LangChain
---

Mem0 supports LangChain as a provider to access a wide range of embedding models. LangChain is a framework for developing applications powered by language models, making it easy to integrate various embedding providers through a consistent interface.

For a complete list of available embedding models supported by LangChain, refer to the [LangChain Text Embedding documentation](https://python.langchain.com/docs/integrations/text_embedding/).

## Usage

<CodeGroup>
"""
logger.info("## Usage")


# os.environ["OPENAI_API_KEY"] = "your-api-key"

openai_embeddings = MLXEmbeddings(
    model="mxbai-embed-large",
    dimensions=1536
)

config = {
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": openai_embeddings
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""

"""


embeddings = new MLXEmbeddings()
config = {
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": embeddings
        }
    }
}

memory = new Memory(config)

messages = [
    { role: "user", content: "I'm planning to watch a movie tonight. Any recommendations?" },
    { role: "assistant", content: "How about a thriller movies? They can be quite engaging." },
    { role: "user", content: "I'm not a big fan of thriller movies but I love sci-fi movies." },
    { role: "assistant", content: "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future." }
]

memory.add(messages, user_id="alice", metadata={"category": "movies"})

"""
</CodeGroup>

## Supported LangChain Embedding Providers

LangChain supports a wide range of embedding providers, including:

- MLX (`MLXEmbeddings`)
- Cohere (`CohereEmbeddings`)
- Google (`VertexAIEmbeddings`)
- Hugging Face (`HuggingFaceEmbeddings`)
- Sentence Transformers (`HuggingFaceEmbeddings`)
- Azure MLX (`AzureMLXEmbeddings`)
- Ollama(`OllamaEmbeddings`)
- Together (`TogetherEmbeddings`)
- And many more

You can use any of these model instances directly in your configuration. For a complete and up-to-date list of available embedding providers, refer to the [LangChain Text Embedding documentation](https://python.langchain.com/docs/integrations/text_embedding/).

## Provider-Specific Configuration

When using LangChain as an embedder provider, you'll need to:

1. Set the appropriate environment variables for your chosen embedding provider
2. Import and initialize the specific model class you want to use
3. Pass the initialized model instance to the config

### Examples with Different Providers

#### HuggingFace Embeddings
"""
logger.info("## Supported LangChain Embedding Providers")


hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

config = {
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": hf_embeddings
        }
    }
}

"""
#### Ollama Embeddings
"""
logger.info("#### Ollama Embeddings")


ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

config = {
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": ollama_embeddings
        }
    }
}

"""
<Note>
  Make sure to install the necessary LangChain packages and any provider-specific dependencies.
</Note>

## Config

All available parameters for the `langchain` embedder config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Config")

logger.info("\n\n[DONE]", bright=True)