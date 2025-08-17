from jet.llm.ollama.base_langchain import MLXEmbeddings
from jet.logger import CustomLogger
from langchain_community.vectorstores import Chroma
from mem0 import Memory
import os
import shutil
import { MLXEmbeddings } from "@langchain/openai"
import { Memory } from "mem0ai"
import { MemoryVectorStore as LangchainMemoryStore } from "langchain/vectorstores/memory"


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

Mem0 supports LangChain as a provider for vector store integration. LangChain provides a unified interface to various vector databases, making it easy to integrate different vector store providers through a consistent API.

<Note>
  When using LangChain as your vector store provider, you must set the collection name to "mem0". This is a required configuration for proper integration with Mem0.
</Note>

## Usage

<CodeGroup>
"""
logger.info("## Usage")


embeddings = MLXEmbeddings()
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="mem0"  # Required collection name
)

config = {
    "vector_store": {
        "provider": "langchain",
        "config": {
            "client": vector_store
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
vectorStore = new LangchainVectorStore(embeddings)

config = {
    "vector_store": {
        "provider": "langchain",
        "config": { "client": vectorStore }
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

## Supported LangChain Vector Stores

LangChain supports a wide range of vector store providers, including:

- Chroma
- FAISS
- Pinecone
- Weaviate
- Milvus
- Qdrant
- And many more

You can use any of these vector store instances directly in your configuration. For a complete and up-to-date list of available providers, refer to the [LangChain Vector Stores documentation](https://python.langchain.com/docs/integrations/vectorstores).

## Limitations

When using LangChain as a vector store provider, there are some limitations to be aware of:

1. **Bulk Operations**: The `get_all` and `delete_all` operations are not supported when using LangChain as the vector store provider. This is because LangChain's vector store interface doesn't provide standardized methods for these bulk operations across all providers.

2. **Provider-Specific Features**: Some advanced features may not be available depending on the specific vector store implementation you're using through LangChain.

## Provider-Specific Configuration

When using LangChain as a vector store provider, you'll need to:

1. Set the appropriate environment variables for your chosen vector store provider
2. Import and initialize the specific vector store class you want to use
3. Pass the initialized vector store instance to the config

<Note>
  Make sure to install the necessary LangChain packages and any provider-specific dependencies.
</Note>

## Config

All available parameters for the `langchain` vector store config are present in [Master List of All Params in Config](../config).
"""
logger.info("## Supported LangChain Vector Stores")

logger.info("\n\n[DONE]", bright=True)