from jet.logger import logger
from llama_index.core.retrievers import SummaryIndexLLMRetriever
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
# Retriever

## Concept

Retrievers are responsible for fetching the most relevant context given a user query (or chat message).

It can be built on top of [indexes](/python/framework/module_guides/indexing), but can also be defined independently.
It is used as a key building block in [query engines](/python/framework/module_guides/deploying/query_engine) (and [Chat Engines](/python/framework/module_guides/deploying/chat_engines)) for retrieving relevant context.

<Aside type="tip">
Confused about where retriever fits in the RAG workflow? Read about [high-level concepts](/python/framework/getting_started/concepts)
</Aside>

## Usage Pattern

Get started with:
"""
logger.info("# Retriever")

retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")

"""
## Get Started

Get a retriever from index:
"""
logger.info("## Get Started")

retriever = index.as_retriever()

"""
Retrieve relevant context for a question:
"""
logger.info("Retrieve relevant context for a question:")

nodes = retriever.retrieve("Who is Paul Graham?")

"""
> Note: To learn how to build an index, see [Indexing](/python/framework/module_guides/indexing)

## High-Level API

### Selecting a Retriever

You can select the index-specific retriever class via `retriever_mode`.
For example, with a `SummaryIndex`:
"""
logger.info("## High-Level API")

retriever = summary_index.as_retriever(
    retriever_mode="llm",
)

"""
This creates a [SummaryIndexLLMRetriever](/python/framework/api_reference/retrievers/summary) on top of the summary index.

See [**Retriever Modes**](/python/framework/module_guides/querying/retriever/retriever_modes) for a full list of (index-specific) retriever modes
and the retriever classes they map to.

### Configuring a Retriever

In the same way, you can pass kwargs to configure the selected retriever.

> Note: take a look at the API reference for the selected retriever class' constructor parameters for a list of valid kwargs.

For example, if we selected the "llm" retriever mode, we might do the following:
"""
logger.info("### Configuring a Retriever")

retriever = summary_index.as_retriever(
    retriever_mode="llm",
    choice_batch_size=5,
)

"""
## Low-Level Composition API

You can use the low-level composition API if you need more granular control.

To achieve the same outcome as above, you can directly import and construct the desired retriever class:
"""
logger.info("## Low-Level Composition API")


retriever = SummaryIndexLLMRetriever(
    index=summary_index,
    choice_batch_size=5,
)

"""
## Examples

See more examples in the [retrievers guide](/python/framework/module_guides/querying/retriever/retrievers).
"""
logger.info("## Examples")

logger.info("\n\n[DONE]", bright=True)