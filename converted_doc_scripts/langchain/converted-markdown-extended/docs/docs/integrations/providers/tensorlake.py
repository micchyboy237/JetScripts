from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_tensorlake import document_markdown_tool
from langgraph.prebuilt import create_react_agent
import asyncio
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
# Tensorlake

Tensorlake is the AI Data Cloud that reliably transforms data from unstructured sources into ingestion-ready formats for AI Applications.

The `langchain-tensorlake` package provides seamless integration between [Tensorlake](https://tensorlake.ai) and [LangChain](https://langchain.com),
enabling you to build sophisticated document processing agents with enhanced parsing features, like signature detection.

## Tensorlake feature overview

Tensorlake gives you tools to:
- Extract: Schema-driven structured data extraction to pull out specific fields from documents.
- Parse: Convert documents to markdown to build RAG/Knowledge Graph systems.
- Orchestrate: Build programmable workflows for large-scale ingestion and enrichment of Documents, Text, Audio, Video and more.

Learn more at [docs.tensorlake.ai](https://docs.tensorlake.ai/introduction)

---

## Installation
"""
logger.info("# Tensorlake")

pip install -U langchain-tensorlake

"""
---

## Examples

Follow a [full tutorial](https://docs.tensorlake.ai/examples/tutorials/real-estate-agent-with-langgraph-cli) on how to detect signatures in unstructured documents using the `langchain-tensorlake` tool.

Or check out this [colab notebook](https://colab.research.google.com/drive/1VRWIPCWYnjcRtQL864Bqm9CJ6g4EpRqs?usp=sharing) for a quick start.

---
## Quick Start

### 1. Set up your environment

You should configure credentials for Tensorlake and Ollama by setting the following environment variables:

export TENSORLAKE_API_KEY="your-tensorlake-api-key"
# export OPENAI_API_KEY = "your-ollama-api-key"

Get your Tensorlake API key from the [Tensorlake Cloud Console](https://cloud.tensorlake.ai/). New users get 100 free credits.

### 2. Import necessary packages
"""
logger.info("## Examples")


"""
### 3. Build a Signature Detection Agent
"""
logger.info("### 3. Build a Signature Detection Agent")

async def main(question):
    agent = create_react_agent(
            model="ollama:llama3.2",
            tools=[document_markdown_tool],
            prompt=(
                """
                I have a document that needs to be parsed. \n\nPlease parse this document and answer the question about it.
                """
            ),
            name="real-estate-agent",
        )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    logger.success(format_json(result))

    logger.debug(result["messages"][-1].content)

"""
*Note:* We highly recommend using `ollama` as the agent model to ensure the agent sets the right parsing parameters

### 4. Example Usage
"""
logger.info("### 4. Example Usage")

path = "path/to/your/document.pdf"

question = f"What contextual information can you extract about the signatures in my document found at {path}?"

if __name__ == "__main__":
    asyncio.run(main(question))

"""
## Need help?

Reach out to us on [Slack](https://join.slack.com/t/tensorlakecloud/shared_invite/zt-32fq4nmib-gO0OM5RIar3zLOBm~ZGqKg) or on the
[package repository on GitHub](https://github.com/tensorlakeai/langchain-tensorlake) directly.
"""
logger.info("## Need help?")

logger.info("\n\n[DONE]", bright=True)