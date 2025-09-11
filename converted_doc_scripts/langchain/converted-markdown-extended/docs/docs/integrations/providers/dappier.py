from jet.logger import logger
from langchain_community.chat_models import ChatDappierAI
from langchain_dappier import (
DappierRealTimeSearchTool,
DappierAIRecommendationTool
)
from langchain_dappier import DappierRetriever
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
# Dappier

[Dappier](https://dappier.com) connects any LLM or your Agentic AI to
real-time, rights-cleared, proprietary data from trusted sources,
making your AI an expert in anything. Our specialized models include
Real-Time Web Search, News, Sports, Financial Stock Market Data,
Crypto Data, and exclusive content from premium publishers. Explore a
wide range of data models in our marketplace at
[marketplace.dappier.com](https://marketplace.dappier.com).

[Dappier](https://dappier.com) delivers enriched, prompt-ready, and
contextually relevant data strings, optimized for seamless integration
with LangChain. Whether you're building conversational AI, recommendation
engines, or intelligent search, Dappier's LLM-agnostic RAG models ensure
your AI has access to verified, up-to-date data—without the complexity of
building and managing your own retrieval pipeline.

## Installation and Setup

Install ``langchain-dappier`` and set environment variable
``DAPPIER_API_KEY``.
"""
logger.info("# Dappier")

pip install -U langchain-dappier
export DAPPIER_API_KEY="your-api-key"

"""
We also need to set our Dappier API credentials, which can be generated at
the [Dappier site.](https://platform.dappier.com/profile/api-keys).

We can find the supported data models by heading over to the
[Dappier marketplace.](https://platform.dappier.com/marketplace)

## Chat models

See a [usage example](/docs/integrations/chat/dappier).
"""
logger.info("## Chat models")


"""
## Retriever

See a [usage example](/docs/integrations/retrievers/dappier).
"""
logger.info("## Retriever")


"""
## Tool

See a [usage example](/docs/integrations/tools/dappier).
"""
logger.info("## Tool")


logger.info("\n\n[DONE]", bright=True)