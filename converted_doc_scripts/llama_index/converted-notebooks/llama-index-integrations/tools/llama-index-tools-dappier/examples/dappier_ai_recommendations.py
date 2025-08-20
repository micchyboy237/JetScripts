from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.dappier import (
DappierAIRecommendationsToolSpec,
)
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
## Building a Dappier AI Recommendations Agent

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-dappier/examples/dappier_ai_recommendations.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This tutorial walks through using the LLM tools provided by [Dappier](https://dappier.com/) to allow LLMs to use Dappier's pre-trained, LLM ready RAG models and natural language APIs to ensure factual, up-to-date, responses from premium content providers across key verticals like News, Finance, Sports, Weather, and more.


To get started, you will need an [MLX API key](https://platform.openai.com/account/api-keys) and a [Dappier API key](https://platform.dappier.com/profile/api-keys)

We will import the relevant agents and tools and pass them our keys here:

## Installation

First, install the `llama-index` and `llama-index-tools-dappier` packages
"""
logger.info("## Building a Dappier AI Recommendations Agent")

# %pip install llama-index llama-index-tools-dappier

"""
## Setup API keys

You'll need to set up your API keys for MLX and Dappier.

You can go to [here](https://platform.openai.com/settings/organization/api-keys) to get API Key from Open AI.
"""
logger.info("## Setup API keys")

# from getpass import getpass

# openai_api_key = getpass("Enter your API key: ")
# os.environ["OPENAI_API_KEY"] = openai_api_key

"""
You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.
"""
logger.info("You can go to [here](https://platform.dappier.com/profile/api-keys) to get API Key from Dappier with **free** credits.")

# dappier_api_key = getpass("Enter your API key: ")
os.environ["DAPPIER_API_KEY"] = dappier_api_key

"""
## Setup Dappier Tool

Initialize the Dappier Real-Time Search Tool, convert it to a tool list.
"""
logger.info("## Setup Dappier Tool")


dappier_tool = DappierAIRecommendationsToolSpec()

dappier_tool_list = dappier_tool.to_tool_list()
for tool in dappier_tool_list:
    logger.debug(tool.metadata.name)

"""
## Usage

We've imported our MLX agent, set up the api key, and initialized our tool. Let's test out the tool before setting up our Agent.

### Sports News

Real-time news, updates, and personalized content from top sports sources like Sportsnaut, Forever Blueshirts, Minnesota Sports Fan, LAFB Network, Bounding Into Sports, and Ringside Intel.
"""
logger.info("## Usage")

logger.debug(
    dappier_tool.get_sports_news_recommendations(
        query="latest sports news", similarity_top_k=3
    )
)

"""
### Lifestyle News

Real-time updates, analysis, and personalized content from top sources like The Mix, Snipdaily, Nerdable, and Familyproof.
"""
logger.info("### Lifestyle News")

logger.debug(
    dappier_tool.get_lifestyle_news_recommendations(
        query="latest lifestyle updates", similarity_top_k=3
    )
)

"""
### iHeartDogs Articles

A dog care expert with access to articles on health, behavior, lifestyle, grooming, ownership, and more.
"""
logger.info("### iHeartDogs Articles")

logger.debug(
    dappier_tool.get_iheartdogs_recommendations(
        query="dog care tips", similarity_top_k=3
    )
)

"""
### iHeartCats Articles

A cat care expert with access to articles on health, behavior, lifestyle, grooming, ownership, and more.
"""
logger.info("### iHeartCats Articles")

logger.debug(
    dappier_tool.get_iheartcats_recommendations(
        query="cat care advice", similarity_top_k=3
    )
)

"""
### GreenMonster Articles

A helpful guide to making conscious and compassionate choices that help people, animals, and the planet.
"""
logger.info("### GreenMonster Articles")

logger.debug(
    dappier_tool.get_greenmonster_recommendations(
        query="sustainable living", similarity_top_k=3
    )
)

"""
### WISH-TV News

Politics, breaking news, multicultural news, Hispanic language content, Entertainment, Health, Education, and many more.
"""
logger.info("### WISH-TV News")

logger.debug(
    dappier_tool.get_wishtv_recommendations(
        query="latest breaking news", similarity_top_k=3
    )
)

"""
### 9 and 10 News

Up-to-date local news, weather forecasts, sports coverage, and community stories for Northern Michigan, including the Cadillac and Traverse City areas.
"""
logger.info("### 9 and 10 News")

logger.debug(
    dappier_tool.get_nine_and_ten_news_recommendations(
        query="northern michigan local news", similarity_top_k=3
    )
)

"""
### Using the Dappier AI Recommendations tool in an Agent

We can create an agent with access to the Dappier AI Recommendations tool start testing it out:
"""
logger.info("### Using the Dappier AI Recommendations tool in an Agent")


agent = FunctionAgent(
    tools=dappier_tool_list,
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit"),
)

logger.debug(
    await agent.run(
        "Get latest sports news, lifestyle news, breaking news, dog care advice and summarize it into different sections, with source links."
    )
)

logger.info("\n\n[DONE]", bright=True)