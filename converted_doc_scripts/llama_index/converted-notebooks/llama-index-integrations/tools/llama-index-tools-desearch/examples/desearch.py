from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index_desearch.tools import DesearchToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Desearch ToolSpace

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-exa/examples/desearch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This tutorial walks through using the LLM tools provided by the [Desearch API](https://desearch.ai) to allow LLMs to use semantic queries to search for and retrieve rich web content from the internet.

To get started, you will need an [Desearch API key](https://console.desearch.ai/api-keys)

We will import the tools and pass them our keys here:
"""
logger.info("# Desearch ToolSpace")

# !pip install llama-index llama-index-core llama-index-tools-desearch


desearch_tool = DesearchToolSpec(
    api_key=os.environ["DESEARCH_API_KEY"],
)

exa_tool_list = desearch_tool.to_tool_list()
for tool in exa_tool_list:
    logger.debug(tool.metadata.name)

"""
ai_search_tool

twitter_search_tool

web_search_tool

## Testing the Desearch tools

We've imported our MLX agent, set up the API keys, and initialized our tool, checking the methods that it has available. Let's test out the tool before setting up our Agent.

All of the Desearch search tools make use of the `AutoPrompt` option where Desearch will pass the query through an LLM to refine it in line with Desearch query best-practice.

The Desearch API allows you to perform AI-powered web searches, gathering relevant information from multiple sources, including web pages, research papers, and social media discussions.
"""
logger.info("## Testing the Desearch tools")

desearch_tool.ai_search_tool(
    prompt="Bittensor",
    tool=["web"],
    model="NOVA",
    date_filter="PAST_24_HOURS"
)

"""
The X Search API enables users to retrieve relevant links and tweets based on specified search queries without utilizing AI-driven models. It analyzes links from X posts that align with the provided search criteria.
"""
logger.info("The X Search API enables users to retrieve relevant links and tweets based on specified search queries without utilizing AI-driven models. It analyzes links from X posts that align with the provided search criteria.")

desearch_tool.twitter_search_tool(
    query="bittensor",
    sort="Top",
    count=20,
)

"""
This API allows users to search for any information over the web. This replicates a typical search engine experience, where users can search for any information they need.
"""
logger.info("This API allows users to search for any information over the web. This replicates a typical search engine experience, where users can search for any information they need.")

desearch_tool.web_search_tool(
    query="bittensor",
    num=10,
    start=0,
)

"""
## Creating the Agent

We now are ready to create an Agent that can use Exa's services to their full potential. We will use our wrapped read and load tools, as well as the `get_date` utility for the following agent and test it out below:
"""
logger.info("## Creating the Agent")


agent = FunctionAgent(
    tools=[*wrapped_retrieve.to_tool_list(), date_tool],
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

logger.debug(
    await agent.run(
        "Can you summarize everything published in the last month regarding news on superconductors"
    )
)

logger.info("\n\n[DONE]", bright=True)