from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_dappier import DappierAIRecommendationTool
from langchain_dappier import DappierRealTimeSearchTool
import ChatModelTabs from "@theme/ChatModelTabs";
import datetime
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

[Dappier](https://dappier.com) connects any LLM or your Agentic AI to real-time, rights-cleared, proprietary data from trusted sources, making your AI an expert in anything. Our specialized models include Real-Time Web Search, News, Sports, Financial Stock Market Data, Crypto Data, and exclusive content from premium publishers. Explore a wide range of data models in our marketplace at [marketplace.dappier.com](https://marketplace.dappier.com).

[Dappier](https://dappier.com) delivers enriched, prompt-ready, and contextually relevant data strings, optimized for seamless integration with LangChain. Whether you're building conversational AI, recommendation engines, or intelligent search, Dappier's LLM-agnostic RAG models ensure your AI has access to verified, up-to-date data—without the complexity of building and managing your own retrieval pipeline.

# Dappier Tool

This will help you get started with the Dappier [tool](https://python.langchain.com/docs/concepts/tools/). For detailed documentation of all DappierRetriever features and configurations head to the [API reference](https://python.langchain.com/en/latest/tools/langchain_dappier.tools.Dappier.DappierRealTimeSearchTool.html).

## Overview

The DappierRealTimeSearchTool and DappierAIRecommendationTool empower AI applications with real-time data and AI-driven insights. The former provides access to up-to-date information across news, weather, travel, and financial markets, while the latter supercharges applications with factual, premium content from diverse domains like News, Finance, and Sports, all powered by Dappier's pre-trained RAG models and natural language APIs.

### Setup

This tool lives in the `langchain-dappier` package.
"""
logger.info("# Dappier")

# %pip install -qU langchain-dappier

"""
### Credentials

We also need to set our Dappier API credentials, which can be generated at the [Dappier site.](https://platform.dappier.com/profile/api-keys).
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("DAPPIER_API_KEY"):
#     os.environ["DAPPIER_API_KEY"] = getpass.getpass("Dappier API key:\n")

"""
If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
## DappierRealTimeSearchTool

Access real-time Google search results, including the latest news, weather, travel, and deals, along with up-to-date financial news, stock prices, and trades from polygon.io, all powered by AI insights to keep you informed.

### Instantiation

- ai_model_id: str
    The AI model ID to use for the query. The AI model ID always starts
    with the prefix "am_".

    Defaults to "am_01j06ytn18ejftedz6dyhz2b15".

    Multiple AI model IDs are available, which can be found at:
    https://marketplace.dappier.com/marketplace
"""
logger.info("## DappierRealTimeSearchTool")


tool = DappierRealTimeSearchTool(
)

"""
### Invocation

#### [Invoke directly with args](/docs/concepts/tools)

The `DappierRealTimeSearchTool` takes a single "query" argument, which should be a natural language query:
"""
logger.info("### Invocation")

tool.invoke({"query": "What happened at the last wimbledon"})

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {"query": "euro 2024 host nation"},
    "id": "1",
    "name": "dappier",
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)

logger.debug(tool_msg.content[:400])

"""
### Chaining

We can use our tool in a chain by first binding it to a [tool-calling model](/docs/how_to/tool_calling/) and then calling it:


<ChatModelTabs customVarName="llm" />
"""
logger.info("### Chaining")


llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0)



today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([tool])

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("who won the last womens singles wimbledon")

"""
## DappierAIRecommendationTool

Supercharge your AI applications with Dappier's pre-trained RAG models and natural language APIs, delivering factual and up-to-date responses from premium content providers across verticals like News, Finance, Sports, Weather, and more.

### Instantiation

- data_model_id: str  
  The data model ID to use for recommendations. Data model IDs always start with the prefix "dm_". Defaults to "dm_01j0pb465keqmatq9k83dthx34".  
  Multiple data model IDs are available, which can be found at [Dappier marketplace](https://marketplace.dappier.com/marketplace).  

- similarity_top_k: int  
  The number of top documents to retrieve based on similarity. Defaults to "9".  

- ref: Optional[str]
  The site domain where AI recommendations should be displayed. Defaults to "None".  

- num_articles_ref: int
  The minimum number of articles to return from the specified reference domain ("ref"). The remaining articles will come from other sites in the RAG model. Defaults to "0".  

- search_algorithm: Literal["most_recent", "semantic", "most_recent_semantic", "trending"]
  The search algorithm to use for retrieving articles. Defaults to "most_recent".
"""
logger.info("## DappierAIRecommendationTool")


tool = DappierAIRecommendationTool(
    data_model_id="dm_01j0pb465keqmatq9k83dthx34",
    similarity_top_k=3,
    ref="sportsnaut.com",
    num_articles_ref=2,
    search_algorithm="most_recent",
)

"""
### Invocation

#### [Invoke directly with args](/docs/concepts/tools)

The `DappierAIRecommendationTool` takes a single "query" argument, which should be a natural language query:
"""
logger.info("### Invocation")

tool.invoke({"query": "latest sports news"})

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {"query": "top 3 news articles"},
    "id": "1",
    "name": "dappier",
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)

logger.debug(tool_msg.content[:400])

"""
## API reference

For detailed documentation of all DappierRealTimeSearchTool features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/tools/langchain_dappier.tools.dappier.tool.DappierRealTimeSearchTool.html)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)