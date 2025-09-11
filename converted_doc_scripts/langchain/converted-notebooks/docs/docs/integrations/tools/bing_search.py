from jet.adapters.langchain.chat_ollama import AzureChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
import json
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
# Bing Search

> [Bing Search](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/) is an Azure service and enables safe, ad-free, location-aware search results, surfacing relevant information from billions of web documents. Help your users find what they're looking for from the world-wide-web by harnessing Bing's ability to comb billions of webpages, images, videos, and news with a single API call.

## Setup
Following the [instruction](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource) to create Azure Bing Search v7 service, and get the subscription key

The integration lives in the `langchain-community` package.
"""
logger.info("# Bing Search")

# %pip install -U langchain-community

# import getpass

# os.environ["BING_SUBSCRIPTION_KEY"] = getpass.getpass()
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"


search = BingSearchAPIWrapper(k=4)

search.run("python")

"""
## Number of results
You can use the `k` parameter to set the number of results
"""
logger.info("## Number of results")

search = BingSearchAPIWrapper(k=1)

search.run("python")

"""
## Metadata Results

Run query through BingSearch and return snippet, title, and link metadata.

- Snippet: The description of the result.
- Title: The title of the result.
- Link: The link to the result.
"""
logger.info("## Metadata Results")

search = BingSearchAPIWrapper()

search.results("apples", 5)

"""
## Tool Usage
"""
logger.info("## Tool Usage")



api_wrapper = BingSearchAPIWrapper()
tool = BingSearchResults(api_wrapper=api_wrapper)
tool


response = tool.invoke("What is the weather in Shanghai?")
response = json.loads(response.replace("'", '"'))
for item in response:
    logger.debug(item)

"""
## Chaining

We show here how to use it as part of an [agent](/docs/tutorials/agents). We use the Ollama Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.
"""
logger.info("## Chaining")

# %pip install --upgrade --quiet langchain langchain-ollama langchainhub langchain-community

# import getpass


# os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass()
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint>.ollama.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "<your-deployment-name>"

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = AzureChatOllama(
#     ollama_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    ollama_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
tool = BingSearchResults(api_wrapper=api_wrapper)
tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)
agent_executor.invoke({"input": "What happened in the latest burning man floods?"})

logger.info("\n\n[DONE]", bright=True)