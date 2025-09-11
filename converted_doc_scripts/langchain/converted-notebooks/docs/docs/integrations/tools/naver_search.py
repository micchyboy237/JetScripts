from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_naver_community.tool import NaverNewsSearch
from langchain_naver_community.tool import NaverSearchResults
from langchain_naver_community.utils import NaverSearchAPIWrapper
from langgraph.prebuilt import create_react_agent
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
# Naver Search

The Naver Search Tool provides a simple interface to search Naver and get results.

### Integration details

| Class | Package | Serializable | JS support |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| NaverSearchResults | [langchain-naver-community](https://pypi.org/project/langchain-naver-community/) | ❌ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-naver-community?style=flat-square&label=%20) |

### Tool features

**Search** : The Naver Search Tool provides a simple interface to search Naver and get results.

## Setup
### Setting Up API Credentials
To use Naver Search, you need to obtain API credentials. Follow these steps:

Sign in to the [Naver Developers portal](https://developers.naver.com/main/).
Create a new application and enable the Search API.
Obtain your **NAVER_CLIENT_ID** and **NAVER_CLIENT_SECRET** from the "Application List" section.

### Setting Up Environment Variables
After obtaining the credentials, set them as environment variables in your script:
"""
logger.info("# Naver Search")

# %pip install --upgrade --quiet  langchain-naver-community

# import getpass

if not os.environ.get("NAVER_CLIENT_ID"):
#     os.environ["NAVER_CLIENT_ID"] = getpass.getpass("Enter your Naver Client ID:\n")

if not os.environ.get("NAVER_CLIENT_SECRET"):
#     os.environ["NAVER_CLIENT_SECRET"] = getpass.getpass(
        "Enter your Naver Client Secret:\n"
    )

"""
## Instantiation
"""
logger.info("## Instantiation")


search = NaverSearchAPIWrapper()

"""
## Invocation
"""
logger.info("## Invocation")

search.results("Seoul")[:3]

"""
## Tool Usage
"""
logger.info("## Tool Usage")


search = NaverSearchAPIWrapper()

tool = NaverSearchResults(api_wrapper=search)

tool.invoke("what is the weather in seoul?")[3:5]

"""
## Use within an agent

The Naver Search tool can be integrated into LangChain agents for more complex tasks. Below we demonstrate how to set up an agent that can search Naver for current information.
"""
logger.info("## Use within an agent")


llm = ChatOllama(model="llama3.2")

system_prompt = """
You are a helpful assistant that can search the web for information.
"""


tools = [NaverNewsSearch()]

agent_executor = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)

"""
Now we can run the agent with a query.
"""
logger.info("Now we can run the agent with a query.")

query = "What is the weather in Seoul?"
result = agent_executor.invoke({"messages": [("human", query)]})
result["messages"][-1].content

"""
## API reference
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)