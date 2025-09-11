from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
import os
import pprint
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
# Google Serper

This notebook goes over how to use the `Google Serper` component to search the web. First you need to sign up for a free account at [serper.dev](https://serper.dev) and get your api key.
"""
logger.info("# Google Serper")

# %pip install --upgrade --quiet  langchain-community langchain-ollama


os.environ["SERPER_API_KEY"] = "your-serper-api-key"


search = GoogleSerperAPIWrapper()

search.run("Obama's first name?")

"""
## As part of a Self Ask With Search Agent

In order to create an agent that uses the Google Serper tool install Langgraph
"""
logger.info("## As part of a Self Ask With Search Agent")

# %pip install --upgrade --quiet langgraph langchain-ollama

"""
# and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPENAI_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.
"""
# logger.info("and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPENAI_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.")


# os.environ["OPENAI_API_KEY"] = "[your ollama key]"

llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate_Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]
agent = create_react_agent(llm, tools)

events = agent.stream(
    {
        "messages": [
            ("user", "What is the hometown of the reigning men's U.S. Open champion?")
        ]
    },
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## Obtaining results with metadata
If you would also like to obtain the results in a structured way including metadata. For this we will be using the `results` method of the wrapper.
"""
logger.info("## Obtaining results with metadata")

search = GoogleSerperAPIWrapper()
results = search.results("Apple Inc.")
pprint.pp(results)

"""
## Searching for Google Images
We can also query Google Images using this wrapper. For example:
"""
logger.info("## Searching for Google Images")

search = GoogleSerperAPIWrapper(type="images")
results = search.results("Lion")
pprint.pp(results)

"""
## Searching for Google News
We can also query Google News using this wrapper. For example:
"""
logger.info("## Searching for Google News")

search = GoogleSerperAPIWrapper(type="news")
results = search.results("Tesla Inc.")
pprint.pp(results)

"""
If you want to only receive news articles published in the last hour, you can do the following:
"""
logger.info("If you want to only receive news articles published in the last hour, you can do the following:")

search = GoogleSerperAPIWrapper(type="news", tbs="qdr:h")
results = search.results("Tesla Inc.")
pprint.pp(results)

"""
Some examples of the `tbs` parameter:

`qdr:h` (past hour)
`qdr:d` (past day)
`qdr:w` (past week)
`qdr:m` (past month)
`qdr:y` (past year)

You can specify intermediate time periods by adding a number:
`qdr:h12` (past 12 hours)
`qdr:d3` (past 3 days)
`qdr:w2` (past 2 weeks)
`qdr:m6` (past 6 months)
`qdr:y2` (past 2 years)

For all supported filters simply go to [Google Search](https://google.com), search for something, click on "Tools", add your date filter and check the URL for "tbs=".

## Searching for Google Places
We can also query Google Places using this wrapper. For example:
"""
logger.info("## Searching for Google Places")

search = GoogleSerperAPIWrapper(type="places")
results = search.results("Italian restaurants in Upper East Side")
pprint.pp(results)

logger.info("\n\n[DONE]", bright=True)