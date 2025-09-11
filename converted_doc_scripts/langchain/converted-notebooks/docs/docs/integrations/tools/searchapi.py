from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.utilities import SearchApiAPIWrapper
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
# SearchApi

This notebook shows examples of how to use SearchApi to search the web. Go to [https://www.searchapi.io/](https://www.searchapi.io/) to sign up for a free account and get API key.
"""
logger.info("# SearchApi")


os.environ["SEARCHAPI_API_KEY"] = ""


search = SearchApiAPIWrapper()

search.run("Obama's first name?")

"""
## Using as part of a Self Ask With Search Chain
"""
logger.info("## Using as part of a Self Ask With Search Chain")

# os.environ["OPENAI_API_KEY"] = ""


llm = ChatOllama(temperature=0)
search = SearchApiAPIWrapper()
tools = [
    Tool(
        name="intermediate_answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "Who lived longer: Plato, Socrates, or Aristotle?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Custom parameters

SearchApi wrapper can be customized to use different engines like [Google News](https://www.searchapi.io/docs/google-news), [Google Jobs](https://www.searchapi.io/docs/google-jobs), [Google Scholar](https://www.searchapi.io/docs/google-scholar), or others which can be found in [SearchApi](https://www.searchapi.io/docs/google) documentation. All parameters supported by SearchApi can be passed when executing the query.
"""
logger.info("## Custom parameters")

search = SearchApiAPIWrapper(engine="google_jobs")

search.run("AI Engineer", location="Portugal", gl="pt")[0:500]

"""
## Getting results with metadata
"""
logger.info("## Getting results with metadata")


search = SearchApiAPIWrapper(engine="google_scholar")
results = search.results("Large Language Models")
pprint.pp(results)

logger.info("\n\n[DONE]", bright=True)
