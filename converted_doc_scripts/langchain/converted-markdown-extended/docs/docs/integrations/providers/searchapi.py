from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import SearchApiAPIWrapper
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
# SearchApi

This page covers how to use the [SearchApi](https://www.searchapi.io/) Google Search API within LangChain. SearchApi is a real-time SERP API for easy SERP scraping.

## Setup

- Go to [https://www.searchapi.io/](https://www.searchapi.io/) to sign up for a free account
- Get the api key and set it as an environment variable (`SEARCHAPI_API_KEY`)

## Wrappers

### Utility

There is a SearchApiAPIWrapper utility which wraps this API. To import this utility:
"""
logger.info("# SearchApi")


"""
You can use it as part of a Self Ask chain:
"""
logger.info("You can use it as part of a Self Ask chain:")



os.environ["SEARCHAPI_API_KEY"] = ""
# os.environ['OPENAI_API_KEY'] = ""

llm = Ollama(temperature=0)
search = SearchApiAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("Who lived longer: Plato, Socrates, or Aristotle?")

"""
#### Output

> Entering new AgentExecutor chain...
 Yes.
Follow up: How old was Plato when he died?
Intermediate answer: eighty
Follow up: How old was Socrates when he died?
Intermediate answer: | Socrates |
| -------- |
| Born | c. 470 BC Deme Alopece, Athens |
| Died | 399 BC (aged approximately 71) Athens |
| Cause of death | Execution by forced suicide by poisoning |
| Spouse(s) | Xanthippe, Myrto |

Follow up: How old was Aristotle when he died?
Intermediate answer: 62 years
So the final answer is: Plato

> Finished chain.
'Plato'

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
"""
logger.info("#### Output")

tools = load_tools(["searchapi"])

"""
For more information on tools, see [this page](/docs/how_to/tools_builtin).
"""
logger.info("For more information on tools, see [this page](/docs/how_to/tools_builtin).")

logger.info("\n\n[DONE]", bright=True)