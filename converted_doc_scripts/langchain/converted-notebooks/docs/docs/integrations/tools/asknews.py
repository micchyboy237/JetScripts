from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain_community.tools.asknews import AskNewsSearch
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
# AskNews

> [AskNews](https://asknews.app) infuses any LLM with the latest global news (or historical news), using a single natural language query. Specifically, AskNews is enriching over 300k articles per day by translating, summarizing, extracting entities, and indexing them into hot and cold vector databases. AskNews puts these vector databases on a low-latency endpoint for you. When you query AskNews, you get back a prompt-optimized string that contains all the most pertinent enrichments (e.g. entities, classifications, translation, summarization). This means that you do not need to manage your own news RAG, and you do not need to worry about how to properly convey news information in a condensed way to your LLM.
> AskNews is also committed to transparency, which is why our coverage is monitored and diversified across hundreds of countries, 13 languages, and 50 thousand sources. If you'd like to track our source coverage, you can visit our [transparency dashboard](https://asknews.app/en/transparency).

## Setup

The integration lives in the `langchain-community` package. We also need to install the `asknews` package itself.

```bash
pip install -U langchain-community asknews
```

We also need to set our AskNews API credentials, which can be obtained at the [AskNews console](https://my.asknews.app).
"""
logger.info("# AskNews")

# import getpass

# os.environ["ASKNEWS_CLIENT_ID"] = getpass.getpass()
# os.environ["ASKNEWS_CLIENT_SECRET"] = getpass.getpass()

"""
## Usage

Here we show how to use the tool individually.
"""
logger.info("## Usage")


tool = AskNewsSearch(max_results=2)
tool.invoke({"query": "Effect of fed policy on tech sector"})

"""
## Chaining
We show here how to use it as part of an agent. We use the Ollama Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.

```bash
pip install -U langchain-ollama langchainhub
```
"""
logger.info("## Chaining")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass()


prompt = hub.pull("hwchase17/ollama-functions-agent")
llm = ChatOllama(model="llama3.2")
asknews_tool = AskNewsSearch()
tools = [asknews_tool]
agent = create_ollama_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"input": "How is the tech sector being affected by fed policy?"})

logger.info("\n\n[DONE]", bright=True)