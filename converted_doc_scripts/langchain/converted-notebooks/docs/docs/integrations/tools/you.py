from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_ollama_functions_agent
from langchain_community.tools.you import YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper
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
# You.com Search

The [you.com API](https://api.you.com) is a suite of tools designed to help developers ground the output of LLMs in the most recent, most accurate, most relevant information that may not have been included in their training dataset.

## Setup

The tool lives in the `langchain-community` package.

You also need to set your you.com API key.
"""
logger.info("# You.com Search")

# %pip install --upgrade --quiet langchain-community


os.environ["YDC_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability")



"""
## Tool Usage
"""
logger.info("## Tool Usage")


api_wrapper = YouSearchAPIWrapper(num_web_results=1)
tool = YouSearchTool(api_wrapper=api_wrapper)

tool

response = tool.invoke("What is the weather in NY")

logger.debug(len(response))

for item in response:
    logger.debug(item)

"""
## Chaining

We show here how to use it as part of an [agent](/docs/tutorials/agents). We use the Ollama Functions Agent, so we will need to setup and install the required dependencies for that. We will also use [LangSmith Hub](https://smith.langchain.com/hub) to pull the prompt from, so we will need to install that.
"""
logger.info("## Chaining")

# !pip install --upgrade --quiet langchain langchain-ollama langchainhub langchain-community


instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = ChatOllama(model="llama3.2")
you_tool = YouSearchTool(api_wrapper=YouSearchAPIWrapper(num_web_results=1))
tools = [you_tool]
agent = create_ollama_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "What is the weather in NY today?"})

logger.info("\n\n[DONE]", bright=True)