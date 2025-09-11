from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
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
# Google Finance

This notebook goes over how to use the Google Finance Tool to get information from the Google Finance page.

To get an SerpApi key key, sign up at: https://serpapi.com/users/sign_up.

To use the tool with Langchain install following packages
"""
logger.info("# Google Finance")

# %pip install --upgrade --quiet google-search-results langchain-community

"""
Then set the environment variable SERPAPI_API_KEY to your SerpApi key or pass the key in as a argument to the wrapper serp_.
"""
logger.info("Then set the environment variable SERPAPI_API_KEY to your SerpApi key or pass the key in as a argument to the wrapper serp_.")


os.environ["SERPAPI_API_KEY"] = ""


tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())

tool.run("Google")

"""
In order to create an agent that uses the Google Finance tool install Langgraph
"""
logger.info("In order to create an agent that uses the Google Finance tool install Langgraph")

# %pip install --upgrade --quiet langgraph langchain-ollama

"""
and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.
"""
logger.info("and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.")


# os.environ["OPENAI_API_KEY"] = ""
os.environ["SERP_API_KEY"] = ""


llm = init_chat_model("llama3.2", model_provider="ollama")


tools = load_tools(["google-scholar", "google-finance"], llm=llm)


agent = create_react_agent(llm, tools)

events = agent.stream(
    {"messages": [("user", "What is Google's stock?")]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)