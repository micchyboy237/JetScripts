from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
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
# Google Jobs

This notebook goes over how to use the Google Jobs Tool to fetch current Job postings.

First, you need to sign up for an `SerpApi key` key at: https://serpapi.com/users/sign_up and get your api key here: https://serpapi.com/manage-api-key.

Then you must install `google-search-results` with the command:
    `pip install google-search-results`

Then you will need to set the environment variable `SERPAPI_API_KEY` to your `SerpApi key`

If you are using conda environment, you can set up using the following commands in kernal:

conda activate [your env name]
conda env confiv vars SERPAPI_API_KEY='[your serp api key]'

## Use the Tool
"""
logger.info("# Google Jobs")

# %pip install --upgrade --quiet google-search-results langchain-community


os.environ["SERPAPI_API_KEY"] = ""


tool = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper())

tool.run("Can I get an entry level job posting related to physics")

"""
# Use the tool within a ReAct agent

In order to create an agent that uses the Google Jobs tool install Langgraph
"""
logger.info("# Use the tool within a ReAct agent")

# %pip install --upgrade --quiet langgraph langchain-ollama

"""
and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.
"""
logger.info("and use the `create_react_agent` functionality to initialize a ReAct agent. You will also need to set up your OPEN_API_KEY (visit https://platform.ollama.com) in order to access Ollama's chat models.")


# os.environ["OPENAI_API_KEY"] = ""
os.environ["SERP_API_KEY"] = ""


tools = load_tools(["google-jobs"])


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

events = agent.stream(
    {"messages": [("user", "Give me an entry level job posting related to physics?")]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)