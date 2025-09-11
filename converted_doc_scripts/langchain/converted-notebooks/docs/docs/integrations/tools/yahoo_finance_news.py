from jet.logger import logger
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
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
# Yahoo Finance News

This notebook goes over how to use the `yahoo_finance_news` tool with an agent. 


## Setting up

First, you need to install `yfinance` python package.
"""
logger.info("# Yahoo Finance News")

# %pip install --upgrade --quiet  yfinance

"""
## Example with Chain
"""
logger.info("## Example with Chain")


# os.environ["OPENAI_API_KEY"] = ".."


tools = [YahooFinanceNewsTool()]
agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What happened today with Microsoft stocks?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

input_message = {
    "role": "user",
    "content": "How does Microsoft feels today comparing with Nvidia?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
# How YahooFinanceNewsTool works?
"""
logger.info("# How YahooFinanceNewsTool works?")

tool = YahooFinanceNewsTool()

tool.invoke("NVDA")

res = tool.invoke("AAPL")
logger.debug(res)

logger.info("\n\n[DONE]", bright=True)