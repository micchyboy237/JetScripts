from infinopy import InfinoClient
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.chains.summarize import load_summarize_chain
from langchain_community.callbacks.infino_callback import InfinoCallbackHandler
from langchain_community.document_loaders import WebBaseLoader
import datetime as dt
import json
import matplotlib.dates as md
import matplotlib.pyplot as plt
import os
import shutil
import time


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
# Infino

>[Infino](https://github.com/infinohq/infino) is a scalable telemetry store designed for logs, metrics, and traces. Infino can function as a standalone observability solution or as the storage layer in your observability stack.

This example shows how one can track the following while calling Ollama and ChatOllama models via `LangChain` and [Infino](https://github.com/infinohq/infino):

* prompt input
* response from `ChatGPT` or any other `LangChain` model
* latency
* errors
* number of tokens consumed

## Initializing
"""
logger.info("# Infino")

# %pip install --upgrade --quiet  infinopy
# %pip install --upgrade --quiet  matplotlib
# %pip install --upgrade --quiet  tiktoken
# %pip install --upgrade --quiet  langchain langchain-ollama langchain-community
# %pip install --upgrade --quiet  beautifulsoup4




"""
## Start Infino server, initialize the Infino client
"""
logger.info("## Start Infino server, initialize the Infino client")

# !docker run --rm --detach --name infino-example -p 3000:3000 infinohq/infino:latest

client = InfinoClient()

"""
## Read the questions dataset
"""
logger.info("## Read the questions dataset")

data = """In what country is Normandy located?
When were the Normans in Normandy?
From which countries did the Norse originate?
Who was the Norse leader?
What century did the Normans first gain their separate identity?
Who gave their name to Normandy in the 1000's and 1100's
What is France a region of?
Who did King Charles III swear fealty to?
When did the Frankish identity emerge?
Who was the duke in the battle of Hastings?
Who ruled the duchy of Normandy
What religion were the Normans
What type of major impact did the Norman dynasty have on modern Europe?
Who was famed for their Christian spirit?
Who assimilted the Roman language?
Who ruled the country of Normandy?
What principality did William the conqueror found?
What is the original meaning of the word Norman?
When was the Latin version of the word Norman first recorded?
What name comes from the English words Normans/Normanz?"""

questions = data.split("\n")

"""
## Example 1: LangChain Ollama Q&A; Publish metrics and logs to Infino
"""
logger.info("## Example 1: LangChain Ollama Q&A; Publish metrics and logs to Infino")

handler = InfinoCallbackHandler(
    model_id="test_ollama", model_version="0.1", verbose=False
)

llm = Ollama(temperature=0.1)

num_questions = 10

questions = questions[0:num_questions]
for question in questions:
    logger.debug(question)

    llm_result = llm.generate([question], callbacks=[handler])
    logger.debug(llm_result)

"""
## Create Metric Charts

We now use matplotlib to create graphs of latency, errors and tokens consumed.
"""
logger.info("## Create Metric Charts")

def plot(data, title):
    data = json.loads(data)

    timestamps = [item["time"] for item in data]
    dates = [dt.datetime.fromtimestamp(ts) for ts in timestamps]
    y = [item["value"] for item in data]

    plt.rcParams["figure.figsize"] = [6, 4]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dates, y)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)

    plt.show()

response = client.search_ts("__name__", "latency", 0, int(time.time()))
plot(response.text, "Latency")

response = client.search_ts("__name__", "error", 0, int(time.time()))
plot(response.text, "Errors")

response = client.search_ts("__name__", "prompt_tokens", 0, int(time.time()))
plot(response.text, "Prompt Tokens")

response = client.search_ts("__name__", "completion_tokens", 0, int(time.time()))
plot(response.text, "Completion Tokens")

response = client.search_ts("__name__", "total_tokens", 0, int(time.time()))
plot(response.text, "Total Tokens")

"""
## Full text query on prompt or prompt outputs.
"""
logger.info("## Full text query on prompt or prompt outputs.")

query = "normandy"
response = client.search_log(query, 0, int(time.time()))
logger.debug("Results for", query, ":", response.text)

logger.debug("===")

query = "king charles III"
response = client.search_log("king charles III", 0, int(time.time()))
logger.debug("Results for", query, ":", response.text)

"""
# Example 2: Summarize a piece of text using ChatOllama
"""
logger.info("# Example 2: Summarize a piece of text using ChatOllama")


handler = InfinoCallbackHandler(
    model_id="test_chatollama", model_version="0.1", verbose=False
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://medium.com/lyft-engineering/lyftlearn-ml-model-training-infrastructure-built-on-kubernetes-aef8218842bb",
    "https://blog.langchain.dev/week-of-10-2-langchain-release-notes/",
]

for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()

    llm = ChatOllama(model="llama3.2")
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)

    chain.run(docs)

"""
## Create Metric Charts
"""
logger.info("## Create Metric Charts")

response = client.search_ts("__name__", "latency", 0, int(time.time()))
plot(response.text, "Latency")

response = client.search_ts("__name__", "error", 0, int(time.time()))
plot(response.text, "Errors")

response = client.search_ts("__name__", "prompt_tokens", 0, int(time.time()))
plot(response.text, "Prompt Tokens")

response = client.search_ts("__name__", "completion_tokens", 0, int(time.time()))
plot(response.text, "Completion Tokens")



query = "machine learning"
response = client.search_log(query, 0, int(time.time()))


logger.debug("===")



# !docker rm -f infino-example

logger.info("\n\n[DONE]", bright=True)