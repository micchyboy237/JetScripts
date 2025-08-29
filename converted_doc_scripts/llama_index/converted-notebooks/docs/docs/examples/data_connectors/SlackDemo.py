from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import SummaryIndex
from llama_index.readers.slack import SlackReader
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/SlackDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Slack Reader
Demonstrates our Slack data connector

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Slack Reader")

# %pip install llama-index-readers-slack

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
Load data using Channel IDs
"""
logger.info("Load data using Channel IDs")

slack_token = os.getenv("SLACK_BOT_TOKEN")
channel_ids = ["<channel_id>"]
documents = SlackReader(slack_token=slack_token).load_data(
    channel_ids=channel_ids
)

"""
Load data using Channel Names/Regex Patterns
"""
logger.info("Load data using Channel Names/Regex Patterns")

slack_token = os.getenv("SLACK_BOT_TOKEN")
channel_patterns = ["<channel_name>", "<regex_pattern>"]
slack_reader = SlackReader(slack_token=slack_token)
channel_ids = slack_reader.get_channel_ids(channel_patterns=channel_patterns)
documents = slack_reader.load_data(channel_ids=channel_ids)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("<query_text>")

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)