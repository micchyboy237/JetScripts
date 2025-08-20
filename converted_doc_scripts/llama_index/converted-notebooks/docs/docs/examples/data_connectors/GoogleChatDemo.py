from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SummaryIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.google import GoogleChatReader
import datetime
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Google Chat Reader Test

Demonstrates our Google Chat data connector.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google Chat Reader Test")

# %pip install llama-index llama-index-readers-google

"""
This loader takes in IDs of Google Chat spaces or messages and parses the chat history into `Document`s. The space/message ID can be found in the URL, as shown below:

- mail.google.com/chat/u/0/#chat/space/**<CHAT_ID>**

Before using this loader, you need to create a Google Cloud Platform (GCP) project with a Google Workspace account. Then, you need to authorize the app with user credentials. Follow the prerequisites and steps 1 and 2 of [this guide](https://developers.google.com/workspace/chat/authenticate-authorize-chat-user). After downloading the client secret JSON file, rename it as **`credentials.json`** and save it into your project folder.

This example parses a chat between two users. They first discuss math homework, then they plan a trip to San Francisco in a thread. At the end, they discuss finishing an essay. See the full thread [here](https://pastebin.com/FrYscMAa).

## Basic Usage

The example below loads the entire chat history into a `SummaryIndex`.
"""
logger.info("## Basic Usage")


space_ids = [
    "AAAAtTPwdzg"
]  # The Google account you authenticated with must have access to this space
reader = GoogleChatReader()
docs = reader.load_data(space_names=space_ids)

index = SummaryIndex.from_documents(docs)

query_engine = index.as_query_engine()
response = query_engine.query("What was the overall conversation about?")


display(Markdown(f"{response}"))

"""
## Filtering and Ordering

### Ordering
You can order the chat history by ascending or descending order.
"""
logger.info("## Filtering and Ordering")

docs = reader.load_data(space_names=space_ids, order_asc=False)

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query(
    "List the things that the users discussed in the order they were discussed in. Make the list short."
)
display(Markdown(f"{response}"))

"""
Even though the messages were retrieved in reverse order, the list is still in the correct order because messages have a timestamp in their metadata.

### Message Limiting
Messages can be limited to a certain number using the `num_messages` parameter. However, the number of messages that are loaded may not be exactly this number. If `order_asc` is True, then takes the first `num_messages` messages within the given time frame. If `order_desc` is True, then takes the last `num_messages` messages within the time frame.
"""
logger.info("### Message Limiting")

docs = reader.load_data(
    space_names=space_ids, num_messages=10
)  # in ascending order, only contains messages about math HW

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What was discussed in this conversation?")
display(Markdown(f"{response}"))

"""
Notice that the summary is only about the first 10 messages, which only involves help on the math homework. Below is an example of retrieving the last 16 messages, which only involves the essay. The "cost of a trip" refers to a reply in the SF trip thread that was made during the discussion of the essay.
"""
logger.info("Notice that the summary is only about the first 10 messages, which only involves help on the math homework. Below is an example of retrieving the last 16 messages, which only involves the essay. The "cost of a trip" refers to a reply in the SF trip thread that was made during the discussion of the essay.")

docs = reader.load_data(
    space_names=space_ids, num_messages=16, order_asc=False
)  # in descending order, only contains messages about essay

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What was discussed in this conversation?")
display(Markdown(f"{response}"))

"""
### Time Frame

A `before` and `after` time frame can also be specified. These parameters take in `datetime` objects.
"""
logger.info("### Time Frame")


date1 = datetime.datetime.fromisoformat(
    "2024-06-25 14:27:00-07:00"
)  # when they start talking about trip
docs = reader.load_data(space_names=space_ids, before=date1)

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What was discussed in this conversation?"
)  # should only be about math HW
display(Markdown(f"{response}"))

date2 = datetime.datetime.fromisoformat(
    "2024-06-25 14:51:00-07:00"
)  # when they start talking about essay
docs = reader.load_data(space_names=space_ids, after=date2)

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What was discussed in this conversation?"
)  # should only be about essay + cost of trip (in thread)
display(Markdown(f"{response}"))

docs = reader.load_data(space_names=space_ids, after=date1, before=date2)

index = SummaryIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What was discussed in this conversation?"
)  # should only be about trip
display(Markdown(f"{response}"))

logger.info("\n\n[DONE]", bright=True)