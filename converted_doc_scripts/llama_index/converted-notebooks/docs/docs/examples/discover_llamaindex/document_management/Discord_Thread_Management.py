from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json
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
# Discord Thread Management

This notebook walks through the process of managing documents that come from ever-updating data sources.

In this example, we have a directory where the #issues-and-help channel on the LlamaIndex discord is dumped periodically. We want to ensure our index always has the latest data, without duplicating any messages.

## Indexing discord data

Discord data is dumped as sequential messages. Every message has useful information such as timestamps, authors, and links to parent messages if the message is part of a thread.

The help channel on our discord commonly uses threads when solving issues, so we will group all the messages into threads, and index each thread as it's own document.

First, let's explore the data we are working with.
"""
logger.info("# Discord Thread Management")


logger.debug(os.listdir("./discord_dumps"))

"""
As you can see, we have two dumps from two different dates. Let's pretend we only have the older dump to start with, and we want to make an index from that data.

First, let's explore the data a bit
"""
logger.info("As you can see, we have two dumps from two different dates. Let's pretend we only have the older dump to start with, and we want to make an index from that data.")


with open("./discord_dumps/help_channel_dump_05_25_23.json", "r") as f:
    data = json.load(f)
logger.debug("JSON keys: ", data.keys(), "\n")
logger.debug("Message Count: ", len(data["messages"]), "\n")
logger.debug("Sample Message Keys: ", data["messages"][0].keys(), "\n")
logger.debug("First Message: ", data["messages"][0]["content"], "\n")
logger.debug("Last Message: ", data["messages"][-1]["content"])

"""
Conviently, I have provided a script that will group these messages into threads. You can see the `group_conversations.py` script for more details. The output file will be a json list where each item in the list is a discord thread.
"""
logger.info("Conviently, I have provided a script that will group these messages into threads. You can see the `group_conversations.py` script for more details. The output file will be a json list where each item in the list is a discord thread.")

# !python ./group_conversations.py ./discord_dumps/help_channel_dump_05_25_23.json

with open("conversation_docs.json", "r") as f:
    threads = json.load(f)
logger.debug("Thread keys: ", threads[0].keys(), "\n")
logger.debug(threads[0]["metadata"], "\n")
logger.debug(threads[0]["thread"], "\n")

"""
Now, we have a list of threads, that we can transform into documents and index!

## Create the initial index
"""
logger.info("## Create the initial index")


documents = []
for thread in threads:
    thread_text = thread["thread"]
    thread_id = thread["metadata"]["id"]
    timestamp = thread["metadata"]["timestamp"]
    documents.append(
        Document(text=thread_text, id_=thread_id, metadata={"date": timestamp})
    )


index = VectorStoreIndex.from_documents(documents)

"""
Let's double check what documents the index has actually ingested
"""
logger.info("Let's double check what documents the index has actually ingested")

logger.debug("ref_docs ingested: ", len(index.ref_doc_info))
logger.debug("number of input documents: ", len(documents))

"""
So far so good. Let's also check a specific thread to make sure the metadata worked, as well as checking how many nodes it was broken into
"""
logger.info("So far so good. Let's also check a specific thread to make sure the metadata worked, as well as checking how many nodes it was broken into")

thread_id = threads[0]["metadata"]["id"]
logger.debug(index.ref_doc_info[thread_id])

"""
Perfect! Our thread is rather short, so it was directly chunked into a single node. Furthermore, we can see the date field was set correctly.

Next, let's backup our index so that we don't have to waste tokens indexing again.
"""
logger.info("Perfect! Our thread is rather short, so it was directly chunked into a single node. Furthermore, we can see the date field was set correctly.")

index.storage_context.persist(persist_dir="./storage")


index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

logger.debug("Double check ref_docs ingested: ", len(index.ref_doc_info))

"""
## Refresh the index with new data!

Now, suddenly we remember we have that new dump of discord messages! Rather than rebuilding the entire index from scratch, we can index only the new documents using the `refresh()` function.

Since we manually set the `doc_id` of each index, LlamaIndex can compare incoming documents with the same `doc_id` to confirm a) if the `doc_id` has actually been ingested and b) if the content as changed

The refresh function will return a boolean array, indicating which documents in the input were refreshed or inserted. We can use this to confirm that only the new discord threads are inserted!

When a documents content has changed, the `update()` function is called, which removes and re-inserts the document from the index.
"""
logger.info("## Refresh the index with new data!")


with open("./discord_dumps/help_channel_dump_06_02_23.json", "r") as f:
    data = json.load(f)
logger.debug("JSON keys: ", data.keys(), "\n")
logger.debug("Message Count: ", len(data["messages"]), "\n")
logger.debug("Sample Message Keys: ", data["messages"][0].keys(), "\n")
logger.debug("First Message: ", data["messages"][0]["content"], "\n")
logger.debug("Last Message: ", data["messages"][-1]["content"])

"""
As we can see, the first message is the same as the orignal dump. But now we have ~200 more messages, and the last message is clearly new! `refresh()` will make updating our index easy.

First, let's create our new threads/documents
"""
logger.info("As we can see, the first message is the same as the orignal dump. But now we have ~200 more messages, and the last message is clearly new! `refresh()` will make updating our index easy.")

# !python ./group_conversations.py ./discord_dumps/help_channel_dump_06_02_23.json

with open("conversation_docs.json", "r") as f:
    threads = json.load(f)
logger.debug("Thread keys: ", threads[0].keys(), "\n")
logger.debug(threads[0]["metadata"], "\n")
logger.debug(threads[0]["thread"], "\n")

new_documents = []
for thread in threads:
    thread_text = thread["thread"]
    thread_id = thread["metadata"]["id"]
    timestamp = thread["metadata"]["timestamp"]
    new_documents.append(
        Document(text=thread_text, id_=thread_id, metadata={"date": timestamp})
    )

logger.debug("Number of new documents: ", len(new_documents) - len(documents))

refreshed_docs = index.refresh(
    new_documents,
    update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
)

"""
By default, if a document's content has changed and it is updated, we can pass an extra flag to `delete_from_docstore`. This flag is `False` by default because indexes can share the docstore. But since we only have one index, removing from the docstore is fine here.

If we kept the option as `False`, the document information would still be removed from the `index_struct`, which effectively makes that document invisibile to the index.
"""
logger.info("By default, if a document's content has changed and it is updated, we can pass an extra flag to `delete_from_docstore`. This flag is `False` by default because indexes can share the docstore. But since we only have one index, removing from the docstore is fine here.")

logger.debug("Number of newly inserted/refreshed docs: ", sum(refreshed_docs))

"""
Interesting, we have 13 new documents, but 15 documents were refreshed. Did someone edit their message? Add more text to a thread? Let's find out
"""
logger.info("Interesting, we have 13 new documents, but 15 documents were refreshed. Did someone edit their message? Add more text to a thread? Let's find out")

logger.debug(refreshed_docs[-25:])

new_documents[-21]

documents[-8]

"""
Nice! The newer documents contained threads that had more messages. As you can see, `refresh()` was able to detect this and automatically replaced the older thread with the updated text.
"""
logger.info("Nice! The newer documents contained threads that had more messages. As you can see, `refresh()` was able to detect this and automatically replaced the older thread with the updated text.")

logger.info("\n\n[DONE]", bright=True)