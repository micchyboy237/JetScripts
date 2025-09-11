from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.adapters.ollama import convert_messages_for_finetuning
from langchain_community.chat_loaders.imessage import IMessageChatLoader
from langchain_community.chat_loaders.utils import (
map_ai_messages,
merge_chat_runs,
)
from langchain_core.chat_sessions import ChatSession
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import json
import ollama
import os
import requests
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
# iMessage

This notebook shows how to use the iMessage chat loader. This class helps convert iMessage conversations to LangChain chat messages.

On MacOS, iMessage stores conversations in a sqlite database at `~/Library/Messages/chat.db` (at least for macOS Ventura 13.4). 
The `IMessageChatLoader` loads from this database file. 

1. Create the `IMessageChatLoader` with the file path pointed to `chat.db` database you'd like to process.
2. Call `loader.load()` (or `loader.lazy_load()`) to perform the conversion. Optionally use `merge_chat_runs` to combine message from the same sender in sequence, and/or `map_ai_messages` to convert messages from the specified sender to the "AIMessage" class.

## 1. Access Chat DB

It's likely that your terminal is denied access to `~/Library/Messages`. To use this class, you can copy the DB to an accessible directory (e.g., Documents) and load from there. Alternatively (and not recommended), you can grant full disk access for your terminal emulator in System Settings > Security and Privacy > Full Disk Access.

We have created an example database you can use at [this linked drive file](https://drive.google.com/file/d/1NebNKqTA2NXApCmeH6mu0unJD2tANZzo/view?usp=sharing).
"""
logger.info("# iMessage")



def download_drive_file(url: str, output_path: str = "chat.db") -> None:
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url)
    if response.status_code != 200:
        logger.debug("Failed to download the file.")
        return

    with open(output_path, "wb") as file:
        file.write(response.content)
        logger.debug(f"File {output_path} downloaded.")


url = (
    "https://drive.google.com/file/d/1NebNKqTA2NXApCmeH6mu0unJD2tANZzo/view?usp=sharing"
)

download_drive_file(url)

"""
## 2. Create the Chat Loader

Provide the loader with the file path to the zip directory. You can optionally specify the user id that maps to an ai message as well as configure whether to merge message runs.
"""
logger.info("## 2. Create the Chat Loader")


loader = IMessageChatLoader(
    path="./chat.db",
)

"""
## 3. Load messages

The `load()` (or `lazy_load`) methods return a list of "ChatSessions" that currently just contain a list of messages per loaded conversation. All messages are mapped to "HumanMessage" objects to start. 

You can optionally choose to merge message "runs" (consecutive messages from the same sender) and select a sender to represent the "AI". The fine-tuned LLM will learn to generate these AI messages.
"""
logger.info("## 3. Load messages")



raw_messages = loader.lazy_load()
merged_messages = merge_chat_runs(raw_messages)
chat_sessions: List[ChatSession] = list(
    map_ai_messages(merged_messages, sender="Tortoise")
)

chat_sessions[0]["messages"][:3]

"""
## 3. Prepare for fine-tuning

Now it's time to convert our chat  messages to Ollama dictionaries. We can use the `convert_messages_for_finetuning` utility to do so.
"""
logger.info("## 3. Prepare for fine-tuning")


training_data = convert_messages_for_finetuning(chat_sessions)
logger.debug(f"Prepared {len(training_data)} dialogues for training")

"""
## 4. Fine-tune the model

It's time to fine-tune the model. Make sure you have `ollama` installed
# and have set your `OPENAI_API_KEY` appropriately
"""
logger.info("## 4. Fine-tune the model")

# %pip install --upgrade --quiet  langchain-ollama



my_file = BytesIO()
for m in training_data:
    my_file.write((json.dumps({"messages": m}) + "\n").encode("utf-8"))

my_file.seek(0)
training_file = ollama.files.create(file=my_file, purpose="fine-tune")

status = ollama.files.retrieve(training_file.id).status
start_time = time.time()
while status != "processed":
    logger.debug(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = ollama.files.retrieve(training_file.id).status
logger.debug(f"File {training_file.id} ready after {time.time() - start_time:.2f} seconds.")

"""
With the file ready, it's time to kick off a training job.
"""
logger.info("With the file ready, it's time to kick off a training job.")

job = ollama.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="llama3.2",
)

"""
Grab a cup of tea while your model is being prepared. This may take some time!
"""
logger.info("Grab a cup of tea while your model is being prepared. This may take some time!")

status = ollama.fine_tuning.jobs.retrieve(job.id).status
start_time = time.time()
while status != "succeeded":
    logger.debug(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    job = ollama.fine_tuning.jobs.retrieve(job.id)
    status = job.status

logger.debug(job.fine_tuned_model)

"""
## 5. Use in LangChain

You can use the resulting model ID directly the `ChatOllama` model class.
"""
logger.info("## 5. Use in LangChain")


model = ChatOllama(
    model=job.fine_tuned_model,
    temperature=1,
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are speaking to hare."),
        ("human", "{input}"),
    ]
)

chain = prompt | model | StrOutputParser()

for tok in chain.stream({"input": "What's the golden thread?"}):
    logger.debug(tok, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)