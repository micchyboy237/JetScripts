from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.adapters.ollama import convert_messages_for_finetuning
from langchain_community.chat_loaders.facebook_messenger import (
FolderFacebookMessengerChatLoader,
SingleFileFacebookMessengerChatLoader,
)
from langchain_community.chat_loaders.utils import (
map_ai_messages,
merge_chat_runs,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import ollama
import os
import requests
import shutil
import time
import zipfile


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
# Facebook Messenger

This notebook shows how to load data from Facebook into a format you can fine-tune on. The overall steps are:

1. Download your messenger data to disk.
2. Create the Chat Loader and call `loader.load()` (or `loader.lazy_load()`) to perform the conversion.
3. Optionally use `merge_chat_runs` to combine message from the same sender in sequence, and/or `map_ai_messages` to convert messages from the specified sender to the "AIMessage" class. Once you've done this, call `convert_messages_for_finetuning` to prepare your data for fine-tuning.


Once this has been done, you can fine-tune your model. To do so you would complete the following steps:

4. Upload your messages to Ollama and run a fine-tuning job.
6. Use the resulting model in your LangChain app!


Let's begin.


## 1. Download Data

To download your own messenger data, follow the instructions [here](https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/). IMPORTANT - make sure to download them in JSON format (not HTML).

We are hosting an example dump at [this google drive link](https://drive.google.com/file/d/1rh1s1o2i7B-Sk1v9o8KNgivLVGwJ-osV/view?usp=sharing) that we will use in this walkthrough.
"""
logger.info("# Facebook Messenger")




def download_and_unzip(url: str, output_path: str = "file.zip") -> None:
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url)
    if response.status_code != 200:
        logger.debug("Failed to download the file.")
        return

    with open(output_path, "wb") as file:
        file.write(response.content)
        logger.debug(f"File {output_path} downloaded.")

    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall()
        logger.debug(f"File {output_path} has been unzipped.")


url = (
    "https://drive.google.com/file/d/1rh1s1o2i7B-Sk1v9o8KNgivLVGwJ-osV/view?usp=sharing"
)

download_and_unzip(url)

"""
## 2. Create Chat Loader

We have 2 different `FacebookMessengerChatLoader` classes, one for an entire directory of chats, and one to load individual files. We
"""
logger.info("## 2. Create Chat Loader")

directory_path = "./hogwarts"


loader = SingleFileFacebookMessengerChatLoader(
    path="./hogwarts/inbox/HermioneGranger/messages_Hermione_Granger.json",
)

chat_session = loader.load()[0]
chat_session["messages"][:3]

loader = FolderFacebookMessengerChatLoader(
    path="./hogwarts",
)

chat_sessions = loader.load()
len(chat_sessions)

"""
## 3. Prepare for fine-tuning

Calling `load()` returns all the chat messages we could extract as human messages. When conversing with chat bots, conversations typically follow a more strict alternating dialogue pattern relative to real conversations. 

You can choose to merge message "runs" (consecutive messages from the same sender) and select a sender to represent the "AI". The fine-tuned LLM will learn to generate these AI messages.
"""
logger.info("## 3. Prepare for fine-tuning")


merged_sessions = merge_chat_runs(chat_sessions)
alternating_sessions = list(map_ai_messages(merged_sessions, "Harry Potter"))

alternating_sessions[0]["messages"][:3]

"""
#### Now we can convert to Ollama format dictionaries
"""
logger.info("#### Now we can convert to Ollama format dictionaries")


training_data = convert_messages_for_finetuning(alternating_sessions)
logger.debug(f"Prepared {len(training_data)} dialogues for training")

training_data[0][:3]

"""
Ollama currently requires at least 10 training examples for a fine-tuning job, though they recommend between 50-100 for most tasks. Since we only have 9 chat sessions, we can subdivide them (optionally with some overlap) so that each training example is comprised of a portion of a whole conversation.

Facebook chat sessions (1 per person) often span multiple days and conversations,
so the long-range dependencies may not be that important to model anyhow.
"""
logger.info("Ollama currently requires at least 10 training examples for a fine-tuning job, though they recommend between 50-100 for most tasks. Since we only have 9 chat sessions, we can subdivide them (optionally with some overlap) so that each training example is comprised of a portion of a whole conversation.")

chunk_size = 8
overlap = 2

training_examples = [
    conversation_messages[i : i + chunk_size]
    for conversation_messages in training_data
    for i in range(0, len(conversation_messages) - chunk_size + 1, chunk_size - overlap)
]

len(training_examples)

"""
## 4. Fine-tune the model

It's time to fine-tune the model. Make sure you have `ollama` installed
# and have set your `OPENAI_API_KEY` appropriately
"""
logger.info("## 4. Fine-tune the model")

# %pip install --upgrade --quiet  langchain-ollama



my_file = BytesIO()
for m in training_examples:
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
        ("human", "{input}"),
    ]
)

chain = prompt | model | StrOutputParser()

for tok in chain.stream({"input": "What classes are you taking?"}):
    logger.debug(tok, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)