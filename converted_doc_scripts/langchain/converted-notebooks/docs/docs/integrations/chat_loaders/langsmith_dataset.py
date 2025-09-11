from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.adapters.ollama import convert_messages_for_finetuning
from langchain_community.chat_loaders.langsmith import LangSmithDatasetChatLoader
from langsmith.client import Client
import json
import ollama
import os
import requests
import shutil
import time
import uuid


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
# LangSmith Chat Datasets

This notebook demonstrates an easy way to load a LangSmith chat dataset and fine-tune a model on that data.
The process is simple and comprises 3 steps.

1. Create the chat dataset.
2. Use the LangSmithDatasetChatLoader to load examples.
3. Fine-tune your model.

Then you can use the fine-tuned model in your LangChain app.

Before diving in, let's install our prerequisites.

## Prerequisites

Ensure you've installed langchain >= 0.0.311 and have configured your environment with your LangSmith API key.
"""
logger.info("# LangSmith Chat Datasets")

# %pip install --upgrade --quiet  langchain langchain-ollama


uid = uuid.uuid4().hex[:6]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "YOUR API KEY"

"""
## 1. Select a dataset

This notebook fine-tunes a model directly on selecting which runs to fine-tune on. You will often curate these from traced runs. You can learn more about LangSmith datasets in the docs [docs](https://docs.smith.langchain.com/evaluation/concepts#datasets).

For the sake of this tutorial, we will upload an existing dataset here that you can use.
"""
logger.info("## 1. Select a dataset")


client = Client()


url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/integrations/chat_loaders/example_data/langsmith_chat_dataset.json"
response = requests.get(url)
response.raise_for_status()
data = response.json()

dataset_name = f"Extraction Fine-tuning Dataset {uid}"
ds = client.create_dataset(dataset_name=dataset_name, data_type="chat")

_ = client.create_examples(
    inputs=[e["inputs"] for e in data],
    outputs=[e["outputs"] for e in data],
    dataset_id=ds.id,
)

"""
## 2. Prepare Data
Now we can create an instance of LangSmithRunChatLoader and load the chat sessions using its lazy_load() method.
"""
logger.info("## 2. Prepare Data")


loader = LangSmithDatasetChatLoader(dataset_name=dataset_name)

chat_sessions = loader.lazy_load()

"""
#### With the chat sessions loaded, convert them into a format suitable for fine-tuning.
"""
logger.info("#### With the chat sessions loaded, convert them into a format suitable for fine-tuning.")


training_data = convert_messages_for_finetuning(chat_sessions)

"""
## 3. Fine-tune the Model
Now, initiate the fine-tuning process using the Ollama library.
"""
logger.info("## 3. Fine-tune the Model")



my_file = BytesIO()
for dialog in training_data:
    my_file.write((json.dumps({"messages": dialog}) + "\n").encode("utf-8"))

my_file.seek(0)
training_file = ollama.files.create(file=my_file, purpose="fine-tune")

job = ollama.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="llama3.2",
)

status = ollama.fine_tuning.jobs.retrieve(job.id).status
start_time = time.time()
while status != "succeeded":
    logger.debug(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = ollama.fine_tuning.jobs.retrieve(job.id).status

"""
## 4. Use in LangChain

After fine-tuning, use the resulting model ID with the ChatOllama model class in your LangChain app.
"""
logger.info("## 4. Use in LangChain")

job = ollama.fine_tuning.jobs.retrieve(job.id)
model_id = job.fine_tuned_model


model = ChatOllama(
    model=model_id,
    temperature=1,
)

model.invoke("There were three ravens sat on a tree.")

"""
Now you have successfully fine-tuned a model using data from LangSmith LLM runs!
"""
logger.info("Now you have successfully fine-tuned a model using data from LangSmith LLM runs!")

logger.info("\n\n[DONE]", bright=True)