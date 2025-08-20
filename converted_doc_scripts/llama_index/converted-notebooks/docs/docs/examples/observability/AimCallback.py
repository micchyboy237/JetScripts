from jet.logger import CustomLogger
from llama_index.callbacks.aim import AimCallback
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.core.callbacks import CallbackManager
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/AimCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Aim Callback

Aim is an easy-to-use & supercharged open-source AI metadata tracker it logs all your AI metadata (experiments, prompts, etc) enables a UI to compare & observe them and SDK to query them programmatically. For more please see the [Github page](https://github.com/aimhubio/aim).

In this demo, we show the capabilities of Aim for logging events while running queries within LlamaIndex. We use the AimCallback to store the outputs and showing how to explore them using Aim Text Explorer.


**NOTE**: This is a beta feature. The usage within different classes and the API interface for the CallbackManager and AimCallback may change!

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Aim Callback")

# %pip install llama-index-callbacks-aim

# !pip install llama-index


"""
Let's read the documents using `SimpleDirectoryReader` from 'examples/data/paul_graham'.

#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

"""
Now lets initialize an AimCallback instance, and add it to the list of callback managers.
"""
logger.info("Now lets initialize an AimCallback instance, and add it to the list of callback managers.")

aim_callback = AimCallback(repo="./")
callback_manager = CallbackManager([aim_callback])

"""
In this snippet, we initialize a callback manager.
Next, we create an instance of `SummaryIndex` class, by passing in the document reader and callback. After which we create a query engine which we will use to run queries on the index and retrieve relevant results.
"""
logger.info("In this snippet, we initialize a callback manager.")

index = SummaryIndex.from_documents(docs, callback_manager=callback_manager)
query_engine = index.as_query_engine()

"""
Finally let's ask a question to the LM based on our provided document
"""
logger.info("Finally let's ask a question to the LM based on our provided document")

response = query_engine.query("What did the author do growing up?")

"""
The callback manager will log the `CBEventType.LLM` type of events as an Aim.Text, and we can explore the LM given prompt and the output in the Text Explorer. By first doing `aim up` and navigating by the given url.
"""
logger.info("The callback manager will log the `CBEventType.LLM` type of events as an Aim.Text, and we can explore the LM given prompt and the output in the Text Explorer. By first doing `aim up` and navigating by the given url.")

logger.info("\n\n[DONE]", bright=True)