from jet.logger import CustomLogger
from llama_index.callbacks.openinference import OpenInferenceCallbackHandler
from llama_index.callbacks.openinference.base import (
as_dataframe,
QueryData,
NodeData,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.web import SimpleWebPageReader
from pathlib import Path
from tqdm import tqdm
from typing import List, Union
import hashlib
import json
import llama_index.core
import os
import pandas as pd
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/OpenInferenceCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OpenInference Callback Handler + Arize Phoenix

[OpenInference](https://github.com/Arize-ai/open-inference-spec) is an open standard for capturing and storing AI model inferences. It enables production LLMapp servers to seamlessly integrate with LLM observability solutions such as [Arize](https://arize.com/) and [Phoenix](https://github.com/Arize-ai/phoenix).

The `OpenInferenceCallbackHandler` saves data from LLM applications for downstream analysis and debugging. In particular, it saves the following data in columnar format:

- query IDs
- query text
- query embeddings
- scores (e.g., cosine similarity)
- retrieved document IDs

This tutorial demonstrates the callback handler's use for both in-notebook experimentation and lightweight production logging.

⚠️ The `OpenInferenceCallbackHandler` is in beta and its APIs are subject to change.

ℹ️ If you find that your particular query engine or use-case is not supported, open an issue on [GitHub](https://github.com/Arize-ai/open-inference-spec/issues).

## Configue OllamaFunctionCalling API key
"""
logger.info("# OpenInference Callback Handler + Arize Phoenix")

# from getpass import getpass

# if os.getenv("OPENAI_API_KEY") is None:
#     os.environ["OPENAI_API_KEY"] = getpass(
        "Paste your OllamaFunctionCalling key from:"
        " https://platform.openai.com/account/api-keys\n"
    )
# assert os.getenv("OPENAI_API_KEY", "").startswith(
    "sk-"
), "This doesn't look like a valid OllamaFunctionCalling API key"
logger.debug("OllamaFunctionCalling API key configured")

"""
## Install Dependencies and Import Libraries

Install notebook dependencies.
"""
logger.info("## Install Dependencies and Import Libraries")

# %pip install -q html2text llama-index pandas pyarrow tqdm
# %pip install -q llama-index-readers-web
# %pip install -q llama-index-callbacks-openinference

"""
Import libraries.
"""
logger.info("Import libraries.")



"""
## Load and Parse Documents

Load documents from Paul Graham's essay "What I Worked On".
"""
logger.info("## Load and Parse Documents")

documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    ]
)
logger.debug(documents[0].text)

"""
Parse the document into nodes. Display the first node's text.
"""
logger.info("Parse the document into nodes. Display the first node's text.")

parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
logger.debug(nodes[0].text)

"""
## Access Data as a Pandas Dataframe

When experimenting with chatbots and LLMapps in a notebook, it's often useful to run your chatbot against a small collection of user queries and collect and analyze the data for iterative improvement. The `OpenInferenceCallbackHandler` stores your data in columnar format and provides convenient access to the data as a pandas dataframe.

Instantiate the OpenInference callback handler.
"""
logger.info("## Access Data as a Pandas Dataframe")

callback_handler = OpenInferenceCallbackHandler()
callback_manager = CallbackManager([callback_handler])
llama_index.core.Settings.callback_manager = callback_manager

"""
Build the index and instantiate a query engine.
"""
logger.info("Build the index and instantiate a query engine.")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

"""
Run your query engine across a collection of queries.
"""
logger.info("Run your query engine across a collection of queries.")

max_characters_per_line = 80
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in queries:
    response = query_engine.query(query)
    logger.debug("Query")
    logger.debug("=====")
    logger.debug(textwrap.fill(query, max_characters_per_line))
    logger.debug()
    logger.debug("Response")
    logger.debug("========")
    logger.debug(textwrap.fill(str(response), max_characters_per_line))
    logger.debug()

"""
The data from your query engine runs can be accessed as a pandas dataframe for analysis and iterative improvement.
"""
logger.info("The data from your query engine runs can be accessed as a pandas dataframe for analysis and iterative improvement.")

query_data_buffer = callback_handler.flush_query_data_buffer()
query_dataframe = as_dataframe(query_data_buffer)
query_dataframe

"""
The dataframe column names conform to the OpenInference spec, which specifies the category, data type, and intent of each column.

## Log Production Data

In a production setting, LlamaIndex application maintainers can log the data generated by their system by implementing and passing a custom `callback` to `OpenInferenceCallbackHandler`. The callback is of type `Callable[List[QueryData]]` that accepts a buffer of query data from the `OpenInferenceCallbackHandler`, persists the data (e.g., by uploading to cloud storage or sending to a data ingestion service), and flushes the buffer after data is persisted. A reference implementation is included below that periodically writes data in OpenInference format to local Parquet files when the buffer exceeds a certain size.
"""
logger.info("## Log Production Data")

class ParquetCallback:
    def __init__(
        self, data_path: Union[str, Path], max_buffer_length: int = 1000
    ):
        self._data_path = Path(data_path)
        self._data_path.mkdir(parents=True, exist_ok=False)
        self._max_buffer_length = max_buffer_length
        self._batch_index = 0

    def __call__(
        self,
        query_data_buffer: List[QueryData],
        node_data_buffer: List[NodeData],
    ) -> None:
        if len(query_data_buffer) >= self._max_buffer_length:
            query_dataframe = as_dataframe(query_data_buffer)
            file_path = self._data_path / f"log-{self._batch_index}.parquet"
            query_dataframe.to_parquet(file_path)
            self._batch_index += 1
            query_data_buffer.clear()  # ⚠️ clear the buffer or it will keep growing forever!
            node_data_buffer.clear()  # didn't log node_data_buffer, but still need to clear it

"""
⚠️ In a production setting, it's important to clear the buffer, otherwise, the callback handler will indefinitely accumulate data in memory and eventually cause your system to crash.

Attach the Parquet writer to your callback and re-run the query engine. The data will be saved to disk.
"""
logger.info("Attach the Parquet writer to your callback and re-run the query engine. The data will be saved to disk.")

data_path = "data"
parquet_writer = ParquetCallback(
    data_path=data_path,
    max_buffer_length=1,
)
callback_handler = OpenInferenceCallbackHandler(callback=parquet_writer)
callback_manager = CallbackManager([callback_handler])
llama_index.core.Settings.callback_manager = callback_manager

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

for query in tqdm(queries):
    query_engine.query(query)

"""
Load and display saved Parquet data from disk to verify that the logger is working.
"""
logger.info("Load and display saved Parquet data from disk to verify that the logger is working.")

query_dataframes = []
for file_name in os.listdir(data_path):
    file_path = os.path.join(data_path, file_name)
    query_dataframes.append(pd.read_parquet(file_path))
query_dataframe = pd.concat(query_dataframes)
query_dataframe

logger.info("\n\n[DONE]", bright=True)