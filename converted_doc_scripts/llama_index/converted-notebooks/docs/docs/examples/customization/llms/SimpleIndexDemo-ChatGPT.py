from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts.chat_prompts import CHAT_REFINE_PROMPT
from llama_index.core.prompts.default_prompts import DEFAULT_REFINE_PROMPT
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/llms/SimpleIndexDemo-ChatGPT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ChatGPT

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# ChatGPT")

# %pip install llama-index-llms-ollama

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.llm = llm
Settings.chunk_size = 512

index = VectorStoreIndex.from_documents(documents)

"""
#### Query Index

By default, with the help of langchain's PromptSelector abstraction, we use 
a modified refine prompt tailored for ChatGPT-use if the ChatGPT model is used.
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(
    similarity_top_k=3,
    streaming=True,
)
response = query_engine.query(
    "What did the author do growing up?",
)

response.print_response_stream()

query_engine = index.as_query_engine(
    similarity_top_k=5,
    streaming=True,
)
response = query_engine.query(
    "What did the author do during his time at RISD?",
)

response.print_response_stream()

"""
**Refine Prompt**: Here is the chat refine prompt
"""


dict(CHAT_REFINE_PROMPT.prompt)

"""
#### Query Index (Using the standard Refine Prompt)

If we use the "standard" refine prompt (where the prompt is one text template instead of multiple messages), we find that the results over ChatGPT are worse.
"""
logger.info("#### Query Index (Using the standard Refine Prompt)")


query_engine = index.as_query_engine(
    refine_template=DEFAULT_REFINE_PROMPT,
    similarity_top_k=5,
    streaming=True,
)
response = query_engine.query(
    "What did the author do during his time at RISD?",
)

response.print_response_stream()

logger.info("\n\n[DONE]", bright=True)