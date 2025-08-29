from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
import logging
import os
import shutil
import sys
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_camel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HuggingFace LLM - Camel-5b

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# HuggingFace LLM - Camel-5b")

# %pip install llama-index-llms-huggingface

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

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)


llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
)

Settings.chunk_size = 512
Settings.llm = llm

index = VectorStoreIndex.from_documents(documents)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

logger.debug(response)

"""
#### Query Index - Streaming
"""
logger.info("#### Query Index - Streaming")

query_engine = index.as_query_engine(streaming=True)

response_stream = query_engine.query("What did the author do growing up?")

response_stream.print_response_stream()

response = response_stream.get_response()
logger.debug(response)

generated_text = ""
for text in response.response_gen:
    generated_text += text

logger.info("\n\n[DONE]", bright=True)