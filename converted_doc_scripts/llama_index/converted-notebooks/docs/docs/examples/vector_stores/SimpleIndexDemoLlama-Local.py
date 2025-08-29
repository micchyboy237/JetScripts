from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import logging
import os
import shutil
import sys
import time
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemoLlama-Local.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Local Llama2 + VectorStoreIndex

This notebook walks through the proper setup to use llama-2 with LlamaIndex locally. Note that you need a decent GPU to run this notebook, ideally an A100 with at least 40GB of memory.

Specifically, we look at using a vector store index.

## Setup
"""
logger.info("# Local Llama2 + VectorStoreIndex")

# %pip install llama-index-llms-huggingface
# %pip install llama-index-embeddings-huggingface

# !pip install llama-index ipywidgets

"""
### Set Up

**IMPORTANT**: Please sign in to HF hub with an account that has access to the llama2 models, using `huggingface-cli login` in your console. For more details, please see: https://ai.meta.com/resources/models-and-libraries/llama-downloads/.
"""
logger.info("### Set Up")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))




LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
LLAMA2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
LLAMA2_70B = "meta-llama/Llama-2-70b-hf"
LLAMA2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"

selected_model = LLAMA2_13B_CHAT

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
)


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)


Settings.llm = llm
Settings.embed_model = embed_model

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


index = VectorStoreIndex.from_documents(documents)

"""
## Querying
"""
logger.info("## Querying")

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
### Streaming Support
"""
logger.info("### Streaming Support")


query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")

start_time = time.time()

token_count = 0
for token in response.response_gen:
    logger.debug(token, end="")
    token_count += 1

time_elapsed = time.time() - start_time
tokens_per_second = token_count / time_elapsed

logger.debug(f"\n\nStreamed output at {tokens_per_second} tokens/s")

logger.info("\n\n[DONE]", bright=True)