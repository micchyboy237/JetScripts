from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/chat_engine/chat_engine_best.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chat Engine - Best Mode

The default chat engine mode is "best", which uses the "openai" mode if you are using an OllamaFunctionCallingAdapter model that supports the latest function calling API, otherwise uses the "react" mode

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chat Engine - Best Mode")

# %pip install llama-index-llms-anthropic
# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Get started in 5 lines of code

Load data and build index
"""
logger.info("### Get started in 5 lines of code")


llm = OllamaFunctionCallingAdapter(model="llama3.2")
data = SimpleDirectoryReader(
    input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
index = VectorStoreIndex.from_documents(data)

"""
Configure chat engine
"""
logger.info("Configure chat engine")

chat_engine = index.as_chat_engine(chat_mode="best", llm=llm, verbose=True)

"""
Chat with your data
"""
logger.info("Chat with your data")

response = chat_engine.chat(
    "What are the first programs Paul Graham tried writing?"
)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
