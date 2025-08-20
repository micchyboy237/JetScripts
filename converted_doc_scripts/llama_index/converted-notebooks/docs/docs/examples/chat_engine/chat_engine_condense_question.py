from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/chat_engine/chat_engine_condense_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chat Engine - Condense Question Mode

Condense question is a simple chat mode built on top of a query engine over your data.

For each chat interaction:
* first generate a standalone question from conversation context and last message, then 
* query the query engine with the condensed question for a response.

This approach is simple, and works for questions directly related to the knowledge base. 
Since it *always* queries the knowledge base, it can have difficulty answering meta questions like "what did I ask you before?"

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chat Engine - Condense Question Mode")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Get started in 5 lines of code

Load data and build index
"""
logger.info("## Get started in 5 lines of code")


data = SimpleDirectoryReader(input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
index = VectorStoreIndex.from_documents(data)

"""
Configure chat engine
"""
logger.info("Configure chat engine")

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

"""
Chat with your data
"""
logger.info("Chat with your data")

response = chat_engine.chat("What did Paul Graham do after YC?")

logger.debug(response)

"""
Ask a follow up question
"""
logger.info("Ask a follow up question")

response = chat_engine.chat("What about after that?")

logger.debug(response)

response = chat_engine.chat("Can you tell me more?")

logger.debug(response)

"""
Reset conversation state
"""
logger.info("Reset conversation state")

chat_engine.reset()

response = chat_engine.chat("What about after that?")

logger.debug(response)

"""
## Streaming Support
"""
logger.info("## Streaming Support")


llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)

data = SimpleDirectoryReader(input_dir="./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(
    chat_mode="condense_question", llm=llm, verbose=True
)

response = chat_engine.stream_chat("What did Paul Graham do after YC?")
for token in response.response_gen:
    logger.debug(token, end="")

logger.info("\n\n[DONE]", bright=True)