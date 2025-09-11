from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.agents import Tool
from langchain.docstore import InMemoryDocstore
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT
import faiss
import os
import shutil


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
# AutoGPT

Implementation of https://github.com/Significant-Gravitas/Auto-GPT but with LangChain primitives (LLMs, PromptTemplates, VectorStores, Embeddings, Tools)

## Set up tools

We'll set up an AutoGPT with a search tool, and write-file tool, and a read-file tool
"""
logger.info("# AutoGPT")


search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

"""
## Set up memory

The memory here is used for the agents intermediate steps
"""
logger.info("## Set up memory")


embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

"""
## Setup model and AutoGPT

Initialize everything! We will use ChatOllama model
"""
logger.info("## Setup model and AutoGPT")


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOllama(model="llama3.2"),
    memory=vectorstore.as_retriever(),
)
agent.chain.verbose = True

"""
## Run an example

Here we will make it write a weather report for SF
"""
logger.info("## Run an example")

agent.run(["write a weather report for SF today"])

"""
## Chat History Memory

In addition to the memory that holds the agent immediate steps, we also have a chat history memory. By default, the agent will use 'ChatMessageHistory' and it can be changed. This is useful when you want to use a different type of memory for example 'FileChatHistoryMemory'
"""
logger.info("## Chat History Memory")


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOllama(model="llama3.2"),
    memory=vectorstore.as_retriever(),
    chat_history_memory=FileChatMessageHistory("chat_history.txt"),
)

"""

"""

logger.info("\n\n[DONE]", bright=True)
