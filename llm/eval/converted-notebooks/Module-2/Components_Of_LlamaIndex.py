from jet.llm.utils import display_jet_source_node
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Settings
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Components Of LlamaIndex
#
# In this notebook we will demonstrate building RAG application and customize it using different components of LlamaIndex.
#
# 1. Question Answering
# 2. Summarization.
# 3. ChatEngine.
# 4. Customizing QA System.
# 5. Index as Retriever.
#
# [ChatEngine Documentation](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/ )

# Installation

# !pip install llama-index

# Setup API Key


# os.environ["OPENAI_API_KEY"] = "sk-..."

# Download Data

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

# Load Data


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

# Set LLM and Embedding Model


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0.2)
embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

Settings.llm = llm
Settings.embed_model = embed_model

# Create Nodes


splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

nodes[0]

# Create Index


index = VectorStoreIndex(nodes)  # VectorStoreIndex.from_documents(documents)

# Create QueryEngine

query_engine = index.as_query_engine(similarity_top_k=5)

# Querying

response = query_engine.query("What did Paul Graham do growing up?")

print(response)

print(len(response.source_nodes))

response.source_nodes[0]

# Summarization


summary_index = SummaryIndex(nodes)

query_engine = summary_index.as_query_engine()

summary = query_engine.query("Provide the summary of the document.")
print(summary)

# ChatEngines

# Simple ChatEngine


chat_engine = SimpleChatEngine.from_defaults()

response = chat_engine.chat("Hello")
response

response = chat_engine.chat("What did steve jobs do growing up?")
response

response = chat_engine.chat("And did he visit India?")
response

chat_engine.chat_repl()

# CondenseQuestion ChatEngine

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

response = chat_engine.chat("What did Paul Graham do after YC?")

print(response)

response = chat_engine.chat("What about after that?")

print(response)

response = chat_engine.chat("Can you tell me more?")

print(response)

# Context ChatEngine


memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
    ),
)

response = chat_engine.chat("Hello")
print(response)

response = chat_engine.chat("What did Paul Graham do after YC?")
print(response)

response = chat_engine.chat("What about after that?")
print(response)

response = chat_engine.chat("Can you tell me more?")
print(response)

# CondenseContext ChatEngine


memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)

response = chat_engine.chat("Hello")

print(response)

response = chat_engine.chat("What did Paul Graham do after YC?")

print(response)

response = chat_engine.chat("What about after that?")

print(response)

response = chat_engine.chat("Can you tell me more?")

print(response)

# Customizing RAG Pipeline


splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)

retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

synthesizer = get_response_synthesizer(response_mode="refine")

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)

response = query_engine.query("What did Paul Graham do growing up?")

print(response)

# Index as Retriever

retriever = index.as_retriever(similarity_top_k=3)

retrieved_nodes = retriever.retrieve("What did Paul Graham do growing up?")


for text_node in retrieved_nodes:
    display_jet_source_node(text_node, source_length=500)

logger.info("\n\n[DONE]", bright=True)
