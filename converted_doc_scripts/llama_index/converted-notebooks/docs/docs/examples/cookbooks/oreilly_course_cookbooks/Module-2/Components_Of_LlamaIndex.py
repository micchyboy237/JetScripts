from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Components Of LlamaIndex

In this notebook we will demonstrate building RAG application and customize it using different components of LlamaIndex.

1. Question Answering
2. Summarization.
3. ChatEngine.
4. Customizing QA System.
5. Index as Retriever.

[ChatEngine Documentation](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/ )

## Installation
"""
logger.info("# Components Of LlamaIndex")

# !pip install llama-index

"""
## Setup API Key
"""
logger.info("## Setup API Key")


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Download Data
"""
logger.info("## Download Data")

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

"""
## Load Data
"""
logger.info("## Load Data")


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

"""
## Set LLM and Embedding Model
"""
logger.info("## Set LLM and Embedding Model")


llm = MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.2)
embed_model = MLXEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model

"""
## Create Nodes
"""
logger.info("## Create Nodes")


splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

nodes[0]

"""
## Create Index
"""
logger.info("## Create Index")


index = VectorStoreIndex(nodes)  # VectorStoreIndex.from_documents(documents)

"""
## Create QueryEngine
"""
logger.info("## Create QueryEngine")

query_engine = index.as_query_engine(similarity_top_k=5)

"""
## Querying
"""
logger.info("## Querying")

response = query_engine.query("What did Paul Graham do growing up?")

logger.debug(response)

logger.debug(len(response.source_nodes))

response.source_nodes[0]

"""
## Summarization
"""
logger.info("## Summarization")


summary_index = SummaryIndex(nodes)

query_engine = summary_index.as_query_engine()

summary = query_engine.query("Provide the summary of the document.")
logger.debug(summary)

"""
## ChatEngines

### Simple ChatEngine
"""
logger.info("## ChatEngines")


chat_engine = SimpleChatEngine.from_defaults()

response = chat_engine.chat("Hello")
response

response = chat_engine.chat("What did steve jobs do growing up?")
response

response = chat_engine.chat("And did he visit India?")
response

chat_engine.chat_repl()

"""
### CondenseQuestion ChatEngine
"""
logger.info("### CondenseQuestion ChatEngine")

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

response = chat_engine.chat("What did Paul Graham do after YC?")

logger.debug(response)

response = chat_engine.chat("What about after that?")

logger.debug(response)

response = chat_engine.chat("Can you tell me more?")

logger.debug(response)

"""
### Context ChatEngine
"""
logger.info("### Context ChatEngine")


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
logger.debug(response)

response = chat_engine.chat("What did Paul Graham do after YC?")
logger.debug(response)

response = chat_engine.chat("What about after that?")
logger.debug(response)

response = chat_engine.chat("Can you tell me more?")
logger.debug(response)

"""
### CondenseContext ChatEngine
"""
logger.info("### CondenseContext ChatEngine")


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

logger.debug(response)

response = chat_engine.chat("What did Paul Graham do after YC?")

logger.debug(response)

response = chat_engine.chat("What about after that?")

logger.debug(response)

response = chat_engine.chat("Can you tell me more?")

logger.debug(response)

"""
## Customizing RAG Pipeline
"""
logger.info("## Customizing RAG Pipeline")


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

logger.debug(response)

"""
## Index as Retriever
"""
logger.info("## Index as Retriever")

retriever = index.as_retriever(similarity_top_k=3)

retrieved_nodes = retriever.retrieve("What did Paul Graham do growing up?")


for text_node in retrieved_nodes:
    display_source_node(text_node, source_length=500)

logger.info("\n\n[DONE]", bright=True)