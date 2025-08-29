from huggingface_hub import notebook_login
from jet.logger import CustomLogger
from llama_index.core import (
    KnowledgeGraphIndex,
    StorageContext,
)
from llama_index.core import (
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    Settings,
    StorageContext,
)
from llama_index.core import PropertyGraphIndex
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.gaudi import GaudiEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.gaudi import GaudiLLM
from llama_index.readers.wikipedia import WikipediaReader
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)
import argparse
import neo4j
import os
import os
import sys
import logging
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LLM Cookbook with Intel Gaudi

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/llama3_cookbook_gaudi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Meta developed and released the Meta [Llama 3](https://ai.meta.com/blog/meta-llama-3/) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

In this notebook, we will demonstrate how to use Llama3 with LlamaIndex. 

We use Llama-3-8B-Instruct for the demonstration through Intel Gaudi.

## Installation and Setup
"""
logger.info("# LLM Cookbook with Intel Gaudi")

# !pip -q install llama-parse
# !pip -q install python-dotenv==1.0.0
# !pip -q install llama_index
# !pip -q install llama-index-llms-gaudi
# !pip -q install llama-index-embeddings-gaudi
# !pip -q install llama-index-graph-stores-neo4j
# !pip -q install llama-index-readers-wikipedia
# !pip -q install wikipedia
# !pip -q install InstructorEmbedding==1.0.1
# !pip -q install sentence-transformers
# !pip -q install --upgrade-strategy eager optimum[habana]
# !pip -q install optimum-habana==1.14.1
# !pip -q install huggingface-hub==0.23.2

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AttributeContainer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


args = AttributeContainer(
    device="hpu",
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    bf16=True,
    max_new_tokens=100,
    max_input_tokens=0,
    batch_size=1,
    warmup=3,
    n_iterations=5,
    local_rank=0,
    use_kv_cache=True,
    use_hpu_graphs=True,
    dataset_name=None,
    column_name=None,
    do_sample=False,
    num_beams=1,
    trim_logits=False,
    seed=27,
    profiling_warmup_steps=0,
    profiling_steps=0,
    profiling_record_shapes=False,
    prompt=None,
    bad_words=None,
    force_words=None,
    assistant_model=None,
    peft_model=None,
    token=None,
    model_revision="main",
    attn_softmax_bf16=False,
    output_dir=None,
    bucket_size=-1,
    dataset_max_samples=-1,
    limit_hpu_graphs=False,
    reuse_cache=False,
    verbose_workers=False,
    simulate_dyn_prompt=None,
    reduce_recompile=False,
    use_flash_attention=False,
    flash_attention_recompute=False,
    flash_attention_causal_mask=False,
    flash_attention_fast_softmax=False,
    book_source=False,
    torch_compile=False,
    ignore_eos=True,
    temperature=1.0,
    top_p=1.0,
    const_serialization_path=None,
    csp=None,
    disk_offload=False,
    trust_remote_code=False,
    quant_config=os.getenv("QUANT_CONFIG", ""),
    num_return_sequences=1,
    bucket_internal=False,
)


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    prompt = prompt + "<|assistant|>\n"

    return prompt


"""
### Setup LLM using Intel Gaudi
"""
logger.info("### Setup LLM using Intel Gaudi")


notebook_login()


llm = GaudiLLM(
    args=args,
    logger=logger,
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    query_wrapper_prompt=PromptTemplate(
        "<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
    ),
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

"""
### Setup Embedding Model
"""
logger.info("### Setup Embedding Model")


embed_model = GaudiEmbedding(
    embedding_input_size=-1, model_name="BAAI/bge-small-en-v1.5"
)

"""
### Define Global Settings Configuration

In LlamaIndex, you can define global settings so you don't have to pass the LLM / embedding model objects everywhere.
"""
logger.info("### Define Global Settings Configuration")


Settings.llm = llm
Settings.embed_model = embed_model

"""
### Download Data

Here you'll download data that's used in section 2 and onwards.
"""
logger.info("### Download Data")

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

"""
### Load Data

We load data using LlamaParse by default, but you can also choose to opt for our free pypdf reader (in SimpleDirectoryReader by default) if you don't have an account! 

1. LlamaParse: Signup for an account here: cloud.llamaindex.ai. You get 1k free pages a day, and paid plan is 7k free pages + 0.3c per additional page. LlamaParse is a good option if you want to parse complex documents, like PDFs with charts, tables, and more. 

2. Default PDF Parser (In `SimpleDirectoryReader`). If you don't want to signup for an account / use a PDF service, just use the default PyPDF reader bundled in our file loader. It's a good choice for getting started!
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

"""
## 1. Basic Completion and Chat

### Call complete with a prompt
"""
logger.info("## 1. Basic Completion and Chat")

response = llm.complete("Who is Paul Graham?")

logger.debug(response)

stream_response = llm.stream_complete(
    "you're a Paul Graham fan. tell me why you like Paul Graham"
)

for t in stream_response:
    logger.debug(t.delta, end="")

"""
### Call chat with a list of messages
"""
logger.info("### Call chat with a list of messages")


messages = [
    ChatMessage(role="system", content="You are Paul Graham."),
    ChatMessage(role="user", content="Write a paragraph about politics."),
]
response = llm.chat(messages)

logger.debug(response)

"""
## 2. Basic RAG (Vector Search, Summarization)

### Basic RAG (Vector Search)
"""
logger.info("## 2. Basic RAG (Vector Search, Summarization)")


index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("Tell me about family matters")

logger.debug(str(response))

"""
### Basic RAG (Summarization)
"""
logger.info("### Basic RAG (Summarization)")


summary_index = SummaryIndex.from_documents(documents)
summary_engine = summary_index.as_query_engine()

response = summary_engine.query(
    "Given your assessment of this article, what is Paul Graham best known for?"
)

logger.debug(str(response))

"""
## 3. Advanced RAG (Routing)

### Build a Router that can choose whether to do vector search or summarization
"""
logger.info("## 3. Advanced RAG (Routing)")


vector_tool = QueryEngineTool(
    index.as_query_engine(llm=llm),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize", llm=llm),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)


query_engine = SubQuestionQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    llm=llm,
    verbose=True,
)
response = query_engine.query("tell me something about paul graham?")

logger.debug(response)

"""
## 4. Text-to-SQL 

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.
"""
logger.info("## 4. Text-to-SQL")

# !wget "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/chinook.zip"
# !unzip "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/chinook.zip"


engine = create_engine("sqlite:///chinook.db")


sql_database = SQLDatabase(engine)


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    llm=llm,
)

response = query_engine.query("What are some albums?")

logger.debug(response)

response = query_engine.query("What are some artists? Limit it to 5.")

logger.debug(response)

"""
This last query should be a more complex join
"""
logger.info("This last query should be a more complex join")

response = query_engine.query(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

logger.debug(response)

logger.debug(response.metadata["sql_query"])

"""
## 5. Structured Data Extraction - Graph RAG with Local NEO4J Database
"""
logger.info(
    "## 5. Structured Data Extraction - Graph RAG with Local NEO4J Database")


graph_store = Neo4jGraphStore(
    username="<user_name for NEO4J server>",
    password="<password for NEO4J server>",
    url="<URL for NEO4J server>",
    database="neo4j",
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)
neo4j_index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    max_triplets_per_chunk=3,
    storage_context=storage_context,
    embed_model=embed_model,
    include_embeddings=True,
)

struct_query_engine = neo4j_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)

response = struct_query_engine.query("who is paul graham?")

logger.debug(response)

"""
## 6. Adding Chat History to RAG (Chat Engine)

In this section we create a stateful chatbot from a RAG pipeline, with our chat engine abstraction.

Unlike a stateless query engine, the chat engine maintains conversation history (through a memory module like buffer memory). It performs retrieval given a condensed question, and feeds the condensed question + context + chat history into the final LLM prompt.

Related resource: https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/
"""
logger.info("## 6. Adding Chat History to RAG (Chat Engine)")


memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about Paul Graham."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)

response = chat_engine.chat(
    "Tell me about the essay Paul Graham wrote on the topic of programming."
)
logger.debug(str(response))

response = chat_engine.chat(
    "What about the essays Paul Graham wrote on other topics?"
)
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
