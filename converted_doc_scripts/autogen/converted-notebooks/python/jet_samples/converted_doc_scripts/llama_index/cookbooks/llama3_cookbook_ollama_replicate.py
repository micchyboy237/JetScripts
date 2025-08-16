from jet.logger import CustomLogger
from llama_index.core import (
SimpleDirectoryReader,
VectorStoreIndex,
StorageContext,
load_index_from_storage,
)
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel
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
from typing import Sequence, List
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Llama3 Cookbook with Ollama and Replicate

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/llama3_cookbook_ollama_replicate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Meta developed and released the Meta [Llama 3](https://ai.meta.com/blog/meta-llama-3/) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

In this notebook, we demonstrate how to use Llama3 with LlamaIndex for a comprehensive set of use cases. 
1. Basic completion / chat 
2. Basic RAG (Vector Search, Summarization)
3. Advanced RAG (Routing, Sub-Questions)
4. Text-to-SQL 
5. Structured Data Extraction
6. Chat Engine + Memory
7. Agents


We use Llama3-8B through Ollama, and Llama3-70B through Replicate.

## Installation and Setup
"""
logger.info("# Llama3 Cookbook with Ollama and Replicate")



# import nest_asyncio

# nest_asyncio.apply()

"""
### Setup LLM using Ollama
"""
logger.info("### Setup LLM using Ollama")


llm = Ollama(model="llama3.2")

"""
### Setup LLM using Replicate

Make sure you have REPLICATE_API_TOKEN specified!
"""
logger.info("### Setup LLM using Replicate")



llm_replicate = Ollama(model="llama3.1")

"""
### Setup Embedding Model
"""
logger.info("### Setup Embedding Model")


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

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

We'll download some articles on Kendrick, Drake, and their beef (as of May 2024).
"""
logger.info("### Download Data")





def save_as_json(docs, file_path, sub_dir):
    json_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".json"
    json_file_path = os.path.join("./data", sub_dir, json_file_name)

    doc_data = [{"text": doc.text, "metadata": doc.metadata} for doc in docs]

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(doc_data, json_file, ensure_ascii=False, indent=2)

    logger.debug(f"Saved: {json_file_path}")

"""
### Load Data

We load data using LlamaParse by default, but you can also choose to opt for our free pypdf reader (in SimpleDirectoryReader by default) if you don't have an account! 

1. LlamaParse: Signup for an account here: cloud.llamaindex.ai. You get 1k free pages a day, and paid plan is 7k free pages + 0.3c per additional page. LlamaParse is a good option if you want to parse complex documents, like PDFs with charts, tables, and more. 

2. Default PDF Parser (In `SimpleDirectoryReader`). If you don't want to signup for an account / use a PDF service, just use the default PyPDF reader bundled in our file loader. It's a good choice for getting started!
"""
logger.info("### Load Data")



"""
## 1. Basic Completion and Chat
"""
logger.info("## 1. Basic Completion and Chat")

docs_kendrick = SimpleDirectoryReader(input_files=["./data/kendrick.pdf"]).load_data()
docs_drake = SimpleDirectoryReader(input_files=["./data/drake.pdf"]).load_data()
docs_both = SimpleDirectoryReader(input_files=["./data/drake_kendrick_beef.pdf"]).load_data()

sub_dir = "simple_directory_reader"
save_as_json(docs_kendrick,"./data/kendrick.pdf", sub_dir)
save_as_json(docs_drake,"./data/drake.pdf", sub_dir)
save_as_json(docs_both,"./data/drake_kendrick_beef.pdf", sub_dir)

"""
### Call complete with a prompt
"""
logger.info("### Call complete with a prompt")

response = llm.complete("do you like drake or kendrick better?")

logger.debug(response)

stream_response = llm.stream_complete(
    "you're a drake fan. tell me why you like drake more than kendrick"
)

for t in stream_response:
    logger.debug(t.delta, end="")

"""
### Call chat with a list of messages
"""
logger.info("### Call chat with a list of messages")


messages = [
    ChatMessage(role="system", content="You are Kendrick."),
    ChatMessage(role="user", content="Write a verse."),
]
response = llm.chat(messages)

logger.debug(response)

"""
## 2. Basic RAG (Vector Search, Summarization)

### Basic RAG (Vector Search)
"""
logger.info("## 2. Basic RAG (Vector Search, Summarization)")


index = VectorStoreIndex.from_documents(docs_both)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("Tell me about family matters")

logger.debug(str(response))

"""
### Basic RAG (Summarization)
"""
logger.info("### Basic RAG (Summarization)")


summary_index = SummaryIndex.from_documents(docs_both)
summary_engine = summary_index.as_query_engine()

response = summary_engine.query(
    "Given your assessment of this article, who won the beef?"
)

logger.debug(str(response))

"""
## 3. Advanced RAG (Routing, Sub-Questions)

### Build a Router that can choose whether to do vector search or summarization
"""
logger.info("## 3. Advanced RAG (Routing, Sub-Questions)")


vector_tool = QueryEngineTool(
    index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)


query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True
)

response = query_engine.query(
    "Tell me about the song meet the grahams - why is it significant"
)

logger.debug(response)

"""
### Break Complex Questions down into Sub-Questions

Our Sub-Question Query Engine breaks complex questions down into sub-questions.
"""
logger.info("### Break Complex Questions down into Sub-Questions")

drake_index = VectorStoreIndex.from_documents(docs_drake)
drake_query_engine = drake_index.as_query_engine(similarity_top_k=3)

kendrick_index = VectorStoreIndex.from_documents(docs_kendrick)
kendrick_query_engine = kendrick_index.as_query_engine(similarity_top_k=3)


drake_tool = QueryEngineTool(
    drake_index.as_query_engine(),
    metadata=ToolMetadata(
        name="drake_search",
        description="Useful for searching over Drake's life.",
    ),
)

kendrick_tool = QueryEngineTool(
    kendrick_index.as_query_engine(),
    metadata=ToolMetadata(
        name="kendrick_summary",
        description="Useful for searching over Kendrick's life.",
    ),
)


query_engine = SubQuestionQueryEngine.from_defaults(
    [drake_tool, kendrick_tool],
    llm=llm_replicate,  # llama3-70b
    verbose=True,
)

response = query_engine.query("Which albums did Drake release in his career?")

logger.debug(response)

"""
## 4. Text-to-SQL 

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.
"""
logger.info("## 4. Text-to-SQL")

# !wget "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "./data/chinook.zip"
# !unzip "./data/chinook.zip"


engine = create_engine("sqlite:///chinook.db")


sql_database = SQLDatabase(engine)


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    llm=llm_replicate,
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
## 5. Structured Data Extraction

An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for this through `structured_predict` - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.

**NOTE**: Since there's no native function calling support with Llama3 / Ollama, the structured extraction is performed by prompting the LLM + output parsing.
"""
logger.info("## 5. Structured Data Extraction")



class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str


llm = Ollama(model="llama3.2")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

restaurant_obj = llm.structured_predict(
    Restaurant, prompt_tmpl, city_name="Miami"
)
logger.debug(restaurant_obj)

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
        " about the Kendrick and Drake beef."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)

response = chat_engine.chat(
    "Tell me about the songs Drake released in the beef."
)
logger.debug(str(response))

response = chat_engine.chat("What about Kendrick?")
logger.debug(str(response))

"""
## 7. Agents

Here we build agents with Llama 3. We perform RAG over simple functions as well as the documents above.

### Agents And Tools
"""
logger.info("## 7. Agents")



# import nest_asyncio

# nest_asyncio.apply()

"""
### Define Tools
"""
logger.info("### Define Tools")

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

"""
### ReAct Agent
"""
logger.info("### ReAct Agent")

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm_replicate,
    verbose=True,
)

"""
### Querying
"""
logger.info("### Querying")

response = agent.chat("What is (121 + 2) * 5?")
logger.debug(str(response))

"""
### ReAct Agent With RAG QueryEngine Tools
"""
logger.info("### ReAct Agent With RAG QueryEngine Tools")



"""
### Create ReAct Agent using RAG QueryEngine Tools
"""
logger.info("### Create ReAct Agent using RAG QueryEngine Tools")

drake_tool = QueryEngineTool(
    drake_index.as_query_engine(),
    metadata=ToolMetadata(
        name="drake_search",
        description="Useful for searching over Drake's life.",
    ),
)

kendrick_tool = QueryEngineTool(
    kendrick_index.as_query_engine(),
    metadata=ToolMetadata(
        name="kendrick_search",
        description="Useful for searching over Kendrick's life.",
    ),
)

query_engine_tools = [drake_tool, kendrick_tool]

agent = ReActAgent.from_tools(
    query_engine_tools,  ## TODO: define query tools
    llm=llm_replicate,
    verbose=True,
)

"""
### Querying
"""
logger.info("### Querying")

response = agent.chat("Tell me about how Kendrick and Drake grew up")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)