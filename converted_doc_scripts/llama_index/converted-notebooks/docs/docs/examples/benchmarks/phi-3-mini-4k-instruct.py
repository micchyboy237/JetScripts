from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.web import BeautifulSoupWebReader
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
from typing import List
import locale
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
# Feature Test for Phi-3-mini-4k-instruct

[Model card on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

Here's the [technical report](https://arxiv.org/abs/2404.14219).
"""
logger.info("# Feature Test for Phi-3-mini-4k-instruct")

# !pip install llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface transformers accelerate bitsandbytes llama-index-readers-web matplotlib flash-attn

hf_token = "hf_"

"""
## Setup

### Data
"""
logger.info("## Setup")


url = "https://www.theverge.com/2023/9/29/23895675/ai-bot-social-network-openai-meta-chatbots"

documents = BeautifulSoupWebReader().load_data([url])

"""
### LLM
"""
logger.info("### LLM")



def messages_to_prompt(messages):
    prompt = ""
    system_found = False
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
            system_found = True
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"
        else:
            prompt += f"<|user|>\n{message.content}<|end|>\n"

    prompt += "<|assistant|>\n"

    if not system_found:
        prompt = (
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
        )

    return prompt


llm = HuggingFaceLLM(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    model_kwargs={
        "trust_remote_code": True,
    },
    generate_kwargs={"do_sample": True, "temperature": 0.1},
    tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
    query_wrapper_prompt=(
        "<|system|>\n"
        "You are a helpful AI assistant.<|end|>\n"
        "<|user|>\n"
        "{query_str}<|end|>\n"
        "<|assistant|>\n"
    ),
    messages_to_prompt=messages_to_prompt,
    is_chat_model=True,
)


Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

"""
### Index Setup
"""
logger.info("### Index Setup")


vector_index = VectorStoreIndex.from_documents(documents)


summary_index = SummaryIndex.from_documents(documents)

"""
### Helpful Imports / Logging
"""
logger.info("### Helpful Imports / Logging")



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Basic Query Engine

### Compact (default)
"""
logger.info("## Basic Query Engine")

query_engine = vector_index.as_query_engine(response_mode="compact")

response = query_engine.query("How do OllamaFunctionCallingAdapter and Meta differ on AI tools?")

display_response(response)

"""
### Refine
"""
logger.info("### Refine")

query_engine = vector_index.as_query_engine(response_mode="refine")

response = query_engine.query("How do OllamaFunctionCallingAdapter and Meta differ on AI tools?")

display_response(response)

"""
### Tree Summarize
"""
logger.info("### Tree Summarize")

query_engine = vector_index.as_query_engine(response_mode="tree_summarize")

response = query_engine.query("How do OllamaFunctionCallingAdapter and Meta differ on AI tools?")

display_response(response)

"""
## Router Query Engine
"""
logger.info("## Router Query Engine")


vector_tool = QueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)

"""
### Single Selector
"""
logger.info("### Single Selector")


query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False
)

response = query_engine.query("What was mentioned about Meta?")

display_response(response)

"""
### Multi Selector
"""
logger.info("### Multi Selector")


query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    select_multi=True,
)

response = query_engine.query(
    "What was mentioned about Meta? Summarize with any other companies mentioned in the entire document."
)

display_response(response)

"""
## SubQuestion Query Engine
"""
logger.info("## SubQuestion Query Engine")


vector_tool = QueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)

# import nest_asyncio

# nest_asyncio.apply()


query_engine = SubQuestionQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    verbose=True,
)

response = query_engine.query(
    "What was mentioned about Meta? How Does it differ from how OllamaFunctionCallingAdapter is talked about?"
)

display_response(response)

"""
## SQL Query Engine

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.
"""
logger.info("## SQL Query Engine")


locale.getpreferredencoding = lambda: "UTF-8"

# !curl "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "./chinook.zip"
# !unzip "./chinook.zip"


engine = create_engine("sqlite:///chinook.db")


sql_database = SQLDatabase(engine)


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
)

response = query_engine.query("What are some albums? Limit to 5.")

display_response(response)

response = query_engine.query("What are some artists? Limit it to 5.")

display_response(response)

"""
This last query should be a more complex join
"""
logger.info("This last query should be a more complex join")

response = query_engine.query(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

display_response(response)

logger.debug(response.metadata["sql_query"])

"""
## Programs

Depending the LLM, you will have to test with either `OllamaFunctionCallingAdapterPydanticProgram` or `LLMTextCompletionProgram`
"""
logger.info("## Programs")




class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Album),
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

output = program(movie_name="The Shining")

logger.debug(output)

"""
## Data Agent

Similar to programs, OllamaFunctionCallingAdapter LLMs will use `FunctionAgent`, while other LLMs will use `ReActAgent`.
"""
logger.info("## Data Agent")


agent = ReActAgent.from_tools(
    [vector_tool, summary_tool], llm=llm, verbose=True
)

response = agent.chat("Hello!")
logger.debug(response)

"""
#### It does not use the tools to answer the query.
"""
logger.info("#### It does not use the tools to answer the query.")

response = agent.chat(
    "What was mentioned about Meta? How Does it differ from how OllamaFunctionCallingAdapter is talked about?"
)
logger.debug(response)

"""
## Agents with Simple Calculator tools
"""
logger.info("## Agents with Simple Calculator tools")



def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
)

response = agent.chat("What is (121 + 2) * 5?")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)