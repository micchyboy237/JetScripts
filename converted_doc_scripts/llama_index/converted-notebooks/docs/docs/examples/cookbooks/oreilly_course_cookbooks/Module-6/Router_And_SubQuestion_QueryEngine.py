from jet.transformers.formatters import format_json
from IPython.display import display, HTML
from jet.logger import CustomLogger
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.core import SimpleKeywordTableIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionQueryEngine,
)
from llama_index.core.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.core.selectors.pydantic_selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools.query_engine import QueryEngineTool, ToolMetadata
import logging
import openai
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
# Router QueryEngine and SubQuestion QueryEngine

In this notebook we will demonstrate:

1. **RouterQueryEngine** - Handle user queries to choose from predefined indices.
2. **SubQuestionQueryEngine** - breaks down the complex query into sub-questions for each relevant data source, then gathers all the intermediate responses and synthesizes a final response.

[RouterQueryEngine Documentation](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/)

[SubQuestionQueryEngine Documentation](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/)

## Router QueryEngine

Routers act as specialized modules that handle user queries and choose from a set of predefined options, each defined by specific metadata.

There are two main types of core router modules:

1. **LLM Selectors**: These modules present the available options as a text prompt and use the LLM text completion endpoint to make decisions.

2. **Pydantic Selectors**: These modules format the options as Pydantic schemas and pass them to a function-calling endpoint, returning the results as Pydantic objects.

### Installation
"""
logger.info("# Router QueryEngine and SubQuestion QueryEngine")

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()


logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

logger.handlers = []

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

logger.addHandler(handler)


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Load Data
"""
logger.info("### Load Data")

documents = SimpleDirectoryReader("data/paul_graham").load_data()

"""
### Create Nodes
"""
logger.info("### Create Nodes")


parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = parser(documents)

"""
### Create VectorStoreIndex and SummaryIndex.
"""
logger.info("### Create VectorStoreIndex and SummaryIndex.")

summary_index = SummaryIndex(nodes)

vector_index = VectorStoreIndex(nodes)

"""
### Define Query Engines.

1. Summary Index Query Engine.
2. Vector Index Query Engine.
"""
logger.info("### Define Query Engines.")

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

vector_query_engine = vector_index.as_query_engine()

"""
### Build summary index and vector index tools
"""
logger.info("### Build summary index and vector index tools")


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham eassy on What I Worked On.",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)

"""
### Define Router Query Engine

Various selectors are at your disposal, each offering unique characteristics.

Pydantic selectors, supported exclusively by gpt-4 and the default gpt-3.5-turbo, utilize the OllamaFunctionCallingAdapter Function Call API. Instead of interpreting raw JSON, they yield pydantic selection objects.

On the other hand, LLM selectors employ the LLM to generate a JSON output, which is then parsed to query the relevant indexes.

For both selector types, you can opt to route to either a single index or multiple indexes.

### PydanticSingleSelector

Use the OllamaFunctionCallingAdapter Function API to generate/parse pydantic objects under the hood for the router selector.
"""
logger.info("### Define Router Query Engine")


query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

"""
### LLMSingleSelector

Utilize OllamaFunctionCallingAdapter (or another LLM) to internally interpret the generated JSON and determine a sub-index for routing.
"""
logger.info("### LLMSingleSelector")

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
)

response = query_engine.query("What is the summary of the document?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = query_engine.query("What did Paul Graham do after RICS?")

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

"""
### PydanticMultiSelector

If you anticipate queries being directed to multiple indexes, it's advisable to use a multi-selector. This selector dispatches the query to various sub-indexes and subsequently aggregates the responses through a summary index to deliver a comprehensive answer.

### Let's create a simplekeywordtable index and corresponding tool.
"""
logger.info("### PydanticMultiSelector")


keyword_index = SimpleKeywordTableIndex(nodes)

keyword_query_engine = keyword_index.as_query_engine()

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description="Useful for retrieving specific context using keywords from Paul Graham essay on What I Worked On.",
)

"""
### Build a router query engine.
"""
logger.info("### Build a router query engine.")

query_engine = RouterQueryEngine(
    selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=[vector_tool, keyword_tool, summary_tool],
)

response = query_engine.query(
    "What were noteable events and people from the authors time at Interleaf and YC?"
)

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

"""
## SubQuestion Query Engine

Here, we will demonstrate how to use a sub-question query engine to address the challenge of answering a complex query using multiple data sources.

The SubQuestion Query Engine first breaks down the complex query into sub-questions for each relevant data source, then gathers all the intermediate responses and synthesizes a final response.

### Download Data
"""
logger.info("## SubQuestion Query Engine")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
### Load Data
"""
logger.info("### Load Data")

lyft_docs = SimpleDirectoryReader(
    input_files=[f"{os.path.dirname(__file__)}/data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=[f"{os.path.dirname(__file__)}/data/10k/uber_2021.pdf"]
).load_data()

logger.debug(f"Loaded lyft 10-K with {len(lyft_docs)} pages")
logger.debug(f"Loaded Uber 10-K with {len(uber_docs)} pages")

"""
### Create Indices
"""
logger.info("### Create Indices")

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

"""
### Define Query Engines
"""
logger.info("### Define Query Engines")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)

uber_engine = uber_index.as_query_engine(similarity_top_k=3)

response = lyft_engine.query(
    "What is the revenue of Lyft in 2021? Answer in millions with page reference"
)
logger.success(format_json(response))

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

response = uber_engine.query(
    "What is the revenue of Uber in 2021? Answer in millions, with page reference"
)
logger.success(format_json(response))

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

"""
### Define QueryEngine Tools
"""
logger.info("### Define QueryEngine Tools")

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]

"""
### SubQuestion QueryEngine
"""
logger.info("### SubQuestion QueryEngine")


sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

"""
### Querying
"""
logger.info("### Querying")

response = sub_question_query_engine.query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)
logger.success(format_json(response))

display(HTML(f'<p style="font-size:20px">{response.response}</p>'))

logger.info("\n\n[DONE]", bright=True)
