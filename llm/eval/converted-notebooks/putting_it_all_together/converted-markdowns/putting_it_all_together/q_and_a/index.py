from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("<summarization_query>")

from llama_index.core import TreeIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool

...

index1 = VectorStoreIndex.from_documents(notion_docs)
index2 = VectorStoreIndex.from_documents(slack_docs)

tool1 = QueryEngineTool.from_defaults(
    query_engine=index1.as_query_engine(),
    description="Use this query engine to do...",
)
tool2 = QueryEngineTool.from_defaults(
    query_engine=index2.as_query_engine(),
    description="Use this query engine for something else...",
)

from llama_index.core.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2]
)

response = query_engine.query(
    "In Notion, give me a summary of the product roadmap."
)

from llama_index.core.query.query_transform.base import DecomposeQueryTransform

decompose_transform = DecomposeQueryTransform(
    service_context.llm, verbose=True
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=sept_engine,
        metadata=ToolMetadata(
            name="sept_22",
            description="Provides information about Uber quarterly financials ending September 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=june_engine,
        metadata=ToolMetadata(
            name="june_22",
            description="Provides information about Uber quarterly financials ending June 2022",
        ),
    ),
    QueryEngineTool(
        query_engine=march_engine,
        metadata=ToolMetadata(
            name="march_22",
            description="Provides information about Uber quarterly financials ending March 2022",
        ),
    ),
]

from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

logger.info("\n\n[DONE]", bright=True)