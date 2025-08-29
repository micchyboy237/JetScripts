async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import SQLDatabase
    from llama_index.core import Settings
    from llama_index.core import StorageContext
    from llama_index.core import VectorStoreIndex
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.agent.workflow import (
    ToolCallResult,
    ToolCall,
    AgentStream,
    AgentInput,
    AgentOutput,
    )
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.indices import SQLStructStoreIndex
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.query_engine import NLSQLTableQueryEngine
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexAutoRetriever
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import TextNode
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools import QueryEngineTool
    from llama_index.core.vector_stores import (
    VectorStoreInfo,
    MetadataInfo,
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    )
    from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
    from llama_index.core.workflow import Context
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.readers.wikipedia import WikipediaReader
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
    from pydantic import BaseModel, Field
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
    from sqlalchemy import insert
    from typing import Any, Annotated
    from typing import List, Tuple, Any
    import os
    import shutil
    import time
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_query_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # OllamaFunctionCallingAdapter Agent + Query Engine Experimental Cookbook
    
    
    In this notebook, we try out the FunctionAgent across a variety of query engine tools and datasets. We explore how FunctionAgent can compare/replace existing workflows solved by our retrievers/query engines.
    
    - Auto retrieval 
    - Joint SQL and vector search
    
    **NOTE:** Any Text-to-SQL application should be aware that executing 
    arbitrary SQL queries can be a security risk. It is recommended to
    take precautions as needed, such as using restricted roles, read-only
    databases, sandboxing, etc.
    
    ## AutoRetrieval from a Vector Database
    
    Our existing "auto-retrieval" capabilities (in `VectorIndexAutoRetriever`) allow an LLM to infer the right query parameters for a vector database - including both the query string and metadata filter.
    
    Since the OllamaFunctionCallingAdapter Function API can infer function parameters, we explore its capabilities in performing auto-retrieval here.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# OllamaFunctionCallingAdapter Agent + Query Engine Experimental Cookbook")
    
    # %pip install llama-index
    # %pip install llama-index-llms-ollama
    # %pip install llama-index-readers-wikipedia
    # %pip install llama-index-vector-stores-pinecone
    
    
    os.environ["PINECONE_API_KEY"] = "..."
    # os.environ["OPENAI_API_KEY"] = "..."
    
    
    Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    pc.create_index(
        name="quickstart-index",
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    
    time.sleep(10)
    
    index = pc.Index("quickstart-index")
    
    
    
    
    
    nodes = [
        TextNode(
            text=(
                "Michael Jordan is a retired professional basketball player,"
                " widely regarded as one of the greatest basketball players of all"
                " time."
            ),
            metadata={
                "category": "Sports",
                "country": "United States",
                "gender": "male",
                "born": 1963,
            },
        ),
        TextNode(
            text=(
                "Angelina Jolie is an American actress, filmmaker, and"
                " humanitarian. She has received numerous awards for her acting"
                " and is known for her philanthropic work."
            ),
            metadata={
                "category": "Entertainment",
                "country": "United States",
                "gender": "female",
                "born": 1975,
            },
        ),
        TextNode(
            text=(
                "Elon Musk is a business magnate, industrial designer, and"
                " engineer. He is the founder, CEO, and lead designer of SpaceX,"
                " Tesla, Inc., Neuralink, and The Boring Company."
            ),
            metadata={
                "category": "Business",
                "country": "United States",
                "gender": "male",
                "born": 1971,
            },
        ),
        TextNode(
            text=(
                "Rihanna is a Barbadian singer, actress, and businesswoman. She"
                " has achieved significant success in the music industry and is"
                " known for her versatile musical style."
            ),
            metadata={
                "category": "Music",
                "country": "Barbados",
                "gender": "female",
                "born": 1988,
            },
        ),
        TextNode(
            text=(
                "Cristiano Ronaldo is a Portuguese professional footballer who is"
                " considered one of the greatest football players of all time. He"
                " has won numerous awards and set multiple records during his"
                " career."
            ),
            metadata={
                "category": "Sports",
                "country": "Portugal",
                "gender": "male",
                "born": 1985,
            },
        ),
    ]
    
    
    vector_store = PineconeVectorStore(pinecone_index=index, namespace="test")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    """
    #### Define Function Tool
    
    Here we define the function interface, which is passed to OllamaFunctionCallingAdapter to perform auto-retrieval.
    
    We were not able to get OllamaFunctionCallingAdapter to work with nested pydantic objects or tuples as arguments,
    so we converted the metadata filter keys and values into lists for the function API to work with.
    """
    logger.info("#### Define Function Tool")
    
    
    
    
    vector_store_info = VectorStoreInfo(
        content_info="brief biography of celebrities",
        metadata_info=[
            MetadataInfo(
                name="category",
                type="str",
                description=(
                    "Category of the celebrity, one of [Sports, Entertainment,"
                    " Business, Music]"
                ),
            ),
            MetadataInfo(
                name="country",
                type="str",
                description=(
                    "Country of the celebrity, one of [United States, Barbados,"
                    " Portugal]"
                ),
            ),
            MetadataInfo(
                name="gender",
                type="str",
                description=("Gender of the celebrity, one of [male, female]"),
            ),
            MetadataInfo(
                name="born",
                type="int",
                description=("Born year of the celebrity, could be any integer"),
            ),
        ],
    )
    
    """
    Define AutoRetrieve Functions
    """
    logger.info("Define AutoRetrieve Functions")
    
    
    
    async def auto_retrieve_fn(
        query: Annotated[str, "The natural language query/question to answer."],
        filter_key_list: Annotated[
            List[str], "List of metadata filter field names"
        ],
        filter_value_list: Annotated[
            List[Any],
            "List of metadata filter field values (corresponding to names in filter_key_list)",
        ],
        filter_operator_list: Annotated[
            List[str],
            "Metadata filters conditions (could be one of <, <=, >, >=, ==, !=)",
        ],
        filter_condition: Annotated[
            str, "Metadata filters condition values (could be AND or OR)"
        ],
        top_k: Annotated[
            int, "The number of results to return from the vector database."
        ],
    ):
        """Auto retrieval function.
    
        Performs auto-retrieval from a vector database, and then applies a set of filters.
    
        """
        query = query or "Query"
    
        metadata_filters = [
            MetadataFilter(key=k, value=v, operator=op)
            for k, v, op in zip(
                filter_key_list, filter_value_list, filter_operator_list
            )
        ]
        retriever = VectorIndexRetriever(
            index,
            filters=MetadataFilters(
                filters=metadata_filters, condition=filter_condition.lower()
            ),
            top_k=top_k,
        )
        query_engine = RetrieverQueryEngine.from_args(retriever)
    
        response = query_engine.query(query)
        logger.success(format_json(response))
        return str(response)
    
    
    description = f"""\
    Use this tool to look up biographical information about celebrities.
    The vector database schema is given below:
    
    <schema>
    {vector_store_info.model_dump_json()}
    </schema>
    """
    
    auto_retrieve_tool = FunctionTool.from_defaults(
        auto_retrieve_fn,
        name="celebrity_bios",
        description=description,
    )
    
    """
    #### Initialize Agent
    """
    logger.info("#### Initialize Agent")
    
    
    agent = FunctionAgent(
        tools=[auto_retrieve_tool],
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
        system_prompt=(
            "You are a helpful assistant that can answer questions about celebrities by writing a filtered query to a vector database. "
            "Unless the user is asking to compare things, you generally only need to make one call to the retriever."
        ),
    )
    
    ctx = Context(agent)
    
    
    handler = agent.run(
        "Tell me about two celebrities from the United States. ", ctx=ctx
    )
    
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"\nCalled tool {ev.tool_name} with args {ev.tool_kwargs}, got response: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            logger.debug(ev.delta, end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    handler = agent.run("Tell me about two celebrities born after 1980. ", ctx=ctx)
    
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"\nCalled tool {ev.tool_name} with args {ev.tool_kwargs}, got response: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            logger.debug(ev.delta, end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    response = await agent.run(
            "Tell me about few celebrities under category business and born after 1950. ",
            ctx=ctx,
        )
    logger.success(format_json(response))
    logger.debug(str(response))
    
    """
    ## Joint Text-to-SQL and Semantic Search
    
    This is currently handled by our `SQLAutoVectorQueryEngine`.
    
    Let's try implementing this by giving our `FunctionAgent` access to two query tools: SQL and Vector 
    
    **NOTE:** Any Text-to-SQL application should be aware that executing 
    arbitrary SQL queries can be a security risk. It is recommended to
    take precautions as needed, such as using restricted roles, read-only
    databases, sandboxing, etc.
    
    #### Load and Index Structured Data
    
    We load sample structured datapoints into a SQL db and index it.
    """
    logger.info("## Joint Text-to-SQL and Semantic Search")
    
    
    engine = create_engine("sqlite:///:memory:", future=True)
    metadata_obj = MetaData()
    
    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    
    metadata_obj.create_all(engine)
    
    metadata_obj.tables.keys()
    
    
    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            cursor = connection.execute(stmt)
    
    with engine.connect() as connection:
        cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
        logger.debug(cursor.fetchall())
    
    sql_database = SQLDatabase(engine, include_tables=["city_stats"])
    
    
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
    )
    
    """
    #### Load and Index Unstructured Data
    
    We load unstructured data into a vector index backed by Pinecone
    """
    logger.info("#### Load and Index Unstructured Data")
    
    # %pip install wikipedia llama-index-readers-wikipedia
    
    
    cities = ["Toronto", "Berlin", "Tokyo"]
    wiki_docs = WikipediaReader().load_data(pages=cities)
    
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    pc.create_index(
        name="quickstart-sql",
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    
    
    time.sleep(10)
    
    index = pc.Index("quickstart-sql")
    
    index.delete(deleteAll=True)
    
    
    Settings.llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    Settings.node_parser = TokenTextSplitter(chunk_size=1024)
    
    vector_store = PineconeVectorStore(
        pinecone_index=index, namespace="wiki_cities"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex([], storage_context=storage_context)
    
    for city, wiki_doc in zip(cities, wiki_docs):
        nodes = Settings.node_parser.get_nodes_from_documents([wiki_doc])
        for node in nodes:
            node.metadata = {"title": city}
        vector_index.insert_nodes(nodes)
    
    """
    #### Define Query Engines / Tools
    """
    logger.info("#### Define Query Engines / Tools")
    
    
    
    vector_store_info = VectorStoreInfo(
        content_info="articles about different cities",
        metadata_info=[
            MetadataInfo(
                name="title", type="str", description="The name of the city"
            ),
        ],
    )
    
    vector_auto_retriever = VectorIndexAutoRetriever(
        vector_index, vector_store_info=vector_store_info
    )
    
    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever,
    )
    
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        name="vector_tool",
        description=(
            "Useful for answering semantic questions about different cities"
        ),
    )
    
    """
    #### Initialize Agent
    """
    logger.info("#### Initialize Agent")
    
    
    agent = FunctionAgent(
        tools=[sql_tool, vector_tool],
        llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096),
    )
    
    ctx = Context(agent)
    
    
    handler = agent.run(
        "Tell me about the arts and culture of the city with the highest population. ",
        ctx=ctx,
    )
    
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"\nCalled tool {ev.tool_name} with args {ev.tool_kwargs}, got response: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            logger.debug(ev.delta, end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    handler = agent.run("Tell me about the history of Berlin", ctx=ctx)
    
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"\nCalled tool {ev.tool_name} with args {ev.tool_kwargs}, got response: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            logger.debug(ev.delta, end="", flush=True)
    
    response = await handler
    logger.success(format_json(response))
    
    response = await agent.run(
            "Can you give me the country corresponding to each city?", ctx=ctx
        )
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())