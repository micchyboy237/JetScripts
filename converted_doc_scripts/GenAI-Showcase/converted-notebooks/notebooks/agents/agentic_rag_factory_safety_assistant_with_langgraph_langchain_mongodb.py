async def main():
    from jet.transformers.formatters import format_json
    from IPython.display import Image, display
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager
    from datasets import load_dataset
    from datetime import datetime
    from datetime import datetime, timezone
    from jet.adapters.langchain.chat_ollama import ChatOllama
    from jet.llm.ollama.base_langchain import OllamaEmbeddings
    from jet.logger import CustomLogger
    from langchain.agents import tool
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnableConfig
    from langchain_mongodb import MongoDBAtlasVectorSearch
    from langchain_mongodb.retrievers import MongoDBAtlasFullTextSearchRetriever
    from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
        SerializerProtocol,
    )
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.prebuilt import tools_condition
    from motor.motor_asyncio import AsyncIOMotorClient
    from pydantic import BaseModel, Field
    from pymongo.errors import BulkWriteError
    from pymongo.operations import SearchIndexModel
    from tqdm import tqdm
    from types import TracebackType
    from typing import Annotated, TypedDict
    from typing import Any, Dict
    from typing import Any, Dict, List, Optional, Tuple, Union
    from typing import List
    from typing_extensions import Self
    import asyncio
    import functools
    import numpy as np
    import operator
    import os
    import pandas as pd
    import pickle
    import pymongo
    import re
    import shutil
    import tabulate
    import tiktoken

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")

    """
    # Agentic RAG: Factory Safety Assistant
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/agentic_rag_factory_safety_assistant_with_langgraph_langchain_mongodb.ipynb)
    
    [![AI Learning Hub For Developers](https://img.shields.io/badge/AI%20Learning%20Hub%20For%20Developers-Click%20Here-blue)](https://www.mongodb.com/resources/use-cases/artificial-intelligence?utm_campaign=ai_learning_hub&utm_source=github&utm_medium=referral)
    """
    logger.info("# Agentic RAG: Factory Safety Assistant")

    # %pip install --quiet datasets pandas pymongo jet.llm.ollama.base_langchain

    # import getpass

    def set_env_securely(var_name, prompt):
        #     value = getpass.getpass(prompt)
        os.environ[var_name] = value

    OPEN_AI_EMBEDDING_MODEL = "mxbai-embed-large"
    OPEN_AI_EMBEDDING_MODEL_DIMENSION = 256

    # set_env_securely("OPENAI_API_KEY", "Enter your Ollama API key: ")

    accidents_df = pd.read_json("accidents_incidents.json")

    safety_df = pd.read_json("safety_procedures.json")

    safety_procedure_ds = load_dataset(
        "MongoDB/safety_procedure_dataset", split="train")
    safety_df = pd.DataFrame(safety_procedure_ds)

    accident_reports_ds = load_dataset(
        "MongoDB/accident_reports", split="train")
    accidents_df = pd.DataFrame(accident_reports_ds)

    accidents_df.info()

    accidents_df.head()

    safety_df.info()

    safety_df.head()

    def combine_attributes(df, attributes):
        """
        Combine specified attributes of a DataFrame into a single column,
        converting all attributes to strings and handling various data types.

        Parameters:
        df (pandas.DataFrame): The input DataFrame
        attributes (list): List of column names to combine

        Returns:
        pandas.DataFrame: The input DataFrame with an additional 'combined_info' column
        """

        def combine_row(row):
            combined = []
            for attr in attributes:
                if attr in row.index:
                    value = row[attr]
                    if isinstance(value, (pd.Series, np.ndarray, list)):
                        if len(value) > 0 and not pd.isna(value).all():
                            combined.append(f"{attr.capitalize()}: {value!s}")
                    elif not pd.isna(value):
                        combined.append(f"{attr.capitalize()}: {value!s}")
            return " ".join(combined)

        df["combined_info"] = df.apply(combine_row, axis=1)
        return df

    accident_attributes_to_combine = [
        "type",
        "description",
        "immediateActions",
        "rootCauses",
    ]
    accidents_df = combine_attributes(
        accidents_df, accident_attributes_to_combine)

    safety_procedures_attributes_to_combine = [
        "title", "description", "category", "steps"]
    safety_df = combine_attributes(
        safety_df, safety_procedures_attributes_to_combine)

    first_datapoint_accident = accidents_df.iloc[0]
    logger.debug(first_datapoint_accident["combined_info"])

    first_datapoint_safety = safety_df.iloc[0]
    logger.debug(first_datapoint_safety["combined_info"])

    MAX_TOKENS = 8191  # Maximum tokens for mxbai-embed-large
    OVERLAP = 50

    embedding_model = OllamaEmbeddings(
        model=OPEN_AI_EMBEDDING_MODEL, dimensions=OPEN_AI_EMBEDDING_MODEL_DIMENSION
    )

    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
        """
        Split the text into overlapping chunks based on token count.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i: i + max_tokens]
            chunk = encoding.decode(chunk_tokens)
            chunks.append(chunk)
        return chunks

    def get_embedding(input_data, model=OPEN_AI_EMBEDDING_MODEL):
        """
        Generate embeddings for the 'combined_attributes' column and duplicate the row for each chunk
        or generate embeddings for a given string.
        """
        if isinstance(input_data, str):
            text = input_data
        else:
            text = input_data["combined_info"]

        if not text.strip():
            logger.debug("Attempted to get embedding for empty text.")
            return []

        chunks = chunk_text(text)

        chunk_embeddings = []
        for chunk in chunks:
            chunk = chunk.replace("\n", " ")
            embedding = embedding_model.embed_query(text=chunk)
            chunk_embeddings.append(embedding)

        if isinstance(input_data, str):
            return chunk_embeddings[0]
        duplicated_rows = []
        for embedding in chunk_embeddings:
            new_row = input_data.copy()
            new_row["embedding"] = embedding
            duplicated_rows.append(new_row)
        return duplicated_rows

    duplicated_data_accidents = []
    for _, row in tqdm(
        accidents_df.iterrows(),
        desc="Generating embeddings and duplicating rows",
        total=len(accidents_df),
    ):
        duplicated_rows = get_embedding(row)
        duplicated_data_accidents.extend(duplicated_rows)

    accidents_df = pd.DataFrame(duplicated_data_accidents)

    duplicated_data_safey = []
    for _, row in tqdm(
        safety_df.iterrows(),
        desc="Generating embeddings and duplicating rows",
        total=len(safety_df),
    ):
        duplicated_rows = get_embedding(row)
        duplicated_data_safey.extend(duplicated_rows)

    safety_df = pd.DataFrame(duplicated_data_safey)

    accidents_df.head()

    safety_df.head()

    set_env_securely("MONGO_URI", "Enter your MongoDB URI: ")

    def get_mongo_client(mongo_uri):
        """Establish and validate connection to the MongoDB."""

        client = pymongo.MongoClient(
            mongo_uri, appname="devrel.showcase.factory_safety_assistant.python"
        )

        ping_result = client.admin.command("ping")
        if ping_result.get("ok") == 1.0:
            logger.debug("Connection to MongoDB successful")
            return client
        logger.debug("Connection to MongoDB failed")
        return None

    MONGO_URI = os.environ["MONGO_URI"]

    if not MONGO_URI:
        logger.debug("MONGO_URI not set in environment variables")

    mongo_client = get_mongo_client(MONGO_URI)

    DB_NAME = "factory_safety_use_case"
    SAFETY_PROCEDURES_COLLECTION = "safety_procedures"
    ACCIDENTS_REPORT_COLLECTION = "accident_report"

    db = mongo_client.get_database(DB_NAME)
    safety_procedure_collection = db.get_collection(
        SAFETY_PROCEDURES_COLLECTION)
    accident_report_collection = db.get_collection(ACCIDENTS_REPORT_COLLECTION)

    def setup_vector_search_index_with_filter(
        collection, index_definition, index_name="vector_index_with_filter"
    ):
        """
        Setup a vector search index for a MongoDB collection.

        Args:
        collection: MongoDB collection object
        index_definition: Dictionary containing the index definition
        index_name: Name of the index (default: "vector_index_with_filter")
        """
        new_vector_search_index_model = SearchIndexModel(
            definition=index_definition,
            name=index_name,
        )

        try:
            result = collection.create_search_index(
                model=new_vector_search_index_model)
            logger.debug(f"Creating index '{index_name}'...")
            logger.debug(
                f"New index '{index_name}' created successfully:", result)
        except Exception as e:
            logger.debug(
                f"Error creating new vector search index '{index_name}': {e!s}")

    vector_search_index_definition_safety_procedure = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "dimensions": 256,
                    "similarity": "cosine",
                    "type": "knnVector",
                },
                "procedureId": {"type": "string"},
            },
        }
    }

    vector_search_index_definition_accident_reports = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "dimensions": 256,
                    "similarity": "cosine",
                    "type": "knnVector",
                },
                "incidentId": {"type": "string"},
            },
        }
    }

    setup_vector_search_index_with_filter(
        safety_procedure_collection, vector_search_index_definition_safety_procedure
    )
    setup_vector_search_index_with_filter(
        accident_report_collection, vector_search_index_definition_accident_reports
    )

    safety_procedure_collection.delete_many({})
    accident_report_collection.delete_many({})

    def insert_df_to_mongodb(df, collection, batch_size=1000):
        """
        Insert a pandas DataFrame into a MongoDB collection.

        Parameters:
        df (pandas.DataFrame): The DataFrame to insert
        collection (pymongo.collection.Collection): The MongoDB collection to insert into
        batch_size (int): Number of documents to insert in each batch

        Returns:
        int: Number of documents successfully inserted
        """
        total_inserted = 0

        records = df.to_dict("records")

        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                total_inserted += len(result.inserted_ids)
                logger.debug(
                    f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} documents"
                )
            except BulkWriteError as bwe:
                total_inserted += bwe.details["nInserted"]
                logger.debug(
                    f"Batch {i//batch_size + 1} partially inserted. {bwe.details['nInserted']} inserted, {len(bwe.details['writeErrors'])} failed."
                )

        return total_inserted

    def print_dataframe_info(df, df_name):
        logger.debug(f"\n{df_name} DataFrame info:")
        logger.debug(df.info())
        logger.debug(f"\nFirst few rows of the {df_name} DataFrame:")
        logger.debug(df.head())

    try:
        total_inserted_safety = insert_df_to_mongodb(
            safety_df, safety_procedure_collection)
        logger.debug(
            f"Safety procedures data ingestion completed. Total documents inserted: {total_inserted_safety}"
        )
    except Exception as e:
        logger.debug(
            f"An error occurred while inserting safety procedures: {e}")
        logger.debug("Pandas version:", pd.__version__)
        print_dataframe_info(safety_df, "Safety Procedures")

    try:
        total_inserted_accidents = insert_df_to_mongodb(
            accidents_df, accident_report_collection
        )
        logger.debug(
            f"Accident reports data ingestion completed. Total documents inserted: {total_inserted_accidents}"
        )
    except Exception as e:
        logger.debug(
            f"An error occurred while inserting accident reports: {e}")
        logger.debug("Pandas version:", pd.__version__)
        print_dataframe_info(accidents_df, "Accident Reports")

    logger.debug("\nInsertion Summary:")
    logger.debug(
        f"Safety Procedures inserted: {total_inserted_safety if 'total_inserted_safety' in locals() else 'Failed'}"
    )
    logger.debug(
        f"Accident Reports inserted: {total_inserted_accidents if 'total_inserted_accidents' in locals() else 'Failed'}"
    )

    def vector_search(user_query, collection):
        """
        Perform a vector search in the MongoDB collection based on the user query.

        Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.

        Returns:
        list: A list of matching documents.
        """

        query_embedding = get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index_with_filter",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5,  # Return top 4 matches
            }
        }

        unset_stage = {
            "$unset": "embedding"  # Exclude the 'embedding' field from the results
        }

        project_stage = {
            "$project": {
                "_id": 0,  # Exclude the _id field,
                "combined_info": 1,
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                },
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_stage]

        results = collection.aggregate(pipeline)
        return list(results)

    def get_vector_search_result(query, collection):
        get_knowledge = vector_search(query, collection)
        search_results = []
        for result in get_knowledge:
            search_results.append(
                [result.get("score", "N/A"),
                 result.get("combined_info", "N/A")]
            )
        return search_results

    query = "Get me a saftey procedure related to helmet incidents"
    source_information = get_vector_search_result(
        query, safety_procedure_collection)

    table_headers = ["Similarity Score", "Combined Information"]
    table = tabulate.tabulate(
        source_information, headers=table_headers, tablefmt="grid")

    combined_information = f"""Query: {query}
    
    Continue to answer the query by using the Search Results:
    
    {table}
    """

    logger.debug(combined_information)

    # %pip install --quiet -U langchain langchain_mongodb langgraph langsmith motor jet.llm.ollama.base_langchain # langchain-groq

    # set_env_securely("ANTHROPIC_API_KEY", "Enter your Ollama API key: ")

    set_env_securely("GROQ_API_KEY", "Enter your Groq API key: ")

    def create_collection_search_index(collection, index_definition, index_name):
        """
        Create a search index for a MongoDB Atlas collection.

        Args:
        collection: MongoDB collection object
        index_definition: Dictionary defining the index mappings
        index_name: String name for the index

        Returns:
        str: Result of the index creation operation
        """

        try:
            search_index_model = SearchIndexModel(
                definition=index_definition, name=index_name
            )

            result = collection.create_search_index(model=search_index_model)
            logger.debug(f"Search index '{index_name}' created successfully")
            return result
        except Exception as e:
            logger.debug(f"Error creating search index: {e!s}")
            return None

    def print_collection_search_indexes(collection):
        """
        Print all search indexes for a given collection.

        Args:
        collection: MongoDB collection object
        """
        logger.debug(f"\nSearch indexes for collection '{collection.name}':")
        for index in collection.list_search_indexes():
            logger.debug(f"Index: {index['name']}")

    safety_procedure_collection_text_index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "category": {"type": "string"},
                "steps.description": {"type": "string"},
            },
        }
    }

    create_collection_search_index(
        safety_procedure_collection,
        safety_procedure_collection_text_index_definition,
        "text_search_index",
    )

    print_collection_search_indexes(safety_procedure_collection)

    accident_report_collection_text_index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {"type": {"type": "string"}, "description": {"type": "string"}},
        }
    }

    create_collection_search_index(
        accident_report_collection,
        accident_report_collection_text_index_definition,
        "text_search_index",
    )

    print_collection_search_indexes(accident_report_collection)

    ATLAS_VECTOR_SEARCH_INDEX = "vector_index_with_filter"
    embedding_model = OllamaEmbeddings(
        model=OPEN_AI_EMBEDDING_MODEL, dimensions=OPEN_AI_EMBEDDING_MODEL_DIMENSION
    )

    vector_store_safety_procedures = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGO_URI,
        namespace=DB_NAME + "." + SAFETY_PROCEDURES_COLLECTION,
        embedding=embedding_model,
        index_name=ATLAS_VECTOR_SEARCH_INDEX,
        text_key="combined_info",
    )

    hybrid_search = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store_safety_procedures,
        search_index_name="text_search_index",
        top_k=5,
    )

    hybrid_search_result = hybrid_search.get_relevant_documents(query)

    def hybrid_search_results_to_table(search_results):
        """
        Convert hybrid search results to a formatted markdown table.

        Args:
        search_results (list): List of Document objects containing search results

        Returns:
        str: Formatted markdown table of search results
        """
        data = []
        for rank, doc in enumerate(search_results, start=1):
            metadata = doc.metadata
            data.append(
                {
                    "Rank": rank,
                    "Procedure ID": metadata["procedureId"],
                    "Title": metadata["title"],
                    "Category": metadata["category"],
                    "Vector Score": round(metadata["vector_score"], 5),
                    "Full-text Score": round(metadata["fulltext_score"], 5),
                    "Total Score": round(metadata["score"], 5),
                }
            )

        df = pd.DataFrame(data)

        table = tabulate.tabulate(
            df, headers="keys", tablefmt="pipe", showindex=False)

        return table

    table = hybrid_search_results_to_table(hybrid_search_result)
    logger.debug(table)

    full_text_search = MongoDBAtlasFullTextSearchRetriever(
        collection=safety_procedure_collection,
        search_index_name="text_search_index",
        search_field="description",
        top_k=5,
    )
    full_text_search_result = full_text_search.get_relevant_documents(
        "Guidelines")

    logger.debug(full_text_search_result)

    """
    ## MongoDB Checkpointer
    """
    logger.info("## MongoDB Checkpointer")

    class JsonPlusSerializerCompat(JsonPlusSerializer):
        def loads(self, data: bytes) -> Any:
            if data.startswith(b"\x80") and data.endswith(b"."):
                return pickle.loads(data)
            return super().loads(data)

    class MongoDBSaver(AbstractContextManager, BaseCheckpointSaver):
        serde = JsonPlusSerializerCompat()

        client: AsyncIOMotorClient
        db_name: str
        collection_name: str

        def __init__(
            self,
            client: AsyncIOMotorClient,
            db_name: str,
            collection_name: str,
            *,
            serde: Optional[SerializerProtocol] = None,
        ) -> None:
            super().__init__(serde=serde)
            self.client = client
            self.db_name = db_name
            self.collection_name = collection_name
            self.collection = client[db_name][collection_name]

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            __exc_type: Optional[type[BaseException]],
            __exc_value: Optional[BaseException],
            __traceback: Optional[TracebackType],
        ) -> Optional[bool]:
            return True

        async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            if config["configurable"].get("thread_ts"):
                query = {
                    "thread_id": config["configurable"]["thread_id"],
                    "thread_ts": config["configurable"]["thread_ts"],
                }
            else:
                query = {"thread_id": config["configurable"]["thread_id"]}

            doc = await self.collection.find_one(query, sort=[("thread_ts", -1)])
            logger.success(format_json(doc))
            if doc:
                return CheckpointTuple(
                    config,
                    self.serde.loads(doc["checkpoint"]),
                    self.serde.loads(doc["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": doc["thread_id"],
                                "thread_ts": doc["parent_ts"],
                            }
                        }
                        if doc.get("parent_ts")
                        else None
                    ),
                )
            return None

        async def alist(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
        ) -> AsyncIterator[CheckpointTuple]:
            query = {}
            if config is not None:
                query["thread_id"] = config["configurable"]["thread_id"]
            if filter:
                for key, value in filter.items():
                    query[f"metadata.{key}"] = value
            if before is not None:
                query["thread_ts"] = {
                    "$lt": before["configurable"]["thread_ts"]}

            cursor = self.collection.find(query).sort("thread_ts", -1)
            if limit:
                cursor = cursor.limit(limit)

            async for doc in cursor:
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "thread_ts": doc["thread_ts"],
                        }
                    },
                    self.serde.loads(doc["checkpoint"]),
                    self.serde.loads(doc["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": doc["thread_id"],
                                "thread_ts": doc["parent_ts"],
                            }
                        }
                        if doc.get("parent_ts")
                        else None
                    ),
                )

        async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Optional[dict[str, Union[str, float, int]]],
        ) -> RunnableConfig:
            doc = {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["id"],
                "checkpoint": self.serde.dumps(checkpoint),
                "metadata": self.serde.dumps(metadata),
            }
            if config["configurable"].get("thread_ts"):
                doc["parent_ts"] = config["configurable"]["thread_ts"]
            await self.collection.insert_one(doc)
            return {
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "thread_ts": checkpoint["id"],
                }
            }

        def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            raise NotImplementedError(
                "Use aget_tuple for asynchronous operations")

        def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
        ):
            raise NotImplementedError("Use alist for asynchronous operations")

        def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
        ) -> RunnableConfig:
            raise NotImplementedError("Use aput for asynchronous operations")

        async def aput_writes(
            self,
            config: RunnableConfig,
            writes: List[Tuple[str, Any]],
            task_id: str,
        ) -> None:
            """Asynchronously store intermediate writes linked to a checkpoint."""
            docs = []
            for channel, value in writes:
                doc = {
                    "thread_id": config["configurable"]["thread_id"],
                    "task_id": task_id,
                    "channel": channel,
                    "value": self.serde.dumps(value),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                docs.append(doc)

            if docs:
                await self.collection.insert_many(docs)

    """
    ## Tool Definitions
    """
    logger.info("## Tool Definitions")

    @tool
    def safety_procedures_vector_search_tool(query: str, k: int = 5):
        """
        Perform a vector similarity search on safety procedures.

        Args:
            query (str): The search query string.
            k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: List of tuples (Document, score), where Document is a safety procedure
                  and score is the similarity score (lower is more similar).

        Note:
            Uses the global vector_store_safety_procedures for the search.
        """

        vector_search_results = vector_store_safety_procedures.similarity_search_with_score(
            query=query, k=k
        )
        return vector_search_results

    @tool
    def safety_procedures_full_text_search_tool(query: str, k: int = 5):
        """
        Perform a full-text search on safety procedures.

        Args:
            query (str): The search query string.
            k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: Relevant safety procedure documents matching the query.
        """

        full_text_search = MongoDBAtlasFullTextSearchRetriever(
            collection=safety_procedure_collection,
            search_index_name="text_search_index",
            search_field="description",
            top_k=k,
        )

        full_text_search_result = full_text_search.get_relevant_documents(
            query)

    @tool
    def safety_procedures_hybrid_search_tool(query: str):
        """
        Perform a hybrid (vector + full-text) search on safety procedures.

        Args:
            query (str): The search query string.

        Returns:
            list: Relevant safety procedure documents from hybrid search.

        Note:
            Uses both vector_store_safety_procedures and text_search_index.
        """

        hybrid_search = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store_safety_procedures,
            search_index_name="text_search_index",
            top_k=5,
        )

        hybrid_search_result = hybrid_search.get_relevant_documents(query)

        return hybrid_search_result

    class Step(BaseModel):
        stepNumber: int = Field(..., ge=1)
        description: str

    class SafetyProcedure(BaseModel):
        procedureId: str
        title: str
        description: str
        category: str
        steps: List[Step]
        lastUpdated: datetime = Field(default_factory=datetime.now)

    def create_safety_procedure_document(procedure_data: dict) -> dict:
        """
        Create a new safety procedure document from a dictionary, using Pydantic for validation.

        Args:
        procedure_data (dict): Dictionary representing the new safety procedure

        Returns:
        dict: Validated and formatted safety procedure document

        Raises:
        ValidationError: If the input data doesn't match the SafetyProcedure schema
        """
        try:
            safety_procedure = SafetyProcedure(**procedure_data)

            document = safety_procedure.dict()

            for i, step in enumerate(document["steps"], start=1):
                step["stepNumber"] = i

            return document
        except Exception as e:
            raise ValueError(f"Invalid safety procedure data: {e!s}")

    @tool
    def create_new_safety_procedures(new_procedure: dict):
        """
        Create and validate a new safety procedure document.

        Args:
            new_procedure (dict): Dictionary containing the new safety procedure data.

        Returns:
            dict: Validated and formatted safety procedure document.

        Raises:
            ValueError: If the input data is invalid or doesn't match the required schema.

        Note:
            Uses Pydantic for data validation via create_safety_procedure_document function.
        """
        new_safety_procedure_document = create_safety_procedure_document(
            new_procedure)
        return new_safety_procedure_document

    vector_store_accident_reports = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGO_URI,
        namespace=DB_NAME + "." + ACCIDENTS_REPORT_COLLECTION,
        embedding=embedding_model,
        index_name=ATLAS_VECTOR_SEARCH_INDEX,
        text_key="combined_info",
    )

    @tool
    def accident_reports_vector_search_tool(query: str, k: int = 5):
        """
        Perform a vector similarity search on accident reports.

        Args:
            query (str): The search query string.
            k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: List of tuples (Document, score), where Document is an accident report
                  and score is the similarity score (lower is more similar).

        Note:
            Uses the global vector_store_accident_reports for the search.
        """
        vector_search_results = vector_store_accident_reports.similarity_search_with_score(
            query=query, k=k
        )
        return vector_search_results

    @tool
    def accident_reports_full_text_search_tool(query: str, k: int = 5):
        """
        Perform a full-text search on accident reports.

        Args:
            query (str): The search query string.
            k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            list: Relevant accident report documents matching the query.
        """
        full_text_search = MongoDBAtlasFullTextSearchRetriever(
            collection=accident_report_collection,
            search_index_name="text_search_index",
            search_field="description",
            top_k=k,
        )

        return full_text_search.get_relevant_documents(query)

    @tool
    def accident_reports_hybrid_search_tool(query: str):
        """
        Perform a hybrid (vector + full-text) search on accident reports.

        Args:
            query (str): The search query string.

        Returns:
            list: Relevant accident report documents from hybrid search.

        Note:
            Uses both vector_store_accident_reports and accident_text_search_index.
        """
        hybrid_search = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store_accident_reports,
            search_index_name="text_search_index",
            top_k=5,
        )

        return hybrid_search.get_relevant_documents(query)

    @tool
    def create_new_accident_report(new_report: dict):
        """
        Create and validate a new accident report document.

        Args:
            new_report (dict): Dictionary containing the new accident report data.

        Returns:
            dict: Validated and formatted accident report document.

        Raises:
            ValueError: If the input data is invalid or doesn't match the required schema.

        Note:
            This function should implement proper validation and formatting for accident reports.
        """
        return new_report  # This should be replaced with actual implementation

    safety_procedure_collection_tools = [
        safety_procedures_vector_search_tool,
        safety_procedures_full_text_search_tool,
        safety_procedures_hybrid_search_tool,
        create_new_safety_procedures,
    ]

    accident_report_collection_tools = [
        accident_reports_vector_search_tool,
        accident_reports_full_text_search_tool,
        accident_reports_hybrid_search_tool,
        create_new_accident_report,
    ]

    """
    ## LLM Defintion
    """
    logger.info("## LLM Defintion")

    llm = ChatOllama(model="llama3.2")

    """
    ## Agent Definition
    """
    logger.info("## Agent Definition")

    def create_agent(llm, tools, system_message: str):
        """Create an agent."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "\nCurrent time: {time}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(time=lambda: str(datetime.now()))
        prompt = prompt.partial(tool_names=", ".join(
            [tool.name for tool in tools]))

        return prompt | llm.bind_tools(tools)

    toolbox = []

    toolbox.extend(safety_procedure_collection_tools)
    toolbox.extend(accident_report_collection_tools)

    chatbot_agent = create_agent(
        llm,
        toolbox,
        system_message="""
          You are an advanced Factory Safety Assistant Agent specializing in managing and providing information about safety procedures and accident reports in industrial settings. Your key responsibilities include:
    
          1. Searching and retrieving safety procedures and accident reports:
            - Use the provided search tools to find relevant safety procedures and accident reports based on user queries
            - Interpret and explain safety procedures and accident reports in detail
            - Provide context and additional information related to specific safety protocols and past incidents
    
          2. Creating new safety procedures and accident reports:
            - When provided with appropriate information, use the create_new_safety_procedures tool to generate new safety procedure documents
            - Use the create_new_accident_report tool to document new accidents or incidents
            - Ensure all necessary details are included in new procedures and reports
    
          3. Answering safety-related queries:
            - Respond to questions about safety protocols, best practices, regulations, and past incidents
            - Offer explanations and clarifications on complex safety issues
            - Provide step-by-step guidance on implementing safety procedures and handling incidents
    
          4. Assisting with safety compliance and incident prevention:
            - Help identify relevant safety procedures for specific tasks or situations
            - Advise on how to adhere to safety guidelines and regulations
            - Suggest improvements or updates to existing safety procedures based on past incidents
            - Analyze accident reports to identify trends and recommend preventive measures
    
          5. Supporting safety training and awareness:
            - Explain the importance and rationale behind safety procedures
            - Offer tips and best practices for maintaining a safe work environment
            - Help users understand the potential risks and consequences of not following safety procedures
            - Use past incident reports to illustrate the importance of safety measures
    
            6. Providing Structured Safety Advice:
       When users ask for safety procedures advice, provide information in the following structured format:
    
       Safety Procedure Advice:
       a. Relevant Procedure:
          - Title: [Procedure Title]
          - ID: [Procedure ID]
          - Description: [Brief description of the procedure]
          - Key Steps:
            1. [Step 1]
            2. [Step 2]
            3. [...]
    
       b. Related Incidents (Past 2 Years):
          - Incident 1:
            - IncidentID: [ID of the Incident document]
            - Date: [Date of incident]
            - Description: [Brief description of the incident]
            - Root Cause(s): [Identified root cause(s)]
          - Incident 2:
            - [Same structure as Incident 1]
          - [Additional incidents if applicable]
    
       c. Possible Root Causes:
          - [List of potential root causes based on the procedure and related incidents]
    
       d. Additional Safety Recommendations:
          - [Any extra safety tips or precautions based on the procedure and incident history]
    
       e. References:
          - Safety Procedure: [Reference to the specific safety procedure document]
          - Incident Reports: [References to the relevant incident reports]
    
    When providing this structured advice:
    - Use the safety procedure search tools to find the most relevant procedure.
    - Utilize the accident report search tools to identify related incidents from the past two years in the same region.
    - Analyze the incident reports to identify common or significant root causes.
    - Provide additional recommendations based on your analysis of both the procedure and the incident history.
    - Always include clear references to the source documents for both procedures and incident reports.
    
    
          When creating a new safety procedure, ensure you have all required information and use the create_new_safety_procedures tool. The required fields are:
          - procedureId
          - title
          - description
          - category
          - steps (a list of step objects, each with a stepNumber and description)
    
          When creating a new accident report, use the create_new_accident_report tool. Ensure you gather all necessary information about the incident.
    
          Provide detailed, accurate, and helpful information to support factory workers, managers, and safety officers in maintaining a safe work environment and properly documenting incidents. If you cannot find specific information or if the information requested is not available, clearly state this and offer to assist in creating a new procedure or report if appropriate.
    
          When discussing safety matters, always prioritize the well-being of workers and adherence to safety regulations. Use information from accident reports to reinforce the importance of following safety procedures and to suggest improvements in safety protocols.
    
          DO NOT MAKE UP ANY INFORMATION.
        """,
    )

    """
    ## State Definition
    """
    logger.info("## State Definition")

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        sender: str

    """
    ## Node Definition
    """
    logger.info("## Node Definition")

    def agent_node(state, agent, name):
        result = agent.invoke(state)
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(
                **result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            "sender": name,
        }

    chatbot_node = functools.partial(
        agent_node, agent=chatbot_agent, name="Factory Safety Assistant Agent( FSAA)"
    )
    tool_node = ToolNode(toolbox, name="tools")

    """
    ## Agentic Workflow Definition
    """
    logger.info("## Agentic Workflow Definition")

    workflow = StateGraph(AgentState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition, {
                                   "tools": "tools", END: END})

    workflow.add_edge("tools", "chatbot")

    mongo_client = AsyncIOMotorClient(MONGO_URI)
    mongodb_checkpointer = MongoDBSaver(mongo_client, DB_NAME, "state_store")

    graph = workflow.compile(checkpointer=mongodb_checkpointer)

    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        pass

    def sanitize_name(name: str) -> str:
        """Sanitize the name to match the pattern '^[a-zA-Z0-9_-]+$'."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    async def chat_loop():
        config = {"configurable": {"thread_id": "0"}}

        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "User: "
            )
            logger.success(format_json(user_input))
            if user_input.lower() in ["quit", "exit", "q"]:
                logger.debug("Goodbye!")
                break

            sanitized_name = (
                sanitize_name("Human") or "Anonymous"
            )  # Fallback if sanitized name is empty
            state = {"messages": [HumanMessage(
                content=user_input, name=sanitized_name)]}

            logger.debug("Assistant: ", end="", flush=True)

            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    for chunk in graph.stream(state, config, stream_mode="values"):
                        if chunk.get("messages"):
                            last_message = chunk["messages"][-1]
                            if isinstance(last_message, AIMessage):
                                last_message.name = (
                                    sanitize_name(
                                        last_message.name or "AI") or "AI"
                                )
                                logger.debug(last_message.content,
                                             end="", flush=True)
                        elif isinstance(last_message, ToolMessage):
                            logger.debug(f"\n[Tool Used: {last_message.name}]")
                            logger.debug(
                                f"Tool Call ID: {last_message.tool_call_id}")
                            logger.debug(f"Content: {last_message.content}")
                            logger.debug("Assistant: ", end="", flush=True)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"\nAn unexpected error occurred: {e!s}")
                        logger.debug(f"\nRetrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.debug(
                            f"\nMax retries reached. Ollama API error: {e!s}")
                        break

            logger.debug("\n")  # New line after the complete response

    # import nest_asyncio

    # nest_asyncio.apply()

    await chat_loop()

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
