from IPython.display import Image, display
from datetime import datetime
from jet.logger import CustomLogger
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from ollama import Ollama
from pydantic import BaseModel, Field
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from typing import Annotated, Dict, List
from typing_extensions import TypedDict
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured_ingest.v2.processes.connectors.fsspec.s3 import (
S3AccessConfig,
S3ConnectionConfig,
S3DownloaderConfig,
S3IndexerConfig,
)
from unstructured_ingest.v2.processes.connectors.local import LocalUploaderConfig
from unstructured_ingest.v2.processes.connectors.mongodb import (
MongoDBAccessConfig,
MongoDBConnectionConfig,
MongoDBUploaderConfig,
MongoDBUploadStagerConfig,
)
from unstructured_ingest.v2.processes.embedder import EmbedderConfig
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
import json
import os
import re
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/self_querying_mongodb_unstructured_langgraph.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/advanced-rag-self-querying-retrieval/?utm_campaign=devrel&utm_source=cross-post&utm_medium=organic_social&utm_content=https%3A%2F%2Fgithub.com%2Fmongodb-developer%2FGenAI-Showcase&utm_term=apoorva.joshi)

# Building an Advanced RAG System with Self-Querying Retrieval

This notebook shows how to incorporate self-querying retrieval into a RAG application using Unstructured, MongoDB and LangGraph.

## Step 1: Install required libraries

- **langgraph**: Python package to build stateful, multi-actor applications with LLMs
<p>
- **ollama**: Python package to interact with Ollama APIs
<p>
- **pymongo**: Python package to interact with MongoDB databases and collections
<p>
- **sentence-transformers**: Python package for open-source language models
<p>
- **unstructured-ingest**: Python package for data processing using Unstructured
"""
logger.info("# Building an Advanced RAG System with Self-Querying Retrieval")

# !pip install -qU langgraph ollama pymongo sentence-transformers "unstructured-ingest[pdf, s3, mongodb, embed-huggingface]"


warnings.filterwarnings("ignore", category=UserWarning)

"""
## Step 2: Setup prerequisites

- **Set the Unstructured API key and URL**: Steps to obtain the API key and URL are [here](https://unstructured.io/api-key-hosted)

- **Set the AWS access keys**: Steps to obtain the AWS access keys are [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)

- **Set the MongoDB connection string**: Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

- **Set the Ollama API key**: Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Step 2: Setup prerequisites")



UNSTRUCTURED_API_KEY = ""
UNSTRUCTURED_URL = ""

AWS_KEY = ""
AWS_SECRET = ""

AWS_S3_NAME = ""

MONGODB_URI = ""
MONGODB_DB_NAME = ""
MONGODB_COLLECTION = ""
mongodb_client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.selfquery_mongodb_unstructured"
)

# os.environ["OPENAI_API_KEY"] = ""
ollama_client = Ollama()

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

COMPLETION_MODEL_NAME = "gpt-4o-2024-08-06"

"""
## Step 3: Partition, chunk and embed PDF files

Let's set up the PDF preprocessing pipeline with Unstructured. The pipeline will:
1. Ingest data from an S3 bucket/local directory
2. Partition documents: extract text and metadata, split the documents into document elements, such as titles, paragraphs (narrative text), tables, images, lists, etc. Learn more about document elements in [Unstructured documentation])https://docs.unstructured.io/api-reference/api-services/document-elements).
3. Chunk the documents.
4. Embed the documents with the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) embedding model the Hugging Face Hub.
5. Save the results locally.
"""
logger.info("## Step 3: Partition, chunk and embed PDF files")



WORK_DIR = "/content/temp"

Pipeline.from_configs(
    context=ProcessorConfig(
        verbose=True, tqdm=True, num_processes=5, work_dir=WORK_DIR
    ),
    indexer_config=S3IndexerConfig(remote_url=AWS_S3_NAME),
    downloader_config=S3DownloaderConfig(),
    source_connection_config=S3ConnectionConfig(
        access_config=S3AccessConfig(key=AWS_KEY, secret=AWS_SECRET)
    ),
    partitioner_config=PartitionerConfig(
        partition_by_api=True,
        api_key=UNSTRUCTURED_API_KEY,
        partition_endpoint=UNSTRUCTURED_URL,
        strategy="hi_res",
        additional_partition_args={
            "split_pdf_page": True,
            "split_pdf_allow_failed": True,
            "split_pdf_concurrency_level": 15,
        },
    ),
    chunker_config=ChunkerConfig(
        chunking_strategy="by_title",
        chunk_max_characters=1500,
        chunk_overlap=150,
    ),
    embedder_config=EmbedderConfig(
        embedding_provider="langchain-huggingface",
        embedding_model_name=EMBEDDING_MODEL_NAME,
    ),
    uploader_config=LocalUploaderConfig(output_dir="/content/ingest-outputs"),
).run()

"""
## Step 4: Add custom metadata to the processed documents

For each document, we want to add the company name and fiscal year as custom metadata, to enable smart pre-filtering for more precise document retrieval.

Luckily the Form-10K documents have a more or less standard page with this information, so we can use regex to extract this information.
"""
logger.info("## Step 4: Add custom metadata to the processed documents")


def get_fiscal_year(elements: dict) -> int:
    """
    Extract fiscal year from document elements.

    Args:
        elements (dict): Document elements

    Returns:
        int: Year
    """
    pattern = r"for the (fiscal\s+)?year ended.*?(\d{4})"
    year = 0
    for i in range(len(elements)):
        match = re.search(pattern, elements[i]["text"], re.IGNORECASE)
        if match:
            year = match.group(0)[-4:]
            try:
                year = int(year)
            except Exception:
                year = 0
    return year

def get_company_name(elements: dict) -> str:
    """
    Extract company name from document elements.

    Args:
        elements (dict): Document elements

    Returns:
        str: Company name
    """
    name = ""
    substring = "(Exact name of registrant as specified"
    for i in range(len(elements)):
        if substring.lower() in elements[i]["text"].lower():
            pattern = (
                r"([A-Z][A-Za-z\s&.,]+?)\s*\(Exact name of registrant as specified"
            )
            match = re.search(pattern, elements[i]["text"], re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = name.split("\n\n")[-1]

    if name == "":
        for i in range(len(elements)):
            match = re.search(
                r"Exact name of registrant as specified in its charter:\n\n(.*?)\n\n",
                elements[i]["text"],
            )
            if match:
                name = match.group(1)
            else:
                match = re.search(
                    r"Commission File Number.*\n\n(.*?)\n\n", elements[i]["text"]
                )
                if match:
                    name = match.group(1)
    return name

"""
We'll walk through the directory with the embedding results, and for each document find the company name and year, and add it as custom metadata to all elements of the document.
"""
logger.info("We'll walk through the directory with the embedding results, and for each document find the company name and year, and add it as custom metadata to all elements of the document.")

directory = f"{WORK_DIR}/embed"

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        logger.debug(f"Processing file {filename}")
        try:
            with open(file_path) as file:
                data = json.load(file)

            company_name = get_company_name(data)
            fiscal_year = get_fiscal_year(data)

            for entry in data:
                entry["metadata"]["custom_metadata"] = {}
                entry["metadata"]["custom_metadata"]["company"] = company_name
                entry["metadata"]["custom_metadata"]["year"] = fiscal_year

            with open(file_path, "w") as file:
                json.dump(data, file, indent=2)

            logger.debug(f"Successfully updated {file_path} with custom metadata fields.")
        except json.JSONDecodeError as e:
            logger.debug(f"Error parsing JSON in {file_path}: {e}")
        except OSError as e:
            logger.debug(f"Error reading from or writing to {file_path}: {e}")

"""
## Step 5: Write the processed documents to MongoDB

To write the final processed documents to MongoDB, we will need to rerun the same pipeline, except we'll now change the destination from local to MongoDB.
The pipeline will not repeat partitioning, chunking and embedding steps, since there are results for them already cached in the `WORK_DIR`. It will pick up the customized embedding results and load them into a MongoDB collection.
"""
logger.info("## Step 5: Write the processed documents to MongoDB")


Pipeline.from_configs(
    context=ProcessorConfig(
        verbose=True, tqdm=True, num_processes=5, work_dir=WORK_DIR
    ),
    indexer_config=S3IndexerConfig(remote_url=AWS_S3_NAME),
    downloader_config=S3DownloaderConfig(),
    source_connection_config=S3ConnectionConfig(
        access_config=S3AccessConfig(key=AWS_KEY, secret=AWS_SECRET)
    ),
    partitioner_config=PartitionerConfig(
        partition_by_api=True,
        api_key=UNSTRUCTURED_API_KEY,
        partition_endpoint=UNSTRUCTURED_URL,
        strategy="hi_res",
        additional_partition_args={
            "split_pdf_page": True,
            "split_pdf_allow_failed": True,
            "split_pdf_concurrency_level": 15,
        },
    ),
    chunker_config=ChunkerConfig(
        chunking_strategy="by_title",
        chunk_max_characters=1500,
        chunk_overlap=150,
    ),
    embedder_config=EmbedderConfig(
        embedding_provider="langchain-huggingface",
        embedding_model_name=EMBEDDING_MODEL_NAME,
    ),
    destination_connection_config=MongoDBConnectionConfig(
        access_config=MongoDBAccessConfig(uri=MONGODB_URI),
        collection=MONGODB_COLLECTION,
        database=MONGODB_DB_NAME,
    ),
    stager_config=MongoDBUploadStagerConfig(),
    uploader_config=MongoDBUploaderConfig(batch_size=100),
).run()

"""
Next, we are going to use LangGraph to build our investment assistant. With LangGraph, we can build LLM systems as graphs with a shared state, conditional edges, and cyclic loops between nodes.

## Step 6: Define graph state

Let's first define the state of our graph. The state is a mutable object that tracks different attributes as we pass through the nodes in the graph. We can include custom attributes within the state that represent parameters we want to track.
"""
logger.info("## Step 6: Define graph state")



class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: User query
        metadata: Extracted metadata
        filter: Filter definition
        documents: List of retrieved documents from vector search
        memory: Conversational history
    """

    question: str
    metadata: Dict
    filter: Dict
    context: List[str]
    memory: Annotated[list, add_messages]

"""
## Step 7: Define graph nodes

Next, let's add the graph nodes. Nodes in LangGraph are functions or tools that your system has access to in order to complete the task. Each node updates one or more attributes in the graph state with its return value after it executes. Our assistant has four nodes:
1. **Metadata Extractor**: Extract metadata from a natural language query
2. **Filter Generator**: Generate a MongoDB Query API filter definition
3. **MongoDB Atlas Vector Search**: Retrieve documents from MongoDB using semantic search
4. **Answer Generator**: Generate an answer to the user question

### Metadata Extractor
"""
logger.info("## Step 7: Define graph nodes")



companies = [
    "AT&T INC.",
    "American International Group, Inc.",
    "Apple Inc.",
    "BERKSHIRE HATHAWAY INC.",
    "Bank of America Corporation",
    "CENCORA, INC.",
    "CVS HEALTH CORPORATION",
    "Cardinal Health, Inc.",
    "Chevron Corporation",
    "Citigroup Inc.",
    "Costco Wholesale Corporation",
    "Exxon Mobil Corporation",
    "Ford Motor Company",
    "GENERAL ELECTRIC COMPANY",
    "GENERAL MOTORS COMPANY",
    "HP Inc.",
    "INTERNATIONAL BUSINESS MACHINES CORPORATION",
    "JPMorgan Chase & Co.",
    "MICROSOFT CORPORATION",
    "MIDLAND COMPANY",
    "McKESSON CORPORATION",
    "THE BOEING COMPANY",
    "THE HOME DEPOT, INC.",
    "THE KROGER CO.",
    "The Goldman Sachs Group, Inc.",
    "UnitedHealth Group Incorporated",
    "VALERO ENERGY CORPORATION",
    "Verizon Communications Inc.",
    "WALMART INC.",
    "WELLS FARGO & COMPANY",
]

class Metadata(BaseModel):
    """Metadata to use for pre-filtering."""

    company: List[str] = Field(description="List of company names")
    year: List[str] = Field(description="List containing start year and end year")

def extract_metadata(state: Dict) -> Dict:
    """
    Extract metadata from natural language query.

    Args:
        state (Dict): The current graph state

    Returns:
        Dict: New key added to state i.e. metadata containing the metadata extracted from the user query.
    """
    logger.debug("---EXTRACTING METADATA---")
    question = state["question"]
    system = f"""Extract the specified metadata from the user question:
    - company: List of company names, eg: Google, Adobe etc. Match the names to companies on this list: {companies}
    - year: List of [start year, end year]. Guidelines for extracting dates:
        - If a single date is found, only include that.
        - For phrases like 'in the past X years/last year', extract the start year by subtracting X from the current year. The current year is {datetime.now().year}.
        - If more than two dates are found, only include the smallest and the largest year."""
    completion = ollama_client.beta.chat.completions.parse(
        model=COMPLETION_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        response_format=Metadata,
    )
    result = completion.choices[0].message.parsed
    if len(result.company) == 0 and len(result.year) == 0:
        return {"metadata": {}}
    metadata = {
        "metadata.custom_metadata.company": result.company,
        "metadata.custom_metadata.year": result.year,
    }
    return {"metadata": metadata}

"""
### Filter Generator
"""
logger.info("### Filter Generator")

def generate_filter(state: Dict) -> Dict:
    """
    Generate MongoDB Query API filter definition.

    Args:
        state (Dict): The current graph state

    Returns:
        Dict: New key added to state i.e. filter.
    """
    logger.debug("---GENERATING FILTER DEFINITION---")
    metadata = state["metadata"]
    system = """Generate a MongoDB filter definition from the provided fields. Follow the guidelines below:
    - Respond in JSON with the filter assigned to a `filter` key.
    - The field `metadata.custom_metadata.company` is a list of companies.
    - The field `metadata.custom_metadata.year` is a list of one or more years.
    - If any of the provided fields are empty lists, DO NOT include them in the filter.
    - If both the metadata fields are empty lists, return an empty dictionary {{}}.
    - The filter should only contain the fields `metadata.custom_metadata.company` and `metadata.custom_metadata.year`
    - The filter can only contain the following MongoDB Query API match expressions:
        - $gt: Greater than
        - $lt: Lesser than
        - $gte: Greater than or equal to
        - $lte: Less than or equal to
        - $eq: Equal to
        - $ne: Not equal to
        - $in: Specified field value equals any value in the specified array
        - $nin: Specified field value is not present in the specified array
        - $nor: Logical NOR operation
        - $and: Logical AND operation
        - $or: Logical OR operation
    - If the `metadata.custom_metadata.year` field has multiple dates, create a date range filter using expressions such as $gt, $lt, $lte and $gte
    - If the `metadata.custom_metadata.company` field contains a single company, use the $eq expression
    - If the `metadata.custom_metadata.company` field contains multiple companies, use the $in expression
    - To combine date range and company filters, use the $and operator
    """
    completion = ollama_client.chat.completions.create(
        model=COMPLETION_MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Fields: {metadata}"},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(completion.choices[0].message.content)
    return {"filter": result.get("filter", {})}

"""
### MongoDB Atlas Vector Search
"""
logger.info("### MongoDB Atlas Vector Search")


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
collection = mongodb_client[MONGODB_DB_NAME][MONGODB_COLLECTION]
VECTOR_SEARCH_INDEX_NAME = "vector_index"

model = {
    "name": VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embeddings",
                "numDimensions": 768,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "metadata.custom_metadata.company"},
            {"type": "filter", "path": "metadata.custom_metadata.year"},
        ]
    },
}
collection.create_search_index(model=model)

def vector_search(state: Dict) -> Dict:
    """
    Get relevant information using MongoDB Atlas Vector Search

    Args:
        state (Dict): The current graph state

    Returns:
        Dict: New key added to state i.e. documents.
    """
    logger.debug("---PERFORMING VECTOR SEARCH---")
    question = state["question"]
    filter = state["filter"]
    if not filter:
        filter = {}
    query_embedding = embedding_model.encode(question).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_SEARCH_INDEX_NAME,
                "path": "embeddings",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
                "filter": filter,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = collection.aggregate(pipeline)
    relevant_results = [doc["text"] for doc in results if doc["score"] >= 0.8]
    context = "\n\n".join([doc for doc in relevant_results])
    return {"context": context}

"""
### Answer Generator
"""
logger.info("### Answer Generator")


def generate_answer(state: Dict) -> Dict:
    """
    Generate the final answer to the user query

    Args:
        state (Dict): The current graph state

    Returns:
        Dict: New key added to state i.e. generation.
    """
    logger.debug("---GENERATING THE ANSWER---")
    question = state["question"]
    context = state["context"]
    memory = state["memory"]
    system = "Answer the question based only on the following context. If the context is empty or if it doesn't provide enough information to answer the question, say I DON'T KNOW"
    completion = ollama_client.chat.completions.create(
        model=COMPLETION_MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\n{memory}\n\nQuestion:{question}",
            },
        ],
    )
    answer = completion.choices[0].message.content
    return {"memory": [HumanMessage(content=context), AIMessage(content=answer)]}

"""
## Step 8: Define conditional edges

Conditional edges in LangGraph decide which node in the graph to visit next. Here, we have a single conditional edge to skip filter generation and go directly to the vector search step if no metadata was extracted from the user query.
"""
logger.info("## Step 8: Define conditional edges")

def check_metadata_extracted(state: Dict) -> str:
    """
    Check if any metadata is extracted.

    Args:
        state (Dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.debug("---CHECK FOR METADATA---")
    metadata = state["metadata"]
    if not metadata:
        logger.debug("---DECISION: SKIP TO VECTOR SEARCH---")
        return "vector_search"
    logger.debug("---DECISION: GENERATE FILTER---")
    return "generate_filter"

"""
## Step 9: Build the graph/flow

This is where we actually define the flow of the graph by connecting nodes to edges.
"""
logger.info("## Step 9: Build the graph/flow")


workflow = StateGraph(GraphState)
memory = MemorySaver()

workflow.add_node("extract_metadata", extract_metadata)
workflow.add_node("generate_filter", generate_filter)
workflow.add_node("vector_search", vector_search)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "extract_metadata")
workflow.add_conditional_edges(
    "extract_metadata",
    check_metadata_extracted,
    {
        "vector_search": "vector_search",
        "generate_filter": "generate_filter",
    },
)
workflow.add_edge("generate_filter", "vector_search")
workflow.add_edge("vector_search", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile(checkpointer=memory)

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass

"""
## Step 10: Execute the graph
"""
logger.info("## Step 10: Execute the graph")

def execute_graph(thread_id: str, question: str) -> None:
    """
    Execute the graph and stream its output

    Args:
        thread_id (str): Conversation thread ID
        question (str): User question
    """
    inputs = {"question": question, "memory": [HumanMessage(content=question)]}
    config = {"configurable": {"thread_id": thread_id}}
    for output in app.stream(inputs, config):
        for key, value in output.items():
            logger.debug(f"Node {key}:")
            logger.debug(value)
    logger.debug("---FINAL ANSWER---")
    logger.debug(value["memory"][-1].content)

execute_graph("1", "Sales summary for Walmart for 2023.")

execute_graph("1", "What did I just ask you?")

execute_graph("1", "What's my name?")

logger.info("\n\n[DONE]", bright=True)