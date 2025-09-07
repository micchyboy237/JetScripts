from jet.logger import CustomLogger
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import group_broken_paragraphs
from unstructured.partition.text import partition_text
import os
import pandas as pd
import pymongo
import requests
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Optimizing Vector Database Performance: Reducing Retrieval Latency with Quantization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/automatic_quantization_of_nomic_emebddings_with_mongodb.ipynb)

---

**Summary**

This notebook explores techniques for optimizing vector database performance, focusing on reducing retrieval latency through the use of quantization methods. We examine the practical application of various embedding types
- float32_embedding
- int8_embedding
- binary_embedding

We analyze their impact on query precision and retrieval speed.

By leveraging quantization strategies like scalar and binary quantization, we highlight the trade-offs between precision and efficiency.

The notebook also includes a step-by-step demonstration of executing vector searches, measuring retrieval latencies, and visualizing results in a comparative framework.

**Use Case:**

The notebook demonstrates how to optimize vector database performance, specifically focusing on reducing retrieval latency using quantization methods.

**Scenario**:
You have a large dataset of text data (in this case, a book from Gutenberg) and you want to build a system that can efficiently find similar pieces of text based on a user's query.

**Approach**:
- Embeddings: The notebook uses SentenceTransformer to convert text into numerical vectors (embeddings) which capture the semantic meaning of the text.
- Vector Database: MongoDB is used as a vector database to store and search these embeddings efficiently.
- Quantization: To speed up retrieval, the notebook applies quantization techniques (scalar and binary) to the embeddings. This reduces the size of the embeddings, making searches faster but potentially impacting precision.
Goal: By comparing the performance of different embedding types (float32, int8, binary), the notebook aims to show the trade-offs between retrieval speed and accuracy when using quantization. This helps in choosing the best approach for a given use case.

## Step 1: Install Libaries

Here's a breakdown of the libraries and their roles:

- **unstructured**: This library is used to process and structure various data formats, including text, enabling efficient analysis and extraction of information.
- **pymongo**: This library provides the tools necessary to interact with MongoDB allowing for storage and retrieval of data within the project.
- **nomic**: This library is used for vector embedding and other functions related to Nomic AI's models, specifically for generating and working with text embeddings.
- **pandas**: This popular library is used for data manipulation and analysis, providing data structures and functions for efficient data handling and exploration.
- **sentence_transformers**: This library is used for generating embeddings for text data using the SentenceTransformer model.

By installing these packages, the code sets up the tools necessary for data processing, embedding generation, and storage with MongoDB.
"""
logger.info("# Optimizing Vector Database Performance: Reducing Retrieval Latency with Quantization")

# %pip install --quiet -U unstructured pymongo nomic pandas sentence_transformers einops

# import getpass


def set_env_securely(var_name, prompt):
#     value = getpass.getpass(prompt)
    os.environ[var_name] = value

"""
## Step 2: Data Loading and Preparation

**Dataset Information**

The dataset used in this example is "Pushing to the Front," an ebook from Project Gutenberg. This book, focusing on self-improvement and success, is freely available for public use.

The code leverages the ```unstructured``` library to process this raw text data, transforming it into a structured format suitable for semantic analysis and search. By chunking the text based on titles, the code creates meaningful units that can be embedded and stored in a vector database for efficient retrieval. This preprocessing is essential for building a robust and performant semantic search system.

The code below ```requests``` library to fetch the text content of the book "Pushing to the Front" from Project Gutenberg's website. The URL points to the raw text file of the book.
"""
logger.info("## Step 2: Data Loading and Preparation")


url = "https://www.gutenberg.org/cache/epub/21291/pg21291.txt"
response = requests.get(url)
response.raise_for_status()
book_text = response.text

"""
Data Cleaning: The ```unstructured``` library is used to clean and structure the raw text. The ```group_broken_paragraphs``` function helps in combining fragmented paragraphs, ensuring better text flow.
"""
logger.info("Data Cleaning: The ```unstructured``` library is used to clean and structure the raw text. The ```group_broken_paragraphs``` function helps in combining fragmented paragraphs, ensuring better text flow.")


cleaned_text = group_broken_paragraphs(book_text)

parsed_sections = partition_text(text=cleaned_text)

"""
The ```partition_text``` function further processes the cleaned text, dividing it into logical sections. These sections could represent chapters, sub-sections, or other meaningful units within the book.
"""
logger.info("The ```partition_text``` function further processes the cleaned text, dividing it into logical sections. These sections could represent chapters, sub-sections, or other meaningful units within the book.")

for text in parsed_sections[:5]:
    logger.debug(text)
    logger.debug("\n")

"""
Chunking by Title: The ```chunk_by_title``` function identifies titles or headings within the parsed sections and uses them to create distinct chunks of text. This step is crucial for organizing the data into manageable units for subsequent embedding generation and semantic search.
"""
logger.info("Chunking by Title: The ```chunk_by_title``` function identifies titles or headings within the parsed sections and uses them to create distinct chunks of text. This step is crucial for organizing the data into manageable units for subsequent embedding generation and semantic search.")


chunks = chunk_by_title(parsed_sections)

for chunk in chunks:
    logger.debug(chunk)
    break

"""
## Step 3: Embeddings Generation
"""
logger.info("## Step 3: Embeddings Generation")


embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
)

max_seq_length = embedding_model.max_seq_length


def chunk_text(text, tokenizer, max_length=8192, overlap=50):
    """
    Split the text into overlapping chunks based on token length.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk_tokens = tokens[i : i + max_length]
        chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk)
    return chunks


def get_embedding(text, task_prefix):
    """
    Generate embeddings for a text string with a task-specific prefix.
    """

    if not text.strip():
        logger.debug("Attempted to get embedding for empty text.")
        return []

    prefixed_text = f"{task_prefix}: {text}"

    tokenizer = embedding_model.tokenizer

    chunks = chunk_text(prefixed_text, tokenizer, max_length=max_seq_length)

    chunk_embeddings = embedding_model.encode(chunks)

    return chunk_embeddings[0].tolist()

"""
The embedding generation might take a approximately 20 minutes
"""
logger.info("The embedding generation might take a approximately 20 minutes")


embeddings = []
for chunk in tqdm(chunks, desc="Generating embeddings"):
    embedding = get_embedding(str(chunk), task_prefix="search_document")
    embeddings.append(embedding)

embedding_data = []
for chunk, embedding in zip(chunks, embeddings):
    embedding_data.append(
        {
            "chunk": chunk.text,
            "float32_embedding": embedding,
            "int8_embedding": embedding,
            "binary_embedding": embedding,
        }
    )


dataset_df = pd.DataFrame(embedding_data)

"""
When visualizing the dataset values, you will observe that the embedding attributes: float32_embedding, int_embedding and binary_emebedding all have the same values.

In downstream proceses the values of the int_embedding and binary_embedding attributes for each data point will be modified to their respective data types, as a result of MongoDB Atlas auto quantization feature.
"""
logger.info("When visualizing the dataset values, you will observe that the embedding attributes: float32_embedding, int_embedding and binary_emebedding all have the same values.")

dataset_df.head()

"""
## Step 4: MongoDB (Operational and Vector Database)

MongoDB acts as both an operational and vector database for the RAG system.
MongoDB Atlas specifically provides a database solution that efficiently stores, queries and retrieves vector embeddings.

Creating a database and collection within MongoDB is made simple with MongoDB Atlas.

1. First, register for a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register). For existing users, sign into MongoDB Atlas.
2. [Follow the instructions](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/). Select Atlas UI as the procedure to deploy your first cluster.

Follow MongoDB’s [steps to get the connection](https://www.mongodb.com/docs/manual/reference/connection-string/) string from the Atlas UI. After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.
"""
logger.info("## Step 4: MongoDB (Operational and Vector Database)")

set_env_securely("MONGO_URI", "Enter your MONGO URI: ")



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.quantized_embeddings_nomic.python"
    )

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    else:
        logger.debug("Connection to MongoDB failed")
    return None


MONGO_URI = os.environ["MONGO_URI"]
if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")


mongo_client = get_mongo_client(MONGO_URI)

DB_NAME = "career_coach"
COLLECTION_NAME = "pushing_to_the_front_orison_quantized"

db = mongo_client[DB_NAME]

if COLLECTION_NAME not in db.list_collection_names():
    try:
        db.create_collection(COLLECTION_NAME)
        logger.debug(f"Collection '{COLLECTION_NAME}' created successfully.")
    except CollectionInvalid as e:
        logger.debug(f"Error creating collection: {e}")
else:
    logger.debug(f"Collection '{COLLECTION_NAME}' already exists.")

collection = db[COLLECTION_NAME]

"""
## Step 5: Data Ingestion
"""
logger.info("## Step 5: Data Ingestion")

collection.delete_many({})

documents = dataset_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## Step 6: Vector Search Index Creation
"""
logger.info("## Step 6: Vector Search Index Creation")




def setup_vector_search_index(collection, index_definition, index_name="vector_index"):
    """
    Setup a vector search index for a MongoDB collection and wait for 30 seconds.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index")
    """
    new_vector_search_index_model = SearchIndexModel(
        definition=index_definition, name=index_name, type="vectorSearch"
    )

    try:
        result = collection.create_search_index(model=new_vector_search_index_model)
        logger.debug(f"Creating index '{index_name}'...")

        logger.debug(f"Waiting for 60 seconds to allow index '{index_name}' to be created...")
        time.sleep(60)

        logger.debug(f"60-second wait completed for index '{index_name}'.")
        return result

    except Exception as e:
        logger.debug(f"Error creating new vector search index '{index_name}': {e!s}")
        return None

def create_vector_index_definition():
    """
    Create a vector index definition with predefined quantization methods.

    This function defines vector index fields with specific paths, dimensionalities,
    and similarity metrics. It includes support for quantization methods:
    - "scalar" quantization is applied to the "int8_embedding" field.
    - "binary" quantization is applied to the "binary_embedding" field.
    - No quantization is applied to the "float32_embedding" field.

    Returns:
      dict: A dictionary containing the vector index definition, including
      fields with their respective paths, quantization methods, dimensions,
      and similarity measures.
    """

    base_fields = [
        {
            "type": "vector",
            "path": "float32_embedding",
            "numDimensions": 768,
            "similarity": "cosine",
        },
        {
            "type": "vector",
            "path": "int8_embedding",
            "quantization": "scalar",
            "numDimensions": 768,
            "similarity": "cosine",
        },
        {
            "type": "vector",
            "path": "binary_embedding",
            "quantization": "binary",
            "numDimensions": 768,
            "similarity": "euclidean",
        },
    ]

    return {"fields": base_fields}

vector_index_definition = create_vector_index_definition()

logger.debug(vector_index_definition)

setup_vector_search_index(collection, vector_index_definition, "vector_index")

"""
## Step 7: Vector Search Operation
"""
logger.info("## Step 7: Vector Search Operation")

def custom_vector_search(
    user_query, collection, embedding_path, vector_search_index_name="vector_index"
):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.
        embedding_path (str): The path of the embedding field in the documents.
        vector_search_index_name (str): The name of the vector search index.

    Returns:
        list: A list of matching documents.
    """

    query_embedding = get_embedding(user_query, task_prefix="search_query")

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_search_index_name,  # Specifies the index to use for the search
            "queryVector": query_embedding,  # The vector representing the query
            "path": embedding_path,  # Field in the documents containing the vectors to search against
            "numCandidates": 1000,  # Number of candidate matches to consider
            "limit": 10,  # Return top 5 matches
        }
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "chunk": 1,
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            },
        }
    }

    pipeline = [vector_search_stage, project_stage]

    explain_result = collection.database.command(
        "explain",
        {"aggregate": collection.name, "pipeline": pipeline, "cursor": {}},
        verbosity="executionStats",
    )

    vector_search_explain = explain_result["stages"][0]["$vectorSearch"]
    execution_time_ms = vector_search_explain["explain"]["query"]["stats"]["context"][
        "millisElapsed"
    ]

    results = list(collection.aggregate(pipeline))

    return results, execution_time_ms

def run_vector_search_operations(
    user_query, collection, vector_search_index_name="vector_index"
):
    """
    Run vector search operations for different embedding paths and store results in a DataFrame.
    """
    embedding_paths = ["float32_embedding", "int8_embedding", "binary_embedding"]
    results_data = []

    for path in embedding_paths:
        try:
            results, execution_time_ms = custom_vector_search(
                user_query=user_query,
                collection=collection,
                embedding_path=path,
                vector_search_index_name=vector_search_index_name,
            )

            formatted_results = "\n".join(
                [f"[{result['score']:.4f}] {result['chunk']}" for result in results]
            )

            results_data.append(
                {
                    "Precision (Data Type)": path.split("_")[0],
                    "Retrieval Latency (ms)": f"{execution_time_ms:.6f}",
                    "Results": formatted_results,
                }
            )

        except Exception as e:
            results_data.append(
                {
                    "Precision (Data Type)": path.split("_")[0],
                    "Retrieval Latency (ms)": "Error",
                    "Results": str(e),
                }
            )

    results_df = pd.DataFrame(results_data)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    return results_df

"""
## Step 8: Retrieving Documents and Analysing Results
"""
logger.info("## Step 8: Retrieving Documents and Analysing Results")

user_query = "How do I increase my productivity for maximum output"
results_df = run_vector_search_operations(user_query, collection)

"""
One key point to note: If you’ve followed this example with a small dataset, you likely won’t observe significant retrieval latency improvements. Quantization methods truly demonstrate their benefits when dealing with large-scale datasets—on the order of one million or more embeddings—where memory savings and speed gains become substantially more noticeable.
"""
logger.info("One key point to note: If you’ve followed this example with a small dataset, you likely won’t observe significant retrieval latency improvements. Quantization methods truly demonstrate their benefits when dealing with large-scale datasets—on the order of one million or more embeddings—where memory savings and speed gains become substantially more noticeable.")

results_df.head()

"""
Quantization is a powerful tool for optimizing vector database performance, especially in applications that handle high-dimensional embeddings like semantic search and recommendation systems. This tutorial demonstrated the implementation of scalar and binary quantization methods using Nomic embeddings with MongoDB as the vector database. 
When leveraged appropriately, effective optimization extends beyond latency improvements. It enables scalability, reduces operational costs, and enhances application user experience. The Benefits of Database Optimization:
Latency Reduction for Improved User Experience: Minimizing delays in data retrieval enhances user satisfaction and engagement.
Efficient Handling of Large-Scale Data: Optimized databases can more effectively manage vast amounts of data, improving performance and scalability.

Cost Reduction and Resource Efficiency: Efficient data storage and retrieval reduce the need for excessive computational resources, leading to cost savings.
By examining the trade-offs between retrieval accuracy and performance across different embedding formats (float32, int8, and binary), we showcased how MongoDB's capabilities, such as vector indexing and automatic quantization, can streamline data storage, retrieval, and analysis. 

From this tutorial, we’ve explored Atlas Vector Search native capabilities for scalar quantization as well as binary quantization with rescoring. Our implementation showed that automatic quantization increases scalability and cost savings by reducing the storage and computational resources for efficient processing of vectors. In most cases, automatic quantization reduces the RAM for mongot by 3.75x for scalar and by 24x for binary; the vector values shrink by 4x and 32x, respectively, but the Hierarchical Navigable Small Worlds graph itself does not shrink.

We recommend automatic quantization if you have a large number of full-fidelity vectors, typically over 10M vectors. After quantization, you index reduced representation vectors without compromising the accuracy of your retrieval.
To further explore quantization techniques and their applications, refer to resources like [Ingesting Quantized Vectors with Cohere](https://www.mongodb.com/developer/products/atlas/ingesting_quantized_vectors_with_cohere/). An [additional notebook](https://github.com/mongodb-developer/GenAI-Showcase/blob/main/notebooks/advanced_techniques/advanced_evaluation_of_quantized_vectors_using_cohere_mongodb_beir.ipynb) for comparing retrieval accuracy between quantized and non-quantized vectors is also available to deepen your understanding of these methods.


"""
logger.info("Quantization is a powerful tool for optimizing vector database performance, especially in applications that handle high-dimensional embeddings like semantic search and recommendation systems. This tutorial demonstrated the implementation of scalar and binary quantization methods using Nomic embeddings with MongoDB as the vector database.")

logger.info("\n\n[DONE]", bright=True)