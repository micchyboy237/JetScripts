from jet.logger import CustomLogger
from llmlingua import PromptCompressor
from pymongo.operations import SearchIndexModel
from tqdm import tqdm
import ast
import ollama
import os
import pandas as pd
import pprint
import pymongo
import shutil
import time
import tqdm


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Enhancing HR Recruitment with MongoDB and Ollama: A GraphRAG Approach
"""
logger.info("# Enhancing HR Recruitment with MongoDB and Ollama: A GraphRAG Approach")

# !pip install --quiet pymongo dataset ollama pandas

# import getpass


def set_env_securely(var_name, prompt):
#     value = getpass.getpass(prompt)
    os.environ[var_name] = value

"""
## Data Loading and Preparation
"""
logger.info("## Data Loading and Preparation")


employee_df = pd.read_csv("employee_dataset_200.csv")


employee_df["skills"] = employee_df["skills"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
employee_df["certifications"] = employee_df["certifications"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

employee_df.head()

# set_env_securely("OPENAI_API_KEY", "Enter your OPENAI API KEY: ")



def summarize_datapoint(data_point):
    """
    Summarize the given data point using Ollama's API.

    Args:
        data_point (str): The text to summarize.

    Returns:
        str: A concise summary of the input data.
    """
    if not data_point or not isinstance(data_point, str):
        raise ValueError("Invalid data point. Please provide a non-empty string.")

    try:
        response = ollama.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert summarizer. Focus on the key points, removing unnecessary details. Write in a concise and clear manner.",
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following data: {data_point}",
                },
            ],
        )

        summary = response.choices[0].message.content

        return summary
    except Exception as e:
        return f"Error summarizing data: {e!s}"

def create_datapoint_summary(row):
    """Concatenates all attributes of a row and generates a summary."""
    attributes = [
        str(value) for key, value in row.items() if key != "datapoint_summary"
    ]
    data_point = " ".join(attributes)

    summary = summarize_datapoint(data_point)  # Call the summarization function
    return summary


employee_df["datapoint_summary"] = employee_df.apply(create_datapoint_summary, axis=1)

employee_df.head()

"""
## Embedding Generation
"""
logger.info("## Embedding Generation")

OPENAI_EMBEDDING_MODEL = "mxbai-embed-large"
OPENAI_EMBEDDING_MODEL_DIMENSION = 1536



def get_embedding(text):
    """Generate an embedding for the given text using Ollama's API."""

    if not text or not isinstance(text, str):
        return None

    try:
        embedding = (
            ollama.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL,
                dimensions=OPENAI_EMBEDDING_MODEL_DIMENSION,
            )
            .data[0]
            .embedding
        )
        return embedding
    except Exception as e:
        logger.debug(f"Error in get_embedding: {e}")
        return None

try:
    employee_df["embedding"] = [
        x
        for x in tqdm(
            employee_df["datapoint_summary"].apply(get_embedding),
            total=len(employee_df),
        )
    ]
    logger.debug("Embeddings generated for employees")
except Exception as e:
    logger.debug(f"Error applying embedding function to DataFrame: {e}")

employee_df.head()

"""
## Connecting to MongoDB
"""
logger.info("## Connecting to MongoDB")

set_env_securely("MONGO_URI", "Enter your MONGO URI: ")



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.rag.graphrag.employees.python"
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

DB_NAME = "acme_corpration"
COLLECTION_NAME = "employees"

db = mongo_client[DB_NAME]

collection = db[COLLECTION_NAME]

collection.delete_many({})

"""
## Data Ingestion
"""
logger.info("## Data Ingestion")

documents = employee_df.to_dict("records")
collection.insert_many(documents)

logger.debug("Data ingestion into MongoDB completed")

"""
## MongoDB Graph Lookup
"""
logger.info("## MongoDB Graph Lookup")

graph_lookup_query = [
    {
        "$match": {"employee_id": 1}  # Start with Employee 1
    },
    {
        "$graphLookup": {
            "from": "employees",  # Collection name
            "startWith": "$team",  # Starting with the employee's skills array
            "connectFromField": "team",  # Match on array elements in the starting employee
            "connectToField": "team",  # Match on array elements in other employees
            "as": "related_employees",  # Output field for related employees
            "maxDepth": 1,  # Limit the depth of recursion
            "depthField": "level",  # Optional: Include the depth level in results
        }
    },
]

project_stage = {"$project": {"embedding": 0, "related_employees.embedding": 0}}

graph_lookup_query.append(project_stage)


result = list(collection.aggregate(graph_lookup_query))


pprint.plogger.debug(result)

"""
## Naive/Baseline RAG
"""
logger.info("## Naive/Baseline RAG")




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

        logger.debug(f"Waiting for 30 seconds to allow index '{index_name}' to be created...")
        time.sleep(30)

        logger.debug(f"30-second wait completed for index '{index_name}'.")
        return result

    except Exception as e:
        logger.debug(f"Error creating new vector search index '{index_name}': {e!s}")
        return None

vector_search_index_definition = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": OPENAI_EMBEDDING_MODEL_DIMENSION,
            "similarity": "cosine",
        }
    ]
}

setup_vector_search_index(
    collection, vector_search_index_definition, index_name="vector_index"
)

def vector_search(user_query, collection, vector_search_index_name="vector_index"):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.
    additional_stages (list): Additional aggregation stages to include in the pipeline.
    vector_search_index_name (str): The name of the vector search index.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 5,  # Return top 5 matches
        }
    }

    project_stage = {
        "$project": {
            "embedding": 0,  # Remove embedding from top-level documents
            "skills": 0,  # Remove skills from results
            "certifications": 0,  # Remove certifications from results
            "Summary": 0,  # Remove summary from results
        }
    }

    pipeline = [vector_search_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

def handle_user_query_naive_rag(query: str):
    results = vector_search(query, collection)

    if results:
        context = results

        response = ollama.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in recommending teams based on employee data. Consider the provided employee data to answer the user's query. If the data is not sufficient to answer the query, simply state that you need more information.",
                },
                {
                    "role": "user",
                    "content": f"Here's the user's query: {query}\n\nHere's some potentially relevant employee data: {context}",
                },
            ],
        )

        return response.choices[0].message.content
    return "No relevant employees found for this query."

query = (
    "Get me employees that can form a team to build a website for HR recruitement firm"
)

naive_rag_results = handle_user_query_naive_rag(query)

logger.debug(naive_rag_results)

"""
## GraphRAG
"""
logger.info("## GraphRAG")

def customGraphRAG(text: str):
    """
    Performs a custom GraphRAG operation by conducting a vector search
    followed by graph traversal using graphLookup.

    Args:
        text (str): The query text.

    Returns:
        list: A list of documents containing the relevant results.
    """

    query_embedding = get_embedding(text)

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 1,  # Return top 5 matches
        }
    }

    graph_lookup_stage = {
        "$graphLookup": {
            "from": "employees",  # Collection to perform graph traversal on
            "startWith": "$skills",  # Start the traversal using skills field
            "connectFromField": "skills",  # Field in the current document to match
            "connectToField": "skills",  # Field in the other documents to match
            "as": "related_employees",  # Output field for connected documents
            "maxDepth": 2,  # Depth of graph traversal
            "depthField": "level",  # Include recursion level in results
        }
    }

    project_stage = {
        "$project": {
            "_id": 1,
            "embedding": 0,  # Remove embedding from top-level documents
            "related_employees.embedding": 0,  # Remove embedding from nested results
            "related_employees.skills": 0,  # Remove skills from nested results
            "related_employees.certifications": 0,  # Remove certifications from nested results
            "related_employees.Summary": 0,  # Remove summary from nested results
        }
    }

    pipeline = [
        vector_search_stage,  # Perform vector search
        graph_lookup_stage,  # Conduct graph traversal
        project_stage,  # Clean up unnecessary fields
    ]

    try:
        result = list(collection.aggregate(pipeline))
        return result
    except Exception as e:
        logger.debug(f"An error occurred: {e!s}")
        return []

results = customGraphRAG(query)

pprint.plogger.debug(results)

"""
### Prompt Compression Technique
Because GraphRAG can be expensive, token wise
"""
logger.info("### Prompt Compression Technique")

# !pip install --quiet -U llmlingua


llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",  # Smaller Model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
    use_llmlingua2=True,
    device_map="cpu",
)

def handle_user_query(query: str, compress_prompt=False):
    results = customGraphRAG(query)

    if results and results[0].get("related_employees"):
        context = results[0]["related_employees"]

        prompt = f"Here's the user's query: {query}\n\nHere's some potentially relevant employee data: {context}"

        if compress_prompt:
            compression_result = llm_lingua.compress_prompt(
                prompt, rate=0.20, force_tokens=["\n", "?"]
            )
            final_prompt = compression_result["compressed_prompt"]
            logger.debug(f"Compressed prompt: {final_prompt}")
            logger.debug()
        else:
            final_prompt = prompt

        response = ollama.chat.completions.create(
            model="llama3.2", log_dir=f"{LOG_DIR}/chats",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in recommending teams based on employee data. Consider the provided employee data to answer the user's query.",
                },
                {"role": "user", "content": final_prompt},
            ],
        )

        return response.choices[0].message.content
    return "No relevant employees found for this query."

results = handle_user_query(query)

logger.debug(results)

results = handle_user_query(query, compress_prompt=True)

logger.debug(results)

logger.info("\n\n[DONE]", bright=True)