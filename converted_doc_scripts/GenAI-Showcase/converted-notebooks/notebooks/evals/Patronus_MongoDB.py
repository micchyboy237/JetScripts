from jet.logger import CustomLogger
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import json
import os
import pymongo
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# The Technical Guide on RAG Evaluation with Patronus and MongoDB

## How to Query and Retrieve Results from Atlas Vector Store

To query and retrieve results from MongoDB Atlas vector store, follow these three steps:

### Set Up the Database on Atlas
First, you need to create an account on MongoDB Atlas. This involves signing in to your MongoDB Atlas account, creating a new cluster, and adding a database and collection. You can skip this step if you have already have your collection for vector search.

### Create an Atlas Index
You can create an index either via code or using the Atlas UI. Here’s an example of how to create an index using the Atlas UI:
1. Navigate to your collection.
2. Click on “Atlas search” and then “Create Index”.
3. Define the index fields and type. 

Alternatively, you can create an index programmatically. The following index definition indexes the vector embeddings field (`fieldToIndex`) for performing vector search.

```python

# Connect to your Atlas deployment
uri = "<connectionString>"
client = MongoClient(uri)

# Access your database and collection
database = client["<databaseName>"]
collection = database["<collectionName>"]

# Create your index model, then create the search index
search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
        "numDimensions": <numberofDimensions>,
        "path": "<fieldToIndex>",
        "similarity": "euclidean | cosine | dotProduct"
      },
      {
        "type": "filter",
        "path": "<fieldToIndex>"
      },
      # Add more fields as needed
    ]
  },
  name="<index name>",
  type="vectorSearch",
)

result = collection.create_search_index(model=search_index_model)
logger.debug(result)

# Perform a Vector Query with Code

After setting up your database and creating the necessary indexes, you can perform a vector query. Below is a sample Python code using the PyMongo library:

```python

# connect to your Atlas cluster
client = pymongo.MongoClient("<connection-string>")

# define pipeline
pipeline = [
  {
    '$vectorSearch': {
      'index': "<index-name>", 
      'path':  "<field-to-search>",
      'queryVector': [<array-of-numbers>],
      'numCandidates': <number-of-candidates>,
      'limit': <number-of-results>
    }
  }
]

# run pipeline
result = client["db_name"]["collection_name"].aggregate(pipeline)

# print results
for i in result:
    logger.debug(i)

# How to Choose the Right Model

Patronus provides two versions of the Lynx model:

## Using the Large Model

The Lynx-70B model requires significant computational resources. Ensure you have enough memory and computing power to handle it. You would require an instance with an A100 or H100 GPU. We would use vLLM to run the model on GPU and get an endpoint for our evaluation process.

## Using Smaller Model

If the 70B model is too large for your use case, consider using a smaller variant of the Lynx model (Lynx-8B-Instruct). This can be executed on a local system using Ollama. You can find different sizes on the Hugging Face model hub under the PatronusAI repository.

# How to Download the Model from Hugging Face

Follow these steps to download the Lynx model from Hugging Face. This involves setting up the environment and the Hugging Face CLI, logging into Hugging Face, and then downloading the model.

## Step 1: Install the Hugging Face Hub CLI

First, you need to install the Hugging Face Hub CLI. This lets you interact directly with Hugging Face’s model hub from the command line.
"""
logger.info("# The Technical Guide on RAG Evaluation with Patronus and MongoDB")

pip3 install -U huggingface_hub[cli]

"""
## Step 2: Log In to Hugging Face

Next, log in to your Hugging Face account. If you don’t have an account, you’ll need to create one at Hugging Face.
"""
logger.info("## Step 2: Log In to Hugging Face")

huggingface-cli login

"""
You will be prompted to enter your Hugging Face token. You can find your token in your Hugging Face account settings under “Access Tokens.”

## Step 3: Download the Lynx Model

After logging in, you can download the Lynx model. Here is an example command to download the 70B variant of the Lynx model:
"""
logger.info("## Step 3: Download the Lynx Model")

huggingface-cli download PatronusAI/Patronus-Lynx-8B-Instruct --local-dir Patronus_8B

"""
This command will download the model to a local directory named Patronus_8B.

# How to Deploy Lynx onto the Server Using vLLM

With the vLLM inference server running, you will obtain a URI (for instance, http://localhost:5123/). You can use the Ollama API specification to send requests and evaluate the faithfulness of AI-generated responses. This section covers sending a cURL request to test the server and implementing a structured prompt template for hallucination detection.

## Step 1: Create a New Conda Environment and Install vLLM

Creating a dedicated conda environment helps manage dependencies and avoid conflicts. Here’s how to set up a new environment with Python 3.10:
"""
logger.info("# How to Deploy Lynx onto the Server Using vLLM")

conda create -n myenv python=3.10 -y
conda activate myenv

"""
Install vLLM, a library designed to efficiently serve large language models. If you have CUDA 12.1 installed, you can install vLLM with CUDA support for better performance.
"""
logger.info("Install vLLM, a library designed to efficiently serve large language models. If you have CUDA 12.1 installed, you can install vLLM with CUDA support for better performance.")

pip install vllm

"""
Step 2: Run the Lynx Model on a Server

Once vLLM is installed, you can start the server to host the Lynx model. This involves specifying the port, model, and tokenizer. The following command runs the model on port 5123:
"""
logger.info("Step 2: Run the Lynx Model on a Server")

python -m vllm.entrypoints.ollama.api_server --port 5123 --model PatronusAI/Patronus-Lynx-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B

"""
# How to Catch Hallucinations in Atlas-based RAG System Using Local Lynx API

With the vLLM inference server running on http://localhost:5123/, you can use the Ollama API specification to send requests and evaluate the faithfulness of AI-generated responses. This section covers sending a cURL request to test the server and implementing a structured prompt template for hallucination detection.

## Step 1: Test the Server with a cURL Request

Verify that the server is working by sending a cURL request. This request queries the model to define what a hallucination is:
"""
logger.info("# How to Catch Hallucinations in Atlas-based RAG System Using Local Lynx API")

curl http://localhost:5123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "PatronusAI/Patronus-Lynx-70B-Instruct",
  "messages": [
    {"role": "user", "content": "What is a hallucination?"}
  ]
}'

"""
## Step 2: Define the Prompt Template

Use a structured prompt template to evaluate the faithfulness of AI-generated responses. The template helps ensure that the answer is faithful to the document provided and does not contain hallucinations.
"""
logger.info("## Step 2: Define the Prompt Template")

Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.

The ANSWER must not offer new information beyond the context provided in the DOCUMENT.

The ANSWER also must not contradict information provided in the DOCUMENT.

Output your final score by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.

Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{{ user_input }}

--
DOCUMENT:
{{ provided_context }}

--
ANSWER:
{{ bot_response }}

--

Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE".

Ensure that the JSON is valid and properly formatted.

{"REASONING": ["<your reasoning as bullet points>"], "SCORE": "<final score>"}

"""
## Step 3: Implement the Evaluation Function

Use Python to send a structured request to the local Lynx API, including the question, document, and answer. The following code demonstrates how to format the request and handle the response:
"""
logger.info("## Step 3: Implement the Evaluation Function")


url = "http://localhost:5123/v1/chat/completions"

prompt_template = """
Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT.

The ANSWER must not offer new information beyond the context provided in the DOCUMENT.

The ANSWER also must not contradict information provided in the DOCUMENT.

Output your final score by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT.

Show your reasoning.

--
QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
{user_input}

--
DOCUMENT:
{provided_context}

--
ANSWER

logger.info("\n\n[DONE]", bright=True)