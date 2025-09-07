from jet.logger import CustomLogger
import boto3
import os
import random
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/mongodb_with_aws_bedrock_agent.ipynb)

# MongoDB with Bedrock agent quick tutorial
MongoDB Atlas and Amazon Bedrock have joined forces to streamline the development of generative AI applications through their seamless integration. MongoDB Atlas, a robust cloud-based database service, now offers native support for Amazon Bedrock, AWS's managed service for generative AI. This integration leverages Atlas's vector search capabilities, enabling the effective utilization of enterprise data to augment the foundational models provided by Bedrock, such as Ollama's Claude and Amazon's Titan. The combination ensures that the generative AI models have access to the most relevant and up-to-date data, significantly improving the accuracy and reliability of AI-driven applications​ with [MongoDB](https://www.mongodb.com/developer/products/atlas/rag-workflow-with-atlas-amazon-bedrock/)​.

This integration simplifies the workflow for developers aiming to implement retrieval-augmented generation (RAG). RAG helps mitigate the issue of hallucinations in AI models by allowing them to fetch and utilize specific data from a predefined knowledge base, in this case, MongoDB Atlas Developers can easily set up this workflow by creating a vector search index in Atlas, which stores the vector embeddings and metadata of the text data. This setup not only enhances the performance and reliability of AI applications but also ensures data privacy and security through features like AWS PrivateLink​​.

This notebook demonstrates how to interact with a predefined agent using [AWS Bedrock](https://aws.amazon.com/bedrock/) in a Google Colab environment. It utilizes the `boto3` library to communicate with the AWS Bedrock service and allows you to input prompts and receive responses directly within the notebook.



## Key Features:
# 1. **Secure Handling of AWS Credentials**: The `getpass` module is used to securely enter your AWS Access Key and Secret Key.
2. **Session Management**: Each session is assigned a random session ID to maintain continuity in conversations.
3. **Agent Invocation**: The notebook sends user prompts to a predefined agent and streams the responses back to the user.

### Requirements:
- AWS Access Key and Secret Key with appropriate permissions.
- Boto3 and Requests libraries for interacting with AWS services and fetching data from URLs.


## Setting up MongoDB Atlas

1. Follow the [getting started with Atlas](https://www.mongodb.com/docs/atlas/getting-started/) guide and setup your cluster with `0.0.0.0/0` allowed connection for this notebook.
2. Predefined an Atlas Vector Index on database `bedrock` collection `agenda`, this collection will host the data for the AWS summit agenda and will serve as a context store for the agent:
**Index name**: `vector_index`
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type" : "filter",
      "path" : "metadata"
    },
    {
      "type" : "filter",
      "path" : "text"
    },
  ]
}
```


## Setup AWS Bedrock

**We will use US-EAST-1 AWS region for this notebook**

Follow our official tutorial to enable a bedrock knowledge base against the created database and collection in MongoDB Atlas. This [guide](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/amazon-bedrock/) highlight a detailed step of action to build the knowledge base and agent.

For this notebook, we will perform the following tasks according to the guide:

1. Go to the bedrock console and enable
- Amazon Titan Text Embedding model (`amazon.titan-embed-text-v2:0`)
- Claude 3 Sonnet Model (The LLM(

2. Upload the following source data about the AWS summit agenda to your S3 bucket:
- https://s3.amazonaws.com/bedrocklogs.pavel/ocr_db.aws_events.json
- https://s3.amazonaws.com/bedrocklogs.pavel/ocr_db.aws_sessions.json

This will be our source data listing the events happening in the summit.

3. Go to Secrets Manager on the AWS console and create credentials to our atlas cluster via "Other type of secret":
- key : username , value : `<ATLAS_USERNAME>`
- key : password , value : `<ATLAS_PASSWORD>`

4. Follow the setup of the knowledge base wizard to connect Bedrock models with Atlas :
- Click "Create Knowledge Base" and input:

|input|value|
|---|---|
|Name| `<NAME>` |
|Chose| Create and use a new service role|
|Data source name| `<NAME>`|
|S3 URI| Browse for the S3 bucket hosting the 2 uploaded source files|
|Embedding Model| Titan Text Embeddings v2|


- let's choose MongoDB Atlas in the "Vector Database" choose the "Choose a vector store you have created" section:

|input|value|
|---|---|
|Select your vector store| **MongoDB Atlas** |
|Hostname| Your atlas srv hostname `eg. cluster0.abcd.mongodb.net`|
|Database name| `bedrock`|
|Collection name| `agenda`|
|Credentials secret ARN| Copy the created credentials from the "Secrets manager"|
|Vector search index name|`vector_index`|
|Vector embedding field path| `embedding`|
|Text field path| `text`|
|Metadata field path| `metadata` |
5. Click Next, review the details and "Create Knowledge Base".

6. Once the knowledge base is marked with "Status : Ready", go to `Data source` section, choose the one datasource we have and click the "Sync" button on its right upper corner. This operation should load the data to Atlas if everything was setup correctly.

## Setting up an agenda agent

We can now set up our agent, who will work with a set of instructions and our knowledge base.

1. Go to the "Agents" tab in the bedrock UI.
2. Click "Create Agent" and give it a meaningful name (e.g. agenda_assistant)
3. Input the following data in the agent builder:

|input|value|
|---|---|
|Agent Name| agenda_assistant |
|Agent resource role| Create and use a new service role |
|Select model| Ollama - Claude 3 Sonnet |
|Instructions for the Agent| **You are a friendly AI chatbot that helps users find and build agenda Items for AWS Summit Tel Aviv.  elaborate as much as possible on the response.** |
|Agent Name| agenda_assistant |
|Knowledge bases| **Choose your Knowledge Base** |
|Aliases| Create a new Alias|

And now, we have a functioning agent that can be tested via the console.
Let's move to the notebook.

**Take note of the Agent ID and create an Agent Alias ID for the notebook**

## Interacting with the agent

To interact with the agent, we need to install the AWS python SDK:
"""
logger.info("# MongoDB with Bedrock agent quick tutorial")

# !pip install boto3

"""
Let's place the credentials for our AWS account.
"""
logger.info("Let's place the credentials for our AWS account.")

# import getpass


# aws_access_key = getpass.getpass("Enter your AWS Access Key: ")
# aws_secret_key = getpass.getpass("Enter your AWS Secret Key: ")

"""
Now, we need to initialise the boto3 client and get the agent ID and alias ID input.
"""
logger.info("Now, we need to initialise the boto3 client and get the agent ID and alias ID input.")

bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name="us-east-1",
)

# agent_id = getpass.getpass("Enter your agent ID")
# agent_alias_id = getpass.getpass("Enter your agent Alias ID")

"""
Let's build the helper function to interact with the agent.
"""
logger.info("Let's build the helper function to interact with the agent.")

def randomise_session_id():
    """
    Generate a random session ID.

    Returns:
        str: A random session ID.
    """
    return str(random.randint(1000, 9999))


def data_stream_generator(response):
    """
    Generator to yield data chunks from the response.

    Args:
        response (dict): The response dictionary.

    Yields:
        str: The next chunk of data.
    """
    for event in response["completion"]:
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            yield chunk["bytes"].decode()


def invoke_agent(bedrock_agent_runtime, agent_id, agent_alias_id, session_id, prompt):
    """
    Sends a prompt for the agent to process and respond to, streaming the response data.

    Args:
        bedrock_agent_runtime (boto3 client): The runtime client to invoke the agent.
        agent_id (str): The unique identifier of the agent to use.
        agent_alias_id (str): The alias of the agent to use.
        session_id (str): The unique identifier of the session. Use the same value across requests to continue the same conversation.
        prompt (str): The prompt that you want the agent to complete.

    Returns:
        str: The response from the agent.
    """
    try:
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
        )

        ret_response = "".join(data_stream_generator(response))

        return ret_response

    except Exception as e:
        return f"Error invoking agent: {e}"

"""
We can now interact with the agent using the application code.
"""
logger.info("We can now interact with the agent using the application code.")

session_id = randomise_session_id()

while True:
    prompt = input("Enter your prompt (or type 'exit' to quit): ")

    if prompt.lower() == "exit":
        break

    response = invoke_agent(
        bedrock_agent_runtime, agent_id, agent_alias_id, session_id, prompt
    )

    logger.debug("Agent Response:")
    logger.debug(response)

"""
Here you go! You have a powerful bedrock agent with MongoDB Atlas.

Conclusions
The integration of MongoDB Atlas with Amazon Bedrock represents a significant advancement in the development and deployment of generative AI applications. By leveraging Atlas's vector search capabilities and the powerful foundational models available through Bedrock, developers can create applications that are both highly accurate and deeply informed by enterprise data. This seamless integration facilitates the retrieval-augmented generation (RAG) workflow, enabling AI models to access and utilize the most relevant data, thereby reducing the likelihood of hallucinations and improving overall performance.

The benefits of this integration extend beyond just technical enhancements. It also simplifies the generative AI stack, allowing companies to rapidly deploy scalable AI solutions with enhanced privacy and security features, such as those provided by AWS PrivateLink. This makes it an ideal solution for enterprises with stringent data security requirements. Overall, the combination of MongoDB Atlas and Amazon Bedrock provides a robust, efficient, and secure platform for building next-generation AI applications​ .
"""
logger.info("Here you go! You have a powerful bedrock agent with MongoDB Atlas.")

logger.info("\n\n[DONE]", bright=True)