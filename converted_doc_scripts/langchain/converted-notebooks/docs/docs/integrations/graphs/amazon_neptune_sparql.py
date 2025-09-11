from jet.logger import logger
from langchain_aws import ChatBedrockConverse
from langchain_aws.chains import create_neptune_sparql_qa_chain
from langchain_aws.graphs import NeptuneRdfGraph
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import shutil
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Amazon Neptune with SPARQL

>[Amazon Neptune](https://aws.amazon.com/neptune/) is a high-performance graph analytics and serverless database for superior scalability and availability.
>
>This example shows the QA chain that queries [Resource Description Framework (RDF)](https://en.wikipedia.org/wiki/Resource_Description_Framework) data 
in an `Amazon Neptune` graph database using the `SPARQL` query language and returns a human-readable response.
>
>[SPARQL](https://en.wikipedia.org/wiki/SPARQL) is a standard query language for `RDF` graphs.


This example uses a `NeptuneRdfGraph` class that connects with the Neptune database and loads its schema. 
The `create_neptune_sparql_qa_chain` is used to connect the graph and LLM to ask natural language questions.

This notebook demonstrates an example using organizational data.

Requirements for running this notebook:
- Neptune 1.2.x cluster accessible from this notebook
- Kernel with Python 3.9 or higher
- For Bedrock access, ensure IAM role has this policy

```json
{
        "Action": [
            "bedrock:ListFoundationModels",
            "bedrock:InvokeModel"
        ],
        "Resource": "*",
        "Effect": "Allow"
}
```

- S3 bucket for staging sample data. The bucket should be in the same account/region as Neptune.

## Setting up

### Seed the W3C organizational data

Seed the W3C organizational data, W3C org ontology plus some instances. 
 
You will need an S3 bucket in the same region and account as the Neptune cluster. Set `STAGE_BUCKET`as the name of that bucket.
"""
logger.info("# Amazon Neptune with SPARQL")

STAGE_BUCKET = "<bucket-name>"

# %%bash  -s "$STAGE_BUCKET"

rm -rf data
mkdir -p data
cd data
echo getting org ontology and sample org instances
wget http://www.w3.org/ns/org.ttl
wget https://raw.githubusercontent.com/aws-samples/amazon-neptune-ontology-example-blog/main/data/example_org.ttl

echo Copying org ttl to S3
aws s3 cp org.ttl s3://$1/org.ttl
aws s3 cp example_org.ttl s3://$1/example_org.ttl

"""
We will use the `%load` magic command from the `graph-notebook` package to insert the W3C data into the Neptune graph. Before running `%load`, use `%%graph_notebook_config` to set the graph connection parameters.
"""
logger.info("We will use the `%load` magic command from the `graph-notebook` package to insert the W3C data into the Neptune graph. Before running `%load`, use `%%graph_notebook_config` to set the graph connection parameters.")

# !pip install --upgrade --quiet graph-notebook

# %load_ext graph_notebook.magics

# %%graph_notebook_config
{
    "host": "<neptune-endpoint>",
    "neptune_service": "neptune-db",
    "port": 8182,
    "auth_mode": "<[DEFAULT|IAM]>",
    "load_from_s3_arn": "<neptune-cluster-load-role-arn>",
    "ssl": true,
    "aws_region": "<region>"
}

"""
Bulk-load the org ttl - both ontology and instances.
"""
logger.info("Bulk-load the org ttl - both ontology and instances.")

# %load -s s3://{STAGE_BUCKET} -f turtle --store-to loadres --run

# %load_status {loadres['payload']['loadId']} --errors --details

"""
### Setup Chain
"""
logger.info("### Setup Chain")

# !pip install --upgrade --quiet langchain-aws

"""
** Restart kernel **

### Prepare an example
"""
logger.info("### Prepare an example")

EXAMPLES = """

<question>
Find organizations.
</question>

<sparql>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX org: <http://www.w3.org/ns/org#>

select ?org ?orgName where {{
    ?org rdfs:label ?orgName .
}}
</sparql>

<question>
Find sites of an organization
</question>

<sparql>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX org: <http://www.w3.org/ns/org#>

select ?org ?orgName ?siteName where {{
    ?org rdfs:label ?orgName .
    ?org org:hasSite/rdfs:label ?siteName .
}}
</sparql>

<question>
Find suborganizations of an organization
</question>

<sparql>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX org: <http://www.w3.org/ns/org#>

select ?org ?orgName ?subName where {{
    ?org rdfs:label ?orgName .
    ?org org:hasSubOrganization/rdfs:label ?subName  .
}}
</sparql>

<question>
Find organizational units of an organization
</question>

<sparql>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX org: <http://www.w3.org/ns/org#>

select ?org ?orgName ?unitName where {{
    ?org rdfs:label ?orgName .
    ?org org:hasUnit/rdfs:label ?unitName .
}}
</sparql>

<question>
Find members of an organization. Also find their manager, or the member they report to.
</question>

<sparql>
PREFIX org: <http://www.w3.org/ns/org#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

select * where {{
    ?person rdf:type foaf:Person .
    ?person  org:memberOf ?org .
    OPTIONAL {{ ?person foaf:firstName ?firstName . }}
    OPTIONAL {{ ?person foaf:family_name ?lastName . }}
    OPTIONAL {{ ?person  org:reportsTo ??manager }} .
}}
</sparql>


<question>
Find change events, such as mergers and acquisitions, of an organization
</question>

<sparql>
PREFIX org: <http://www.w3.org/ns/org#>

select ?event ?prop ?obj where {{
    ?org rdfs:label ?orgName .
    ?event rdf:type org:ChangeEvent .
    ?event org:originalOrganization ?origOrg .
    ?event org:resultingOrganization ?resultingOrg .
}}
</sparql>

"""

"""
### Create the Neptune Database RDF Graph
"""
logger.info("### Create the Neptune Database RDF Graph")


host = "<your host>"
port = 8182  # change if different
region = "us-east-1"  # change if different
graph = NeptuneRdfGraph(host=host, port=port, use_iam_auth=True, region_name=region)

"""
## Using the Neptune SPARQL QA Chain

This QA chain queries the Neptune graph database using SPARQL and returns a human-readable response.
"""
logger.info("## Using the Neptune SPARQL QA Chain")


MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
llm = ChatBedrockConverse(
    model_id=MODEL_ID,
    temperature=0,
)

chain = create_neptune_sparql_qa_chain(
    llm=llm,
    graph=graph,
    examples=EXAMPLES,
)

result = chain.invoke("How many organizations are in the graph?")
logger.debug(result["result"].content)

"""
Here are a few more prompts to try on the graph data that was ingested.
"""
logger.info("Here are a few more prompts to try on the graph data that was ingested.")

result = chain.invoke("Are there any mergers or acquisitions?")
logger.debug(result["result"].content)

result = chain.invoke("Find organizations.")
logger.debug(result["result"].content)

result = chain.invoke("Find sites of MegaSystems or MegaFinancial.")
logger.debug(result["result"].content)

result = chain.invoke("Find a member who is a manager of one or more members.")
logger.debug(result["result"].content)

result = chain.invoke("Find five members and their managers.")
logger.debug(result["result"].content)

result = chain.invoke(
    "Find org units or suborganizations of The Mega Group. What are the sites of those units?"
)
logger.debug(result["result"].content)

"""
### Adding Message History

The Neptune SPARQL QA chain has the ability to be wrapped by [`RunnableWithMessageHistory`](https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory). This adds message history to the chain, allowing us to create a chatbot that retains conversation state across multiple invocations.

To start, we need a way to store and load the message history. For this purpose, each thread will be created as an instance of [`InMemoryChatMessageHistory`](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.InMemoryChatMessageHistory.html), and stored into a dictionary for repeated access.

(Also see: https://python.langchain.com/docs/versions/migrating_memory/chat_history/#chatmessagehistory)
"""
logger.info("### Adding Message History")


chats_by_session_id = {}


def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history

"""
Now, the QA chain and message history storage can be used to create the new `RunnableWithMessageHistory`. Note that we must set `query` as the input key to match the format expected by the base chain.
"""
logger.info("Now, the QA chain and message history storage can be used to create the new `RunnableWithMessageHistory`. Note that we must set `query` as the input key to match the format expected by the base chain.")


runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="query",
)

"""
Before invoking the chain, a unique `session_id` needs to be generated for the conversation that the new `InMemoryChatMessageHistory` will remember.
"""
logger.info("Before invoking the chain, a unique `session_id` needs to be generated for the conversation that the new `InMemoryChatMessageHistory` will remember.")


session_id = uuid.uuid4()

"""
Finally, invoke the message history enabled chain with the `session_id`.
"""
logger.info("Finally, invoke the message history enabled chain with the `session_id`.")

result = runnable_with_history.invoke(
    {"query": "How many org units or suborganizations does the The Mega Group have?"},
    config={"configurable": {"session_id": session_id}},
)
logger.debug(result["result"].content)

"""
As the chain continues to be invoked with the same `session_id`, responses will be returned in the context of previous queries in the conversation.
"""
logger.info("As the chain continues to be invoked with the same `session_id`, responses will be returned in the context of previous queries in the conversation.")

result = runnable_with_history.invoke(
    {"query": "List the sites for each of the units."},
    config={"configurable": {"session_id": session_id}},
)
logger.debug(result["result"].content)

logger.info("\n\n[DONE]", bright=True)