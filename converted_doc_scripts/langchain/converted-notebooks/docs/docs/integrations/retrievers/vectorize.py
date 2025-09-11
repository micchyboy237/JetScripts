from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_vectorize.retrievers import VectorizeRetriever
import ChatModelTabs from "@theme/ChatModelTabs";
import json
import os
import shutil
import urllib3
import vectorize_client as v


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
---
sidebar_label: Vectorize
---

# VectorizeRetriever

This notebook shows how to use the LangChain Vectorize retriever.

> [Vectorize](https://vectorize.io/) helps you build AI apps faster and with less hassle.
> It automates data extraction, finds the best vectorization strategy using RAG evaluation,
> and lets you quickly deploy real-time RAG pipelines for your unstructured data.
> Your vector search indexes stay up-to-date, and it integrates with your existing vector database,
> so you maintain full control of your data.
> Vectorize handles the heavy lifting, freeing you to focus on building robust AI solutions without getting bogged down by data management.

## Setup

In the following steps, we'll setup the Vectorize environment and create a RAG pipeline.

### Create a Vectorize Account & Get Your Access Token

Sign up for a free Vectorize account [here](https://platform.vectorize.io/)
Generate an access token in the [Access Token](https://docs.vectorize.io/rag-pipelines/retrieval-endpoint#access-tokens) section
Gather your organization ID. From the browser url, extract the UUID from the URL after /organization/

### Configure token and organization ID
"""
logger.info("# VectorizeRetriever")

# import getpass

# VECTORIZE_ORG_ID = getpass.getpass("Enter Vectorize organization ID: ")
# VECTORIZE_API_TOKEN = getpass.getpass("Enter Vectorize API Token: ")

"""
### Installation

This retriever lives in the `langchain-vectorize` package:
"""
logger.info("### Installation")

# !
p
i
p

i
n
s
t
a
l
l

-
q
U

l
a
n
g
c
h
a
i
n
-
v
e
c
t
o
r
i
z
e

"""
#
#
#
 
D
o
w
n
l
o
a
d
 
a
 
P
D
F
 
f
i
l
e
"""
logger.info("#")

# !
w
g
e
t

"
h
t
t
p
s
:
/
/
r
a
w
.
g
i
t
h
u
b
u
s
e
r
c
o
n
t
e
n
t
.
c
o
m
/
v
e
c
t
o
r
i
z
e
-
i
o
/
v
e
c
t
o
r
i
z
e
-
c
l
i
e
n
t
s
/
r
e
f
s
/
t
a
g
s
/
p
y
t
h
o
n
-
0
.
1
.
3
/
t
e
s
t
s
/
p
y
t
h
o
n
/
t
e
s
t
s
/
r
e
s
e
a
r
c
h
.
p
d
f
"

"""
#
#
#
 
I
n
i
t
i
a
l
i
z
e
 
t
h
e
 
v
e
c
t
o
r
i
z
e
 
c
l
i
e
n
t
"""
logger.info("#")


api = v.ApiClient(v.Configuration(access_token=VECTORIZE_API_TOKEN))

"""
#
#
#
 
C
r
e
a
t
e
 
a
 
F
i
l
e
 
U
p
l
o
a
d
 
S
o
u
r
c
e
 
C
o
n
n
e
c
t
o
r
"""
logger.info("#")



connectors_api = v.ConnectorsApi(api)
response = connectors_api.create_source_connector(
    VECTORIZE_ORG_ID, [{"type": "FILE_UPLOAD", "name": "From API"}]
)
source_connector_id = response.connectors[0].id

"""
#
#
#
 
U
p
l
o
a
d
 
t
h
e
 
P
D
F
 
f
i
l
e
"""
logger.info("#")

file_path = "research.pdf"

http = urllib3.PoolManager()
uploads_api = v.UploadsApi(api)
metadata = {"created-from-api": True}

upload_response = uploads_api.start_file_upload_to_connector(
    VECTORIZE_ORG_ID,
    source_connector_id,
    v.StartFileUploadToConnectorRequest(
        name=file_path.split("/")[-1],
        content_type="application/pdf",
        metadata=json.dumps(metadata),
    ),
)

with open(file_path, "rb") as f:
    response = http.request(
        "PUT",
        upload_response.upload_url,
        body=f,
        headers={
            "Content-Type": "application/pdf",
            "Content-Length": str(os.path.getsize(file_path)),
        },
    )

if response.status != 200:
    logger.debug("Upload failed: ", response.data)
else:
    logger.debug("Upload successful")

"""
#
#
#
 
C
o
n
n
e
c
t
 
t
o
 
t
h
e
 
A
I
 
P
l
a
t
f
o
r
m
 
a
n
d
 
V
e
c
t
o
r
 
D
a
t
a
b
a
s
e
"""
logger.info("#")

ai_platforms = connectors_api.get_ai_platform_connectors(VECTORIZE_ORG_ID)
builtin_ai_platform = [
    c.id for c in ai_platforms.ai_platform_connectors if c.type == "VECTORIZE"
][0]

vector_databases = connectors_api.get_destination_connectors(VECTORIZE_ORG_ID)
builtin_vector_db = [
    c.id for c in vector_databases.destination_connectors if c.type == "VECTORIZE"
][0]

"""
#
#
#
 
C
o
n
f
i
g
u
r
e
 
a
n
d
 
D
e
p
l
o
y
 
t
h
e
 
P
i
p
e
l
i
n
e
"""
logger.info("#")

pipelines = v.PipelinesApi(api)
response = pipelines.create_pipeline(
    VECTORIZE_ORG_ID,
    v.PipelineConfigurationSchema(
        source_connectors=[
            v.SourceConnectorSchema(
                id=source_connector_id, type="FILE_UPLOAD", config={}
            )
        ],
        destination_connector=v.DestinationConnectorSchema(
            id=builtin_vector_db, type="VECTORIZE", config={}
        ),
        ai_platform=v.AIPlatformSchema(
            id=builtin_ai_platform, type="VECTORIZE", config={}
        ),
        pipeline_name="My Pipeline From API",
        schedule=v.ScheduleSchema(type="manual"),
    ),
)
pipeline_id = response.data.id

"""
### Configure tracing (optional)

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("### Configure tracing (optional)")



"""
#
#
 
I
n
s
t
a
n
t
i
a
t
i
o
n
"""
logger.info("#")


retriever = VectorizeRetriever(
    api_token=VECTORIZE_API_TOKEN,
    organization=VECTORIZE_ORG_ID,
    pipeline_id=pipeline_id,
)

"""
## Usage
"""
logger.info("## Usage")

query = "Apple Shareholders equity"
retriever.invoke(query, num_results=2)

"""
## Use within a chain

Like other retrievers, VectorizeRetriever can be incorporated into LLM applications via [chains](/docs/how_to/sequence/).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within a chain")


llm = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

c
h
a
i
n
.
i
n
v
o
k
e
(
"
.
.
.
"
)

"""
## API reference

For detailed documentation of all VectorizeRetriever features and configurations head to the [API reference](https://python.langchain.com/api_reference/vectorize/langchain_vectorize.retrievers.VectorizeRetriever.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)