from jet.logger import logger
from langchain_core.documents import Document
from langchain_lindorm_integration.embeddings import LindormAIEmbeddings
from langchain_lindorm_integration.vectorstores import LindormVectorStore
import os
import shutil


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
sidebar_label: Lindorm
---

# LindormVectorStore

This notebook covers how to get started with the Lindorm vector store.

## Setup

To access Lindorm vector stores you'll need to create a Lindorm account, get the ak/sk, and install the `langchain-lindorm-integration` integration package.
"""
logger.info("# LindormVectorStore")

# %
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

"
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
l
i
n
d
o
r
m
-
i
n
t
e
g
r
a
t
i
o
n
"

"""
### Credentials

Head to [here](https://help.aliyun.com/document_detail/2773369.html?spm=a2c4g.11186623.help-menu-172543.d_2_5_0.2a383f96gr5N3M&scm=20140722.H_2773369._.OR_help-T_cn~zh-V_1) to sign up to Lindorm and generate the ak/sk.
"""
logger.info("### Credentials")



class Config:
    SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT", "SEARCH_ENDPOINT")
    SEARCH_USERNAME = os.environ.get("SEARCH_USERNAME", "root")
    SEARCH_PWD = os.environ.get("SEARCH_PASSWORD", "<PASSWORD>")
    AI_LLM_ENDPOINT = os.environ.get("AI_ENDPOINT", "<AI_ENDPOINT>")
    AI_USERNAME = os.environ.get("AI_USERNAME", "root")
    AI_PWD = os.environ.get("AI_PASSWORD", "<PASSWORD>")
    AI_DEFAULT_EMBEDDING_MODEL = "bge_m3_model"  # set to your model

"""
## Initialization

here we use the embedding model deployed on Lindorm AI Service.
"""
logger.info("## Initialization")


embeddings = LindormAIEmbeddings(
    endpoint=Config.AI_LLM_ENDPOINT,
    username=Config.AI_USERNAME,
    password=Config.AI_PWD,
    model_name=Config.AI_DEFAULT_EMBEDDING_MODEL,
)

index = "test_index"
vector = embeddings.embed_query("hello word")
dimension = len(vector)
vector_store = LindormVectorStore(
    lindorm_search_url=Config.SEARCH_ENDPOINT,
    embedding=embeddings,
    http_auth=(Config.SEARCH_USERNAME, Config.SEARCH_PWD),
    dimension=dimension,
    embeddings=embeddings,
    index_name=index,
)

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


document_1 = Document(page_content="foo", metadata={"source": "https://example.com"})

document_2 = Document(page_content="bar", metadata={"source": "https://example.com"})

document_3 = Document(page_content="baz", metadata={"source": "https://example.com"})

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents, ids=["1", "2", "3"])

"""
#
#
#
 
D
e
l
e
t
e
 
i
t
e
m
s
 
f
r
o
m
 
v
e
c
t
o
r
 
s
t
o
r
e
"""
logger.info("#")

vector_store.delete(ids=["3"])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(query="thud", k=1)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
I
f
 
y
o
u
 
w
a
n
t
 
t
o
 
e
x
e
c
u
t
e
 
a
 
s
i
m
i
l
a
r
i
t
y
 
s
e
a
r
c
h
 
a
n
d
 
r
e
c
e
i
v
e
 
t
h
e
 
c
o
r
r
e
s
p
o
n
d
i
n
g
 
s
c
o
r
e
s
 
y
o
u
 
c
a
n
 
r
u
n
:
"""
logger.info("I")

results = vector_store.similarity_search_with_score(query="thud", k=1)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/#retrieval)

## API reference

For detailed documentation of all LindormVectorStore features and configurations head to [the API reference](https://pypi.org/project/langchain-lindorm-integration/).
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)