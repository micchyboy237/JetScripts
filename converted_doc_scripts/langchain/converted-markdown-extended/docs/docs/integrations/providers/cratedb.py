from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.globals import set_llm_cache
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.utilities import SQLDatabase
from langchain_cratedb import CrateDBCache
from langchain_cratedb import CrateDBChatMessageHistory
from langchain_cratedb import CrateDBLoader
from langchain_cratedb import CrateDBSemanticCache
from langchain_cratedb import CrateDBVectorStore
import os
import shutil
import sqlalchemy as sa


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
# CrateDB

> [CrateDB] is a distributed and scalable SQL database for storing and
> analyzing massive amounts of data in near real-time, even with complex
> queries. It is PostgreSQL-compatible, based on Lucene, and inheriting
> from Elasticsearch.


## Installation and Setup

### Setup CrateDB
There are two ways to get started with CrateDB quickly. Alternatively,
choose other [CrateDB installation options].

#### Start CrateDB on your local machine
Example: Run a single-node CrateDB instance with security disabled,
using Docker or Podman. This is not recommended for production use.
"""
logger.info("# CrateDB")

docker run --name=cratedb --rm \
  --publish=4200:4200 --publish=5432:5432 --env=CRATE_HEAP_SIZE=2g \
  crate:latest -Cdiscovery.type=single-node

"""
#### Deploy cluster on CrateDB Cloud
[CrateDB Cloud] is a managed CrateDB service. Sign up for a
[free trial][CrateDB Cloud Console].

### Install Client
Install the most recent version of the [langchain-cratedb] package
and a few others that are needed for this tutorial.
"""
logger.info("#### Deploy cluster on CrateDB Cloud")

pip install --upgrade langchain-cratedb langchain-ollama unstructured

"""
## Documentation
For a more detailed walkthrough of the CrateDB wrapper, see
[using LangChain with CrateDB]. See also [all features of CrateDB]
to learn about other functionality provided by CrateDB.


## Features
The CrateDB adapter for LangChain provides APIs to use CrateDB as vector store,
document loader, and storage for chat messages.

### Vector Store
Use the CrateDB vector store functionality around `FLOAT_VECTOR` and `KNN_MATCH`
for similarity search and other purposes. See also [CrateDBVectorStore Tutorial].

Make sure you've configured a valid Ollama API key.
"""
logger.info("## Documentation")

# export OPENAI_API_KEY=sk-XJZ...


loader = UnstructuredURLLoader(urls=["https://github.com/langchain-ai/langchain/raw/refs/tags/langchain-core==0.3.28/docs/docs/how_to/state_of_the_union.txt"])
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

CONNECTION_STRING = "crate://?schema=testdrive"

store = CrateDBVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="state_of_the_union",
    connection=CONNECTION_STRING,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = store.similarity_search_with_score(query)

"""
### Document Loader
Load load documents from a CrateDB database table, using the document loader
`CrateDBLoader`, which is based on SQLAlchemy. See also [CrateDBLoader Tutorial].

To use the document loader in your applications:
"""
logger.info("### Document Loader")


CONNECTION_STRING = "crate://?schema=testdrive"

db = SQLDatabase(engine=sa.create_engine(CONNECTION_STRING))

loader = CrateDBLoader(
    'SELECT * FROM sys.summits LIMIT 42',
    db=db,
)
documents = loader.load()

"""
### Chat Message History
Use CrateDB as the storage for your chat messages.
See also [CrateDBChatMessageHistory Tutorial].

To use the chat message history in your applications:
"""
logger.info("### Chat Message History")


CONNECTION_STRING = "crate://?schema=testdrive"

message_history = CrateDBChatMessageHistory(
    session_id="test-session",
    connection=CONNECTION_STRING,
)

message_history.add_user_message("hi!")

"""
### Full Cache
The standard / full cache avoids invoking the LLM when the supplied
prompt is exactly the same as one encountered already.
See also [CrateDBCache Example].

To use the full cache in your applications:
"""
logger.info("### Full Cache")


engine = sa.create_engine("crate://crate@localhost:4200/?schema=testdrive")
set_llm_cache(CrateDBCache(engine))

llm = ChatOllama(
    model_name="chatgpt-4o-latest",
    temperature=0.7,
)
logger.debug()
logger.debug("Asking with full cache:")
answer = llm.invoke("What is the answer to everything?")
logger.debug(answer.content)

"""
### Semantic Cache

The semantic cache allows users to retrieve cached prompts based on semantic
similarity between the user input and previously cached inputs. It also avoids
invoking the LLM when not needed.
See also [CrateDBSemanticCache Example].

To use the semantic cache in your applications:
"""
logger.info("### Semantic Cache")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

engine = sa.create_engine("crate://crate@localhost:4200/?schema=testdrive")
set_llm_cache(
    CrateDBSemanticCache(
        embedding=embeddings,
        connection=engine,
        search_threshold=1.0,
    )
)

llm = ChatOllama(model="llama3.2")
logger.debug()
logger.debug("Asking with semantic cache:")
answer = llm.invoke("What is the answer to everything?")
logger.debug(answer.content)

"""
[all features of CrateDB]: https://cratedb.com/docs/guide/feature/
[CrateDB]: https://cratedb.com/database
[CrateDB Cloud]: https://cratedb.com/database/cloud
[CrateDB Cloud Console]: https://console.cratedb.cloud/?utm_source=langchain&utm_content=documentation
[CrateDB installation options]: https://cratedb.com/docs/guide/install/
[CrateDBCache Example]: https://github.com/crate/langchain-cratedb/blob/main/examples/basic/cache.py
[CrateDBSemanticCache Example]: https://github.com/crate/langchain-cratedb/blob/main/examples/basic/cache.py
[CrateDBChatMessageHistory Tutorial]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/conversational_memory.ipynb
[CrateDBLoader Tutorial]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/document_loader.ipynb
[CrateDBVectorStore Tutorial]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/vector_search.ipynb
[langchain-cratedb]: https://pypi.org/project/langchain-cratedb/
[using LangChain with CrateDB]: https://cratedb.com/docs/guide/integrate/langchain/
"""

logger.info("\n\n[DONE]", bright=True)