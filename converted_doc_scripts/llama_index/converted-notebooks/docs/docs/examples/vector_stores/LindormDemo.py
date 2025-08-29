from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import (
MetadataFilters,
MetadataFilter,
FilterOperator,
FilterCondition,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.lindorm import (
LindormVectorStore,
LindormVectorClient,
)
import needed package dependencies:
import os
import regex as re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/LindormDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Lindorm

>[Lindorm](https://www.alibabacloud.com/help/en/lindorm) is a cloud native multi-model database service. It allows you to store data of all sizes. Lindorm supports low-cost storage and processing of large amounts of data and the pay-as-you-go billing method. It is compatible with the open standards of multiple open source software, such as Apache HBase, Apache Cassandra, Apache Phoenix, OpenTSDB, Apache Solr, and SQL.


To run this notebook you need a Lindorm instance running in the cloud. You can get one following [this link](https://alibabacloud.com/help/en/lindorm/latest/create-an-instance).

After creating the instance, you can get your instance [information](https://www.alibabacloud.com/help/en/lindorm/latest/view-endpoints) and run [curl commands](https://www.alibabacloud.com/help/en/lindorm/latest/connect-and-use-the-search-engine-with-the-curl-command) to connect to and use LindormSearch

## Setup

If you're opening this Notebook on colab, you will probably need to ensure you have `llama-index` installed:
"""
logger.info("# Lindorm")

# !pip install llama-index

# !pip install opensearch-py

# %pip install llama-index-vector-stores-lindorm

# %pip install llama-index-embeddings-dashscope
# %pip install llama-index-llms-dashscope

"""
"""
logger.info("import needed package dependencies:")


"""
Config dashscope embedding and llm model, your can also use default openai or other model to test
"""
logger.info("Config dashscope embedding and llm model, your can also use default openai or other model to test")


Settings.embed_model = DashScopeEmbedding()


dashscope_llm = DashScope(model_name=DashScopeGenerationModels.QWEN_MAX)

"""
## Download example data:
"""
logger.info("## Download example data:")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data:
"""
logger.info("## Load Data:")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")
logger.debug(
    "First document, text"
    f" ({len(documents[0].text)} characters):\n{'='*20}\n{documents[0].text[:360]} ..."
)

"""
## Create the Lindorm Vector Store object:
"""
logger.info("## Create the Lindorm Vector Store object:")

# import nest_asyncio

# nest_asyncio.apply()

host = "ld-bp******jm*******-proxy-search-pub.lindorm.aliyuncs.com"
port = 30070
username = "your_username"
password = "your_password"


index_name = "lindorm_rag_test"

nprobe = "2"

reorder_factor = "10"

client = LindormVectorClient(
    host,
    port,
    username,
    password,
    index=index_name,
    dimension=1536,  # match dimension of your embedding model
    nprobe=nprobe,
    reorder_factor=reorder_factor,
)

vector_store = LindormVectorStore(client)

"""
## Build the Index from the Documents:
"""
logger.info("## Build the Index from the Documents:")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context, show_progress=True
)

"""
## Querying the store:

### Search Test
"""
logger.info("## Querying the store:")

vector_retriever = index.as_retriever()
source_nodes = vector_retriever.retrieve("What did the author do growing up?")
for node in source_nodes:
    logger.debug(f"---------------------------------------------")
    logger.debug(f"Score: {node.score:.3f}")
    logger.debug(node.get_content())
    logger.debug(f"---------------------------------------------\n\n")

"""
### Basic Querying
"""
logger.info("### Basic Querying")

query_engine = index.as_query_engine(llm=dashscope_llm)
res = query_engine.query("What did the author do growing up?")
res.response

"""
### Metadata Filtering

Lindorm Vector Store now supports metadata filtering in the form of exact-match `key=value` pairs and range fliter in the form of `>`、`<`、`>=`、`<=` at query time.
"""
logger.info("### Metadata Filtering")


text_chunks = documents[0].text.split("\n\n")

footnotes = [
    Document(
        text=chunk,
        id=documents[0].doc_id,
        metadata={
            "is_footnote": bool(re.search(r"^\s*\[\d+\]\s*", chunk)),
            "mark_id": i,
        },
    )
    for i, chunk in enumerate(text_chunks)
    if bool(re.search(r"^\s*\[\d+\]\s*", chunk))
]

for f in footnotes:
    index.insert(f)

retriever = index.as_retriever(
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="is_footnote", value="true", operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="mark_id", value=0, operator=FilterOperator.GTE
            ),
        ],
        condition=FilterCondition.AND,
    ),
)

result = retriever.retrieve("What did the author about space aliens and lisp?")

logger.debug(result)

footnote_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="is_footnote", value="true", operator=FilterOperator.EQ
            ),
            MetadataFilter(
                key="mark_id", value=0, operator=FilterOperator.GTE
            ),
        ],
        condition=FilterCondition.AND,
    ),
    llm=dashscope_llm,
)

res = footnote_query_engine.query(
    "What did the author about space aliens and lisp?"
)
res.response

"""
### Hybrid Search

The Lindorm search support hybrid search, note the minimum search granularity of query str is one token.
"""
logger.info("### Hybrid Search")


retriever = index.as_retriever(
    vector_store_query_mode=VectorStoreQueryMode.HYBRID
)

result = retriever.retrieve("What did the author about space aliens and lisp?")

logger.debug(result)

query_engine = index.as_query_engine(
    llm=dashscope_llm, vector_store_query_mode=VectorStoreQueryMode.HYBRID
)
res = query_engine.query("What did the author about space aliens and lisp?")
res.response

logger.info("\n\n[DONE]", bright=True)