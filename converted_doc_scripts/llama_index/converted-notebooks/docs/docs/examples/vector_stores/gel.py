from datetime import datetime
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
MetadataFilter,
MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.gel import GelVectorStore
import csv
import openai
import os
import re
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/postgres.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Gel Vector Store
[Gel](https://www.geldata.com/) is an open-source PostgreSQL data layer optimized for fast development to production cycle. It comes with a high-level strictly typed graph-like data model, composable hierarchical query language, full SQL support, migrations, Auth and AI modules.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Gel Vector Store")

# ! pip install gel llama-index-vector-stores-gel

# ! pip install llama-index


"""
### Setup MLX
The first step is to configure the openai key. It will be used to created embeddings for the documents loaded into the index
"""
logger.info("### Setup MLX")


# os.environ["OPENAI_API_KEY"] = "<your key>"
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug("Document ID:", documents[0].doc_id)

"""
### Create the Database

In order to use Gel as a backend for your vectorstore, you're going to need a working Gel instance.
Fortunately, it doesn't have to involve Docker containers or anything complicated, unless you want to!

To set up a local instance, run:
"""
logger.info("### Create the Database")

# ! gel project init --non-interactive

"""
If you are using [Gel Cloud](cloud.geldata.com) (and you should!), add one more argument to that command:

```bash
gel project init --server-instance <org-name>/<instance-name>
```

For a comprehensive list of ways to run Gel, take a look at [Running Gel](https://docs.geldata.com/reference/running) section of the reference docs.

### Set up the schema

[Gel schema](https://docs.geldata.com/reference/datamodel) is an explicit high-level description of your application's data model. 
Aside from enabling you to define exactly how your data is going to be laid out, it drives Gel's many powerful features such as links, access policies, functions, triggers, constraints, indexes, and more.

The LlamaIndex's `GelVectorStore` expects the following layout for the schema:
"""
logger.info("### Set up the schema")

schema_content = """
using extension pgvector;

module default {
    scalar type EmbeddingVector extending ext::pgvector::vector<1536>;

    type Record {
        required collection: str;
        text: str;
        embedding: EmbeddingVector;
        external_id: str {
            constraint exclusive;
        };
        metadata: json;

        index ext::pgvector::hnsw_cosine(m := 16, ef_construction := 128)
            on (.embedding)
    }
}
""".strip()

with open("dbschema/default.gel", "w") as f:
    f.write(schema_content)

"""
In order to apply schema changes to the database, run the migration using the Gel's [migration tool](https://docs.geldata.com/reference/datamodel/migrations):
"""
logger.info("In order to apply schema changes to the database, run the migration using the Gel's [migration tool](https://docs.geldata.com/reference/datamodel/migrations):")

# ! gel migration create --non-interactive
# ! gel migrate

"""
From this point onward, `GelVectorStore` can be used as a drop-in replacement for any other vectorstore available in LlamaIndex.

### Create the index
"""
logger.info("### Create the index")

vector_store = GelVectorStore()

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

"""
### Query the index
We can now ask questions using our index.
"""
logger.info("### Query the index")

response = query_engine.query("What did the author do?")

logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What happened in the mid 1980s?")

logger.debug(textwrap.fill(str(response), 100))

"""
### Metadata filters

GelVectorStore supports storing metadata in nodes, and filtering based on that metadata during the retrieval step.

#### Download git commits dataset
"""
logger.info("### Metadata filters")

# !mkdir -p 'data/git_commits/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/csv/commit_history.csv' -O 'data/git_commits/commit_history.csv'


with open(f"{GENERATED_DIR}/git_commits/commit_history.csv", "r") as f:
    commits = list(csv.DictReader(f))

logger.debug(commits[0])
logger.debug(len(commits))

"""
#### Add nodes with custom metadata
"""
logger.info("#### Add nodes with custom metadata")


nodes = []
dates = set()
authors = set()
for commit in commits[:100]:
    author_email = commit["author"].split("<")[1][:-1]
    commit_date = datetime.strptime(
        commit["date"], "%a %b %d %H:%M:%S %Y %z"
    ).strftime("%Y-%m-%d")
    commit_text = commit["change summary"]
    if commit["change details"]:
        commit_text += "\n\n" + commit["change details"]
    fixes = re.findall(r"#(\d+)", commit_text, re.IGNORECASE)
    nodes.append(
        TextNode(
            text=commit_text,
            metadata={
                "commit_date": commit_date,
                "author": author_email,
                "fixes": fixes,
            },
        )
    )
    dates.add(commit_date)
    authors.add(author_email)

logger.debug(nodes[0])
logger.debug(min(dates), "to", max(dates))
logger.debug(authors)

vector_store = GelVectorStore()

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
index.insert_nodes(nodes)

logger.debug(index.as_query_engine().query("How did Lakshmi fix the segfault?"))

"""
#### Apply metadata filters

Now we can filter by commit author or by date when retrieving nodes.
"""
logger.info("#### Apply metadata filters")


filters = MetadataFilters(
    filters=[
        MetadataFilter(key="author", value="mats@timescale.com"),
        MetadataFilter(key="author", value="sven@timescale.com"),
    ],
    condition="or",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-15", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-25", operator="<="),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
#### Apply nested filters

In the above examples, we combined multiple filters using AND or OR. We can also combine multiple sets of filters.
"""
logger.info("#### Apply nested filters")

filters = MetadataFilters(
    filters=[
        MetadataFilters(
            filters=[
                MetadataFilter(
                    key="commit_date", value="2023-08-01", operator=">="
                ),
                MetadataFilter(
                    key="commit_date", value="2023-08-15", operator="<="
                ),
            ],
            condition="and",
        ),
        MetadataFilters(
            filters=[
                MetadataFilter(key="author", value="mats@timescale.com"),
                MetadataFilter(key="author", value="sven@timescale.com"),
            ],
            condition="or",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
The above can be simplified by using the IN operator. `GelVectorStore` supports `in`, `nin`, and `contains` for comparing an element with a list.
"""
logger.info("The above can be simplified by using the IN operator. `GelVectorStore` supports `in`, `nin`, and `contains` for comparing an element with a list.")

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-01", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-15", operator="<="),
        MetadataFilter(
            key="author",
            value=["mats@timescale.com", "sven@timescale.com"],
            operator="in",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-01", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-15", operator="<="),
        MetadataFilter(
            key="author",
            value=["mats@timescale.com", "sven@timescale.com"],
            operator="nin",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="fixes", value="5680", operator="contains"),
    ]
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("How did these commits fix the issue?")
for node in retrieved_nodes:
    logger.debug(node.node.metadata)

logger.info("\n\n[DONE]", bright=True)