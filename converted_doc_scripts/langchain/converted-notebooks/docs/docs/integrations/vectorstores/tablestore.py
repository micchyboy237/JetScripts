from jet.logger import logger
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import TablestoreVectorStore
from langchain_core.documents import Document
import os
import shutil
import tablestore


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
# Tablestore

[Tablestore](https://www.aliyun.com/product/ots) is a fully managed NoSQL cloud database service.

Tablestore enables storage of a massive amount of structured and semi-structured data.

This notebook shows how to use functionality related to the `Tablestore` vector database.

To use Tablestore, you must create an instance.
Here are the [creating instance instructions](https://help.aliyun.com/zh/tablestore/getting-started/manage-the-wide-column-model-in-the-tablestore-console).

#
#
 
S
e
t
u
p
"""
logger.info("# Tablestore")

# %pip install --upgrade --quiet  langchain-community tablestore

"""
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
a
t
i
o
n
"""
logger.info("#")

# import getpass

# os.environ["end_point"] = getpass.getpass("Tablestore end_point:")
# os.environ["instance_name"] = getpass.getpass("Tablestore instance_name:")
# os.environ["access_key_id"] = getpass.getpass("Tablestore access_key_id:")
# os.environ["access_key_secret"] = getpass.getpass("Tablestore access_key_secret:")

"""
C
r
e
a
t
e
 
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
.
"""
logger.info("C")


test_embedding_dimension_size = 4
embeddings = FakeEmbeddings(size=test_embedding_dimension_size)

store = TablestoreVectorStore(
    embedding=embeddings,
    endpoint=os.getenv("end_point"),
    instance_name=os.getenv("instance_name"),
    access_key_id=os.getenv("access_key_id"),
    access_key_secret=os.getenv("access_key_secret"),
    vector_dimension=test_embedding_dimension_size,
    metadata_mappings=[
        tablestore.FieldSchema(
            "type", tablestore.FieldType.KEYWORD, index=True, enable_sort_and_agg=True
        ),
        tablestore.FieldSchema(
            "time", tablestore.FieldType.LONG, index=True, enable_sort_and_agg=True
        ),
    ],
)

"""
#
#
 
M
a
n
a
g
e
 
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

C
r
e
a
t
e
 
t
a
b
l
e
 
a
n
d
 
i
n
d
e
x
.
"""
logger.info("#")

store.create_table_if_not_exist()
store.create_search_index_if_not_exist()

"""
A
d
d
 
d
o
c
u
m
e
n
t
s
.
"""
logger.info("A")

store.add_documents(
    [
        Document(
            id="1", page_content="1 hello world", metadata={"type": "pc", "time": 2000}
        ),
        Document(
            id="2", page_content="abc world", metadata={"type": "pc", "time": 2009}
        ),
        Document(
            id="3", page_content="3 text world", metadata={"type": "sky", "time": 2010}
        ),
        Document(
            id="4", page_content="hi world", metadata={"type": "sky", "time": 2030}
        ),
        Document(
            id="5", page_content="hi world", metadata={"type": "sky", "time": 2030}
        ),
    ]
)

"""
D
e
l
e
t
e
 
d
o
c
u
m
e
n
t
.
"""
logger.info("D")

store.delete(["3"])

"""
G
e
t
 
d
o
c
u
m
e
n
t
s
.

#
#
 
Q
u
e
r
y
 
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

store.get_by_ids(["1", "3", "5"])

"""
S
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
.
"""
logger.info("S")

store.similarity_search(query="hello world", k=2)

"""
S
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
 
w
i
t
h
 
f
i
l
t
e
r
s
.
"""
logger.info("S")

store.similarity_search(
    query="hello world",
    k=10,
    tablestore_filter_query=tablestore.BoolQuery(
        must_queries=[tablestore.TermQuery(field_name="type", column_value="sky")],
        should_queries=[tablestore.RangeQuery(field_name="time", range_from=2020)],
        must_not_queries=[tablestore.TermQuery(field_name="type", column_value="pc")],
    ),
)

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval)

## API reference

For detailed documentation of all `TablestoreVectorStore` features and configurations head to the API reference:
 https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.tablestore.TablestoreVectorStore.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)