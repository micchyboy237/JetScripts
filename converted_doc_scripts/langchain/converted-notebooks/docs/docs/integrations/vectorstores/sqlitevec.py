from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVec
from langchain_text_splitters import CharacterTextSplitter
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
sidebar_label: SQLiteVec
---

# SQLite as a Vector Store with SQLiteVec

This notebook covers how to get started with the SQLiteVec vector store.

>[SQLite-Vec](https://alexgarcia.xyz/sqlite-vec/) is an `SQLite` extension designed for vector search, emphasizing local-first operations and easy integration into applications without external servers. It is the successor to [SQLite-VSS](https://alexgarcia.xyz/sqlite-vss/) by the same author. It is written in zero-dependency C and designed to be easy to build and use.

This notebook shows how to use the `SQLiteVec` vector database.

## Setup
You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
"""
logger.info("# SQLite as a Vector Store with SQLiteVec")

# %pip install --upgrade --quiet  sqlite-vec

"""
### Credentials
SQLiteVec does not require any credentials to use as the vector store is a simple SQLite file.

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
logger.info("### Credentials")


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = SQLiteVec(
    table="state_union", db_file="/tmp/vec.db", embedding=embedding_function
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

#
#
#
 
A
d
d
 
i
t
e
m
s
 
t
o
 
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

v
e
c
t
o
r
_
s
t
o
r
e
.
a
d
d
_
t
e
x
t
s
(
t
e
x
t
s
=
[
"
K
e
t
a
n
j
i

B
r
o
w
n

J
a
c
k
s
o
n

i
s

a
w
e
s
o
m
e
"
,

"
f
o
o
"
,

"
b
a
r
"
]
)

"""
### Update items in vector store
Not supported yet

### Delete items from vector store
Not supported yet

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

#
#
#
 
Q
u
e
r
y
 
d
i
r
e
c
t
l
y
"""
logger.info("### Update items in vector store")

d
a
t
a

=

v
e
c
t
o
r
_
s
t
o
r
e
.
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
_
s
e
a
r
c
h
(
"
K
e
t
a
n
j
i

B
r
o
w
n

J
a
c
k
s
o
n
"
,

k
=
4
)

"""
### Query by turning into retriever
Not supported yet

## Usage for retrieval-augmented generation
Refer to the documentation on sqlite-vec at https://alexgarcia.xyz/sqlite-vec/ for more information on how to use it for retrieval-augmented generation.

## API reference
For detailed documentation of all SQLiteVec features and configurations head to the API reference: https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.sqlitevec.SQLiteVec.html

#
#
#
 
O
t
h
e
r
 
e
x
a
m
p
l
e
s
"""
logger.info("### Query by turning into retriever")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = SQLiteVec.from_texts(
    texts=texts,
    embedding=embedding_function,
    table="state_union",
    db_file="/tmp/vec.db",
)

query = "What did the president say about Ketanji Brown Jackson"
data = db.similarity_search(query)

data[0].page_content

"""
#
#
#
 
E
x
a
m
p
l
e
 
u
s
i
n
g
 
e
x
i
s
t
i
n
g
 
S
Q
L
i
t
e
 
c
o
n
n
e
c
t
i
o
n
"""
logger.info("#")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
connection = SQLiteVec.create_connection(db_file="/tmp/vec.db")

db1 = SQLiteVec(
    table="state_union", embedding=embedding_function, connection=connection
)

db1.add_texts(["Ketanji Brown Jackson is awesome"])
query = "What did the president say about Ketanji Brown Jackson"
data = db1.similarity_search(query)

data[0].page_content

logger.info("\n\n[DONE]", bright=True)