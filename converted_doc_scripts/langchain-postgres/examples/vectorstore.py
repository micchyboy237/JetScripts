from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

initialize_ollama_settings()

"""
# Vectorstore

This is an implementation of a LangChain vectorstore using `postgres` as the backend.

The `postgres` requires the the `pgvector` extension!

You can run the following command to spin up a pg-vector container:

```shell
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```

## Status

This code has been ported over from langchain_community. The following changes have been made:

* langchain_postgres works only with psycopg3. Please update your connnecion strings from `postgresql+psycopg2://...` to `postgresql+psycopg://langchain:langchain@...` (yes, it's the driver name is `psycopg` not `psycopg3`, but it'll use `psycopg3`.
* The schema of the embedding store and collection have been changed to make add_documents work correctly with user specified ids.
* One has to pass an explicit connection object now.


Currently, there is **no mechanism** that supports easy data migration on schema changes. So any schema changes in the vectorstore will require the user to recreate the tables and re-add the documents.
If this is a concern, please use a different vectorstore. If not, this implementation should be fine for your use case.

## Install dependencies
"""

# !pip install --quiet -U langchain_cohere

"""
## Initialize the vectorstore
"""


connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
# model = "mxbai-embed-large"
model = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=model)

embedding_length = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
delete_collection = True  # For testing

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    pre_delete_collection=delete_collection,
    embedding_length=embedding_length
)

"""
## Drop tables

If you need to drop tables (e.g., updating the embedding to a different dimension or just updating the embedding provider):

```python
vectorstore.drop_tables()
````

## Add documents

Add documents to the vectorstore
"""
# vectorstore.drop_tables()


docs = [
    Document(page_content='there are cats in the pond', metadata={
             "id": 1, "location": "pond", "topic": "animals"}),
    Document(page_content='ducks are also found in the pond', metadata={
             "id": 2, "location": "pond", "topic": "animals"}),
    Document(page_content='fresh apples are available at the market',
             metadata={"id": 3, "location": "market", "topic": "food"}),
    Document(page_content='the market also sells fresh oranges',
             metadata={"id": 4, "location": "market", "topic": "food"}),
    Document(page_content='the new art exhibit is fascinating',
             metadata={"id": 5, "location": "museum", "topic": "art"}),
    Document(page_content='a sculpture exhibit is also at the museum',
             metadata={"id": 6, "location": "museum", "topic": "art"}),
    Document(page_content='a new coffee shop opened on Main Street',
             metadata={"id": 7, "location": "Main Street", "topic": "food"}),
    Document(page_content='the book club meets at the library', metadata={
             "id": 8, "location": "library", "topic": "reading"}),
    Document(page_content='the library hosts a weekly story time for kids',
             metadata={"id": 9, "location": "library", "topic": "reading"}),
    Document(page_content='a cooking class for beginners is offered at the community center',
             metadata={"id": 10, "location": "community center", "topic": "classes"})
]

vectorstore.add_documents(docs, ids=[doc.metadata['id'] for doc in docs])

query = 'kitty'
results = vectorstore.similarity_search_with_relevance_scores(query, k=10)
logger.newline()
logger.debug(f"Query: \"{query}\" | Results: {len(results)}")
logger.success(results)

"""
Adding documents by ID will over-write any existing documents that match that ID.
"""

docs = [
    Document(page_content='there are cats in the pond', metadata={
             "id": 1, "location": "pond", "topic": "animals"}),
    Document(page_content='ducks are also found in the pond', metadata={
             "id": 2, "location": "pond", "topic": "animals"}),
    Document(page_content='fresh apples are available at the market',
             metadata={"id": 3, "location": "market", "topic": "food"}),
    Document(page_content='the market also sells fresh oranges',
             metadata={"id": 4, "location": "market", "topic": "food"}),
    Document(page_content='the new art exhibit is fascinating',
             metadata={"id": 5, "location": "museum", "topic": "art"}),
    Document(page_content='a sculpture exhibit is also at the museum',
             metadata={"id": 6, "location": "museum", "topic": "art"}),
    Document(page_content='a new coffee shop opened on Main Street',
             metadata={"id": 7, "location": "Main Street", "topic": "food"}),
    Document(page_content='the book club meets at the library', metadata={
             "id": 8, "location": "library", "topic": "reading"}),
    Document(page_content='the library hosts a weekly story time for kids',
             metadata={"id": 9, "location": "library", "topic": "reading"}),
    Document(page_content='a cooking class for beginners is offered at the community center',
             metadata={"id": 10, "location": "community center", "topic": "classes"})
]

"""
## Filtering Support

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.

| Operator  | Meaning/Category        |
|-----------|-------------------------|
| \$eq      | Equality (==)           |
| \$ne      | Inequality (!=)         |
| \$lt      | Less than (<)           |
| \$lte     | Less than or equal (<=) |
| \$gt      | Greater than (>)        |
| \$gte     | Greater than or equal (>=) |
| \$in      | Special Cased (in)      |
| \$nin     | Special Cased (not in)  |
| \$between | Special Cased (between) |
| \$exists  | Special Cased (is null) |
| \$like    | Text (like)             |
| \$ilike   | Text (case-insensitive like) |
| \$and     | Logical (and)           |
| \$or      | Logical (or)            |
"""

query = 'kitty'
results = vectorstore.similarity_search_with_relevance_scores(query, k=10, filter={
    'id': {'$in': [1, 5, 2, 9]}
})
logger.newline()
logger.debug(f"Query: \"{query}\" | Results: {len(results)}")
logger.success(results)

"""
If you provide a dict with multiple fields, but no operators, the top level will be interpreted as a logical **AND** filter
"""

query = 'ducks'
results = vectorstore.similarity_search_with_relevance_scores(query, k=10, filter={
    'id': {'$in': [1, 5, 2, 9]},
    'location': {'$in': ["pond", "market"]}
})
logger.newline()
logger.debug(f"Query: \"{query}\" | Results: {len(results)}")
logger.success(results)

query = 'ducks'
results = vectorstore.similarity_search_with_relevance_scores(query, k=10, filter={
    '$and': [
        {'id': {'$in': [1, 5, 2, 9]}},
        {'location': {'$in': ["pond", "market"]}},
    ]
}
)
logger.newline()
logger.debug(f"Query: \"{query}\" | Results: {len(results)}")
logger.success(results)

query = 'bird'
results = vectorstore.similarity_search_with_relevance_scores(query, k=10, filter={
    'location': {"$ne": 'pond'}
})
logger.newline()
logger.debug(f"Query: \"{query}\" | Results: {len(results)}")
logger.success(results)

logger.info("\n\n[DONE]", bright=True)
