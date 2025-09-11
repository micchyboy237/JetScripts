from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_neo4j import Neo4jVector
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
# Neo4j Vector Index

>[Neo4j](https://neo4j.com/) is an open-source graph database with integrated support for vector similarity search

It supports:

- approximate nearest neighbor search
- Euclidean similarity and cosine similarity
- Hybrid search combining vector and keyword searches

This notebook shows how to use the Neo4j vector index (`Neo4jVector`).

See the [installation instruction](https://neo4j.com/docs/operations-manual/current/installation/).
"""
logger.info("# Neo4j Vector Index")

# %pip install --upgrade --quiet  neo4j
# %pip install --upgrade --quiet  langchain-ollama langchain-neo4j
# %pip install --upgrade --quiet  tiktoken

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


loader = TextLoader("../../how_to/state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

url = "bolt://localhost:7687"
username = "neo4j"
password = "password"

"""
## Similarity Search with Cosine Distance (Default)
"""
logger.info("## Similarity Search with Cosine Distance (Default)")

db = Neo4jVector.from_documents(
    docs, OllamaEmbeddings(model="mxbai-embed-large"), url=url, username=username, password=password
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query, k=2)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
## Working with vectorstore

Above, we created a vectorstore from scratch. However, often times we want to work with an existing vectorstore.
In order to do that, we can initialize it directly.
"""
logger.info("## Working with vectorstore")

index_name = "vector"  # default index name

store = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
)

"""
We can also initialize a vectorstore from existing graph using the `from_existing_graph` method. This method pulls relevant text information from the database, and calculates and stores the text embeddings back to the database.
"""
logger.info("We can also initialize a vectorstore from existing graph using the `from_existing_graph` method. This method pulls relevant text information from the database, and calculates and stores the text embeddings back to the database.")

store.query(
    "CREATE (p:Person {name: 'Tomaz', location:'Slovenia', hobby:'Bicycle', age: 33})"
)

existing_graph = Neo4jVector.from_existing_graph(
    embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    node_label="Person",
    text_node_properties=["name", "location"],
    embedding_node_property="embedding",
)
result = existing_graph.similarity_search("Slovenia", k=1)

result[0]

"""
Neo4j also supports relationship vector indexes, where an embedding is stored as a relationship property and indexed. A relationship vector index cannot be populated via LangChain, but you can connect it to existing relationship vector indexes.
"""
logger.info("Neo4j also supports relationship vector indexes, where an embedding is stored as a relationship property and indexed. A relationship vector index cannot be populated via LangChain, but you can connect it to existing relationship vector indexes.")

store.query(
    "MERGE (p:Person {name: 'Tomaz'}) "
    "MERGE (p1:Person {name:'Leann'}) "
    "MERGE (p1)-[:FRIEND {text:'example text', embedding:$embedding}]->(p2)",
    params={"embedding": OllamaEmbeddings(
        model="mxbai-embed-large").embed_query("example text")},
)
relationship_index = "relationship_vector"
store.query(
    """
CREATE VECTOR INDEX $relationship_index
IF NOT EXISTS
FOR ()-[r:FRIEND]-() ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}}
""",
    params={"relationship_index": relationship_index},
)

relationship_vector = Neo4jVector.from_existing_relationship_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name=relationship_index,
    text_node_property="text",
)
relationship_vector.similarity_search("Example")

"""
### Metadata filtering

Neo4j vector store also supports metadata filtering by combining parallel runtime and exact nearest neighbor search.
_Requires Neo4j 5.18 or greater version._

Equality filtering has the following syntax.
"""
logger.info("### Metadata filtering")

existing_graph.similarity_search(
    "Slovenia",
    filter={"hobby": "Bicycle", "name": "Tomaz"},
)

"""
Metadata filtering also support the following operators:

* `$eq: Equal`
* `$ne: Not Equal`
* `$lt: Less than`
* `$lte: Less than or equal`
* `$gt: Greater than`
* `$gte: Greater than or equal`
* `$in: In a list of values`
* `$nin: Not in a list of values`
* `$between: Between two values`
* `$like: Text contains value`
* `$ilike: lowered text contains value`
"""
logger.info("Metadata filtering also support the following operators:")

existing_graph.similarity_search(
    "Slovenia",
    filter={"hobby": {"$eq": "Bicycle"}, "age": {"$gt": 15}},
)

"""
You can also use `OR` operator between filters
"""
logger.info("You can also use `OR` operator between filters")

existing_graph.similarity_search(
    "Slovenia",
    filter={"$or": [{"hobby": {"$eq": "Bicycle"}}, {"age": {"$gt": 15}}]},
)

"""
### Add documents
We can add documents to the existing vectorstore.
"""
logger.info("### Add documents")

store.add_documents([Document(page_content="foo")])

docs_with_score = store.similarity_search_with_score("foo")

docs_with_score[0]

"""
## Customize response with retrieval query

You can also customize responses by using a custom Cypher snippet that can fetch other information from the graph.
Under the hood, the final Cypher statement is constructed like so:

```
read_query = (
  "CALL db.index.vector.queryNodes($index, $k, $embedding) "
  "YIELD node, score "
) + retrieval_query
```

The retrieval query must return the following three columns:

* `text`: Union[str, Dict] = Value used to populate `page_content` of a document
* `score`: Float = Similarity score
* `metadata`: Dict = Additional metadata of a document

Learn more in this [blog post](https://medium.com/neo4j/implementing-rag-how-to-write-a-graph-retrieval-query-in-langchain-74abf13044f2).
"""
logger.info("## Customize response with retrieval query")

retrieval_query = """
RETURN "Name:" + node.name AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1)

"""
Here is an example of passing all node properties except for `embedding` as a dictionary to `text` column,
"""
logger.info("Here is an example of passing all node properties except for `embedding` as a dictionary to `text` column,")

retrieval_query = """
RETURN node {.name, .age, .hobby} AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1)

"""
You can also pass Cypher parameters to the retrieval query.
Parameters can be used for additional filtering, traversals, etc...
"""
logger.info("You can also pass Cypher parameters to the retrieval query.")

retrieval_query = """
RETURN node {.*, embedding:Null, extra: $extra} AS text, score, {foo:"bar"} AS metadata
"""
retrieval_example = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name="person_index",
    retrieval_query=retrieval_query,
)
retrieval_example.similarity_search("Foo", k=1, params={"extra": "ParamInfo"})

"""
## Hybrid search (vector + keyword)

Neo4j integrates both vector and keyword indexes, which allows you to use a hybrid search approach
"""
logger.info("## Hybrid search (vector + keyword)")

hybrid_db = Neo4jVector.from_documents(
    docs,
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    search_type="hybrid",
)

"""
To load the hybrid search from existing indexes, you have to provide both the vector and keyword indices
"""
logger.info("To load the hybrid search from existing indexes, you have to provide both the vector and keyword indices")

index_name = "vector"  # default index name
keyword_index_name = "keyword"  # default keyword index name

store = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=url,
    username=username,
    password=password,
    index_name=index_name,
    keyword_index_name=keyword_index_name,
    search_type="hybrid",
)

"""
## Retriever options

This section shows how to use `Neo4jVector` as a retriever.
"""
logger.info("## Retriever options")

retriever = store.as_retriever()
retriever.invoke(query)[0]

"""
## Question Answering with Sources

This section goes over how to do question-answering with sources over an Index. It does this by using the `RetrievalQAWithSourcesChain`, which does the lookup of the documents from an Index.
"""
logger.info("## Question Answering with Sources")


chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOllama(model="llama3.2"), chain_type="stuff", retriever=retriever
)

chain.invoke(
    {"question": "What did the president say about Justice Breyer"},
    return_only_outputs=True,
)

logger.info("\n\n[DONE]", bright=True)
