from dotenv import load_dotenv
from hdbcli import dbapi
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_hana import HanaDB
from langchain_hana import HanaInternalEmbeddings
from langchain_hana.utils import DistanceStrategy
from langchain_text_splitters import CharacterTextSplitter
import EmbeddingTabs from "@theme/EmbeddingTabs"
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
# SAP HANA Cloud Vector Engine

>[SAP HANA Cloud Vector Engine](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/sap-hana-cloud-sap-hana-database-vector-engine-guide) is a vector store fully integrated into the `SAP HANA Cloud` database.

## Setup

Install the `langchain-hana` external integration package, as well as the other packages used throughout this notebook.
"""
logger.info("# SAP HANA Cloud Vector Engine")

# %pip install -qU langchain-hana

"""
### Credentials

Ensure your SAP HANA instance is running. Load your credentials from environment variables and create a connection:
"""
logger.info("### Credentials")


load_dotenv()
connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    sslValidateCertificate=False,
)

"""
Learn more about SAP HANA in [What is SAP HANA?](https://www.sap.com/products/data-cloud/hana/what-is-sap-hana.html).

### Initialization
To initialize a `HanaDB` vector store, you need a database connection and an embedding instance. SAP HANA Cloud Vector Engine supports both external and internal embeddings.

- #### Using External Embeddings


<EmbeddingTabs/>
"""
logger.info("### Initialization")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
- #### Using Internal Embeddings

Alternatively, you can compute embeddings directly in SAP HANA using its native `VECTOR_EMBEDDING()` function. To enable this, create an instance of `HanaInternalEmbeddings` with your internal model ID and pass it to `HanaDB`. Note that the `HanaInternalEmbeddings` instance is specifically designed for use with `HanaDB` and is not intended for use with other vector store implementations. For more information about internal embedding, see the [SAP HANA VECTOR_EMBEDDING Function](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/vector-embedding-function-vector).

> **Caution:** Ensure NLP is enabled in your SAP HANA Cloud instance.
"""
logger.info(
    "Alternatively, you can compute embeddings directly in SAP HANA using its native `VECTOR_EMBEDDING()` function. To enable this, create an instance of `HanaInternalEmbeddings` with your internal model ID and pass it to `HanaDB`. Note that the `HanaInternalEmbeddings` instance is specifically designed for use with `HanaDB` and is not intended for use with other vector store implementations. For more information about internal embedding, see the [SAP HANA VECTOR_EMBEDDING Function](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/vector-embedding-function-vector).")


embeddings = HanaInternalEmbeddings(
    internal_embedding_model_id="SAP_NEB.20240715")

"""
Once you have your connection and embedding instance, create the vector store by passing them to `HanaDB` along with a table name for storing vectors:
"""
logger.info("Once you have your connection and embedding instance, create the vector store by passing them to `HanaDB` along with a table name for storing vectors:")


db = HanaDB(
    embedding=embeddings, connection=connection, table_name="STATE_OF_THE_UNION"
)

"""
## Example

Load the sample document "state_of_the_union.txt" and create chunks from it.
"""
logger.info("## Example")


text_documents = TextLoader(
    "../../how_to/state_of_the_union.txt", encoding="UTF-8"
).load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_chunks = text_splitter.split_documents(text_documents)
logger.debug(f"Number of document chunks: {len(text_chunks)}")

"""
Add the loaded document chunks to the table. For this example, we delete any previous content from the table which might exist from previous runs.
"""
logger.info("Add the loaded document chunks to the table. For this example, we delete any previous content from the table which might exist from previous runs.")

db.delete(filter={})

db.add_documents(text_chunks)

"""
Perform a query to get the two best-matching document chunks from the ones that were added in the previous step.
By default "Cosine Similarity" is used for the search.
"""
logger.info("Perform a query to get the two best-matching document chunks from the ones that were added in the previous step.")

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query, k=2)

for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

"""
Query the same content with "Euclidian Distance". The results shoud be the same as with "Cosine Similarity".
"""
logger.info("Query the same content with "Euclidian Distance". The results shoud be the same as with "Cosine Similarity".")


db = HanaDB(
    embedding=embeddings,
    connection=connection,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    table_name="STATE_OF_THE_UNION",
)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query, k=2)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

"""
## Maximal Marginal Relevance Search (MMR)

`Maximal marginal relevance` optimizes for similarity to query AND diversity among selected documents. The first 20 (fetch_k) items will be retrieved from the DB. The MMR algorithm will then find the best 2 (k) matches.
"""
logger.info("## Maximal Marginal Relevance Search (MMR)")

docs = db.max_marginal_relevance_search(query, k=2, fetch_k=20)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

"""
## Creating an HNSW Vector Index

A vector index can significantly speed up top-k nearest neighbor queries for vectors. Users can create a Hierarchical Navigable Small World (HNSW) vector index using the `create_hnsw_index` function.

For more information about creating an index at the database level, please refer to the [official documentation](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/create-vector-index-statement-data-definition).
"""
logger.info("## Creating an HNSW Vector Index")

db_cosine = HanaDB(
    embedding=embeddings, connection=connection, table_name="STATE_OF_THE_UNION"
)

# If no other parameters are specified, the default values will be used
db_cosine.create_hnsw_index()


db_l2 = HanaDB(
    embedding=embeddings,
    connection=connection,
    table_name="STATE_OF_THE_UNION",
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,  # Specify L2 distance
)

db_l2.create_hnsw_index(
    index_name="STATE_OF_THE_UNION_L2_index",
    m=100,  # Max number of neighbors per graph node (valid range: 4 to 1000)
    # Max number of candidates during graph construction (valid range: 1 to 100000)
    ef_construction=200,
    # Min number of candidates during the search (valid range: 1 to 100000)
    ef_search=500,
)

docs = db_l2.max_marginal_relevance_search(query, k=2, fetch_k=20)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

"""
**Key Points**:
- **Similarity Function**: The similarity function for the index is **cosine similarity** by default. If you want to use a different similarity function (e.g., `L2` distance), you need to specify it when initializing the `HanaDB` instance.
- **Default Parameters**: In the `create_hnsw_index` function, if the user does not provide custom values for parameters like `m`, `ef_construction`, or `ef_search`, the default values (e.g., `m=64`, `ef_construction=128`, `ef_search=200`) will be used automatically. These values ensure the index is created with reasonable performance without requiring user intervention.

## Basic Vectorstore Operations
"""
logger.info("## Basic Vectorstore Operations")

db = HanaDB(
    connection=connection, embedding=embeddings, table_name="LANGCHAIN_DEMO_BASIC"
)

db.delete(filter={})

"""
We can add simple text documents to the existing table.
"""
logger.info("We can add simple text documents to the existing table.")

docs = [Document(page_content="Some text"),
        Document(page_content="Other docs")]
db.add_documents(docs)

"""
Add documents with metadata.
"""
logger.info("Add documents with metadata.")

docs = [
    Document(
        page_content="foo",
        metadata={"start": 100, "end": 150,
                  "doc_name": "foo.txt", "quality": "bad"},
    ),
    Document(
        page_content="bar",
        metadata={"start": 200, "end": 250,
                  "doc_name": "bar.txt", "quality": "good"},
    ),
]
db.add_documents(docs)

"""
Query documents with specific metadata.
"""
logger.info("Query documents with specific metadata.")

docs = db.similarity_search("foobar", k=2, filter={"quality": "bad"})
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)
    logger.debug(doc.metadata)

"""
Delete documents with specific metadata.
"""
logger.info("Delete documents with specific metadata.")

db.delete(filter={"quality": "bad"})

docs = db.similarity_search("foobar", k=2, filter={"quality": "bad"})
logger.debug(len(docs))

"""
## Advanced filtering
In addition to the basic value-based filtering capabilities, it is possible to use more advanced filtering.
The table below shows the available filter operators.

| Operator | Semantic                 |
|----------|-------------------------|
| `$eq`    | Equality (==)           |
| `$ne`    | Inequality (!=)         |
| `$lt`    | Less than (&lt;)           |
| `$lte`   | Less than or equal (&lt;=) |
| `$gt`    | Greater than (>)        |
| `$gte`   | Greater than or equal (>=) |
| `$in`    | Contained in a set of given values  (in)    |
| `$nin`   | Not contained in a set of given values  (not in)  |
| `$between` | Between the range of two boundary values |
| `$like`  | Text equality based on the "LIKE" semantics in SQL (using "%" as wildcard)  |
| `$contains` | Filters documents containing a specific keyword |
| `$and`   | Logical "and", supporting 2 or more operands |
| `$or`    | Logical "or", supporting 2 or more operands |
"""
logger.info("## Advanced filtering")

docs = [
    Document(
        page_content="First",
        metadata={"name": "Adam Smith",
                  "is_active": True, "id": 1, "height": 10.0},
    ),
    Document(
        page_content="Second",
        metadata={"name": "Bob Johnson",
                  "is_active": False, "id": 2, "height": 5.7},
    ),
    Document(
        page_content="Third",
        metadata={"name": "Jane Doe",
                  "is_active": True, "id": 3, "height": 2.4},
    ),
]

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_ADVANCED_FILTER",
)

db.delete(filter={})
db.add_documents(docs)


def print_filter_result(result):
    if len(result) == 0:
        logger.debug("<empty result>")
    for doc in result:
        logger.debug(doc.metadata)


"""
Filtering with `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
"""
logger.info("Filtering with `$ne`, `$gt`, `$gte`, `$lt`, `$lte`")

advanced_filter = {"id": {"$ne": 1}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$gt": 1}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$gte": 1}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$lt": 1}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"id": {"$lte": 1}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

"""
Filtering with `$between`, `$in`, `$nin`
"""
logger.info("Filtering with `$between`, `$in`, `$nin`")

advanced_filter = {"id": {"$between": (1, 2)}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$in": ["Adam Smith", "Bob Johnson"]}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$nin": ["Adam Smith", "Bob Johnson"]}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

"""
Text filtering with `$like`
"""
logger.info("Text filtering with `$like`")

advanced_filter = {"name": {"$like": "a%"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$like": "%a%"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

"""
Text filtering with `$contains`
"""
logger.info("Text filtering with `$contains`")

advanced_filter = {"name": {"$contains": "bob"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$contains": "bo"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$contains": "Adam Johnson"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"name": {"$contains": "Adam Smith"}}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

"""
Combined filtering with `$and`, `$or`
"""
logger.info("Combined filtering with `$and`, `$or`")

advanced_filter = {"$or": [{"id": 1}, {"name": "bob"}]}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"$and": [{"id": 1}, {"id": 2}]}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {"$or": [{"id": 1}, {"id": 2}, {"id": 3}]}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

advanced_filter = {
    "$and": [{"name": {"$contains": "bob"}}, {"name": {"$contains": "johnson"}}]
}
logger.debug(f"Filter: {advanced_filter}")
print_filter_result(db.similarity_search(
    "just testing", k=5, filter=advanced_filter))

"""
## Using a VectorStore as a retriever in chains for retrieval augmented generation (RAG)
"""
logger.info(
    "## Using a VectorStore as a retriever in chains for retrieval augmented generation (RAG)")

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_RETRIEVAL_CHAIN",
)

db.delete(filter={})

db.add_documents(text_chunks)

retriever = db.as_retriever()

"""
Define the prompt.
"""
logger.info("Define the prompt.")


prompt_template = """
You are an expert in state of the union topics. You are provided multiple context items that are related to the prompt you have to answer.
Use the following pieces of context to answer the question at the end.

'''
{context}
'''

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

"""
Create the ConversationalRetrievalChain, which handles the chat history and the retrieval of similar document chunks to be added to the prompt.
"""
logger.info("Create the ConversationalRetrievalChain, which handles the chat history and the retrieval of similar document chunks to be added to the prompt.")


llm = ChatOllama(model="llama3.2")
memory = ConversationBufferMemory(
    memory_key="chat_history", output_key="answer", return_messages=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

"""
Ask the first question (and verify how many text chunks have been used).
"""
logger.info(
    "Ask the first question (and verify how many text chunks have been used).")

question = "What about Mexico and Guatemala?"

result = qa_chain.invoke({"question": question})
logger.debug("Answer from LLM:")
logger.debug("================")
logger.debug(result["answer"])

source_docs = result["source_documents"]
logger.debug("================")
logger.debug(f"Number of used source document chunks: {len(source_docs)}")

"""
Examine the used chunks of the chain in detail. Check if the best ranked chunk contains info about "Mexico and Guatemala" as mentioned in the question.
"""
logger.info("Examine the used chunks of the chain in detail. Check if the best ranked chunk contains info about "Mexico and Guatemala" as mentioned in the question.")

for doc in source_docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)
    logger.debug(doc.metadata)

"""
Ask another question on the same conversational chain. The answer should relate to the previous answer given.
"""
logger.info("Ask another question on the same conversational chain. The answer should relate to the previous answer given.")

question = "How many casualties were reported after that?"

result = qa_chain.invoke({"question": question})
logger.debug("Answer from LLM:")
logger.debug("================")
logger.debug(result["answer"])

"""
## Standard tables vs. "custom" tables with vector data

As default behaviour, the table for the embeddings is created with 3 columns:

- A column `VEC_TEXT`, which contains the text of the Document
- A column `VEC_META`, which contains the metadata of the Document
- A column `VEC_VECTOR`, which contains the embeddings-vector of the Document's text
"""
logger.info("## Standard tables vs. "custom" tables with vector data")

db = HanaDB(
    connection=connection, embedding=embeddings, table_name="LANGCHAIN_DEMO_NEW_TABLE"
)

db.delete(filter={})

docs = [
    Document(
        page_content="A simple document",
        metadata={"start": 100, "end": 150, "doc_name": "simple.txt"},
    )
]
db.add_documents(docs)

"""
Show the columns in table "LANGCHAIN_DEMO_NEW_TABLE"
"""
logger.info("Show the columns in table "LANGCHAIN_DEMO_NEW_TABLE"")

cur = connection.cursor()
cur.execute(
    "SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = 'LANGCHAIN_DEMO_NEW_TABLE'"
)
rows = cur.fetchall()
for row in rows:
    logger.debug(row)
cur.close()

"""
Show the value of the inserted document in the three columns
"""
logger.info("Show the value of the inserted document in the three columns")

cur = connection.cursor()
cur.execute(
    "SELECT VEC_TEXT, VEC_META, TO_NVARCHAR(VEC_VECTOR) FROM LANGCHAIN_DEMO_NEW_TABLE LIMIT 1"
)
rows = cur.fetchall()
logger.debug(rows[0][0])  # The text
logger.debug(rows[0][1])  # The metadata
logger.debug(rows[0][2])  # The vector
cur.close()

"""
Custom tables must have at least three columns that match the semantics of a standard table

- A column with type `NCLOB` or `NVARCHAR` for the text/context of the embeddings
- A column with type `NCLOB` or `NVARCHAR` for the metadata 
- A column with type `REAL_VECTOR` for the embedding vector

The table can contain additional columns. When new Documents are inserted into the table, these additional columns must allow NULL values.
"""
logger.info(
    "Custom tables must have at least three columns that match the semantics of a standard table")

my_own_table_name = "MY_OWN_TABLE_ADD"
cur = connection.cursor()
cur.execute(
    (
        f"CREATE TABLE {my_own_table_name} ("
        "SOME_OTHER_COLUMN NVARCHAR(42), "
        "MY_TEXT NVARCHAR(2048), "
        "MY_METADATA NVARCHAR(1024), "
        "MY_VECTOR REAL_VECTOR )"
    )
)

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name=my_own_table_name,
    content_column="MY_TEXT",
    metadata_column="MY_METADATA",
    vector_column="MY_VECTOR",
)

docs = [
    Document(
        page_content="Some other text",
        metadata={"start": 400, "end": 450, "doc_name": "other.txt"},
    )
]
db.add_documents(docs)

cur.execute(f"SELECT * FROM {my_own_table_name} LIMIT 1")
rows = cur.fetchall()
# Value of column "SOME_OTHER_DATA". Should be NULL/None
logger.debug(rows[0][0])
logger.debug(rows[0][1])  # The text
logger.debug(rows[0][2])  # The metadata
logger.debug(rows[0][3])  # The vector

cur.close()

"""
Add another document and perform a similarity search on the custom table.
"""
logger.info(
    "Add another document and perform a similarity search on the custom table.")

docs = [
    Document(
        page_content="Some more text",
        metadata={"start": 800, "end": 950, "doc_name": "more.txt"},
    )
]
db.add_documents(docs)

query = "What's up?"
docs = db.similarity_search(query, k=2)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

"""
### Filter Performance Optimization with Custom Columns

To allow flexible metadata values, all metadata is stored as JSON in the metadata column by default. If some of the used metadata keys and value types are known, they can be stored in additional columns instead by creating the target table with the key names as column names and passing them to the HanaDB constructor via the specific_metadata_columns list. Metadata keys that match those values are copied into the special column during insert. Filters use the special columns instead of the metadata JSON column for keys in the specific_metadata_columns list.
"""
logger.info("### Filter Performance Optimization with Custom Columns")

my_own_table_name = "PERFORMANT_CUSTOMTEXT_FILTER"
cur = connection.cursor()
cur.execute(
    (
        f"CREATE TABLE {my_own_table_name} ("
        "CUSTOMTEXT NVARCHAR(500), "
        "MY_TEXT NVARCHAR(2048), "
        "MY_METADATA NVARCHAR(1024), "
        "MY_VECTOR REAL_VECTOR )"
    )
)

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name=my_own_table_name,
    content_column="MY_TEXT",
    metadata_column="MY_METADATA",
    vector_column="MY_VECTOR",
    specific_metadata_columns=["CUSTOMTEXT"],
)

docs = [
    Document(
        page_content="Some other text",
        metadata={
            "start": 400,
            "end": 450,
            "doc_name": "other.txt",
            "CUSTOMTEXT": "Filters on this value are very performant",
        },
    )
]
db.add_documents(docs)

cur.execute(f"SELECT * FROM {my_own_table_name} LIMIT 1")
rows = cur.fetchall()
logger.debug(
    rows[0][0]
)  # Value of column "CUSTOMTEXT". Should be "Filters on this value are very performant"
logger.debug(rows[0][1])  # The text
logger.debug(
    rows[0][2]
)  # The metadata without the "CUSTOMTEXT" data, as this is extracted into a sperate column
logger.debug(rows[0][3])  # The vector

cur.close()

"""
The special columns are completely transparent to the rest of the langchain interface. Everything works as it did before, just more performant.
"""
logger.info("The special columns are completely transparent to the rest of the langchain interface. Everything works as it did before, just more performant.")

docs = [
    Document(
        page_content="Some more text",
        metadata={
            "start": 800,
            "end": 950,
            "doc_name": "more.txt",
            "CUSTOMTEXT": "Another customtext value",
        },
    )
]
db.add_documents(docs)

advanced_filter = {"CUSTOMTEXT": {"$like": "%value%"}}
query = "What's up?"
docs = db.similarity_search(query, k=2, filter=advanced_filter)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content)

logger.info("\n\n[DONE]", bright=True)
