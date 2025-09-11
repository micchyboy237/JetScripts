from hdbcli import dbapi
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.base import (
StructuredQueryOutputParser,
get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.hanavector import HanaTranslator
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.documents import Document
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

For more information on how to setup the SAP HANA vetor store, take a look at the [documentation](/docs/integrations/vectorstores/sap_hanavector.ipynb).

We use the same setup here:
"""
logger.info("# SAP HANA Cloud Vector Engine")



connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    sslValidateCertificate=False,
)

"""
To be able to self query with good performance we create additional metadata fields
for our vectorstore table in HANA:
"""
logger.info("To be able to self query with good performance we create additional metadata fields")

cur = connection.cursor()
cur.execute("DROP TABLE LANGCHAIN_DEMO_SELF_QUERY", ignoreErrors=True)
cur.execute(
    (
        """CREATE TABLE "LANGCHAIN_DEMO_SELF_QUERY"  (
        "name" NVARCHAR(100), "is_active" BOOLEAN, "id" INTEGER, "height" DOUBLE,
        "VEC_TEXT" NCLOB,
        "VEC_META" NCLOB,
        "VEC_VECTOR" REAL_VECTOR
        )"""
    )
)

"""
Let's add some documents.
"""
logger.info("Let's add some documents.")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

docs = [
    Document(
        page_content="First",
        metadata={"name": "adam", "is_active": True, "id": 1, "height": 10.0},
    ),
    Document(
        page_content="Second",
        metadata={"name": "bob", "is_active": False, "id": 2, "height": 5.7},
    ),
    Document(
        page_content="Third",
        metadata={"name": "jane", "is_active": True, "id": 3, "height": 2.4},
    ),
]

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_SELF_QUERY",
    specific_metadata_columns=["name", "is_active", "id", "height"],
)

db.delete(filter={})
db.add_documents(docs)

"""
## Self querying

Now for the main act: here is how to construct a SelfQueryRetriever for HANA vectorstore:
"""
logger.info("## Self querying")


llm = ChatOllama(model="llama3.2")

metadata_field_info = [
    AttributeInfo(
        name="name",
        description="The name of the person",
        type="string",
    ),
    AttributeInfo(
        name="is_active",
        description="Whether the person is active",
        type="boolean",
    ),
    AttributeInfo(
        name="id",
        description="The ID of the person",
        type="integer",
    ),
    AttributeInfo(
        name="height",
        description="The height of the person",
        type="float",
    ),
]

document_content_description = "A collection of persons"

hana_translator = HanaTranslator()

retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    document_content_description,
    metadata_field_info,
    structured_query_translator=hana_translator,
)

"""
Let's use this retriever to prepare a (self) query for a person:
"""
logger.info("Let's use this retriever to prepare a (self) query for a person:")

query_prompt = "Which person is not active?"

docs = retriever.invoke(input=query_prompt)
for doc in docs:
    logger.debug("-" * 80)
    logger.debug(doc.page_content, " ", doc.metadata)

"""
We can also take a look at how the query is being constructed:
"""
logger.info("We can also take a look at how the query is being constructed:")


prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

sq = query_constructor.invoke(input=query_prompt)

logger.debug("Structured query: ", sq)

logger.debug("Translated for hana vector store: ", hana_translator.visit_structured_query(sq))

logger.info("\n\n[DONE]", bright=True)