from jet.adapters.langchain.chat_ollama import AzureChatOllama, AzureOllamaEmbeddings
from jet.logger import logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_sqlserver import SQLServer_VectorStore
from typing import List, Tuple
import os
import pandas as pd
import pyodbc
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
# SQLServer

>Azure SQL provides a dedicated [Vector data type](https:\learn.microsoft.com\sql\t-sql\data-types\vector-data-type?view=azuresqldb-current&viewFallbackFrom=sql-server-ver16&tabs=csharp-sample) that simplifies the creation, storage, and querying of vector embeddings directly within a relational database. This eliminates the need for separate vector databases and related integrations, increasing the security of your solutions while reducing the overall complexity.

Azure SQL is a robust service that combines scalability, security, and high availability, providing all the benefits of a modern database solution. It leverages a sophisticated query optimizer and enterprise features to perform vector similarity searches alongside traditional SQL queries, enhancing data analysis and decision-making.  
  
Read more on using [Intelligent applications with Azure SQL Database](https://learn.microsoft.com/azure/azure-sql/database/ai-artificial-intelligence-intelligent-applications?view=azuresql)

This notebook shows you how to leverage this integrated SQL [vector database](https://devblogs.microsoft.com/azure-sql/exciting-announcement-public-preview-of-native-vector-support-in-azure-sql-database/) to store documents and perform vector search queries using Cosine (cosine distance), L2 (Euclidean distance), and IP (inner product) to locate documents close to the query vectors

## Setup
  
Install the `langchain-sqlserver` python package.

The code lives in an integration package called:[langchain-sqlserver](https:\github.com\langchain-ai\langchain-azure\tree\main\libs\sqlserver).
"""
logger.info("# SQLServer")

# !pip install langchain-sqlserver==0.1.1

"""
## Credentials

There are no credentials needed to run this notebook, just make sure you downloaded the `langchain-sqlserver` package
If you want to get best in-class automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("## Credentials")


"""
## Initialization
"""
logger.info("## Initialization")


"""
Find your Azure SQL DB connection string in the Azure portal under your database settings

For more info: [Connect to Azure SQL DB - Python](https:\learn.microsoft.com\en-us\azure\azure-sql\database\connect-query-python?view=azuresql)
"""
logger.info(
    "Find your Azure SQL DB connection string in the Azure portal under your database settings")


_CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    "Server=<YOUR_DBSERVER>.database.windows.net,1433;"
    "Database=test;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=60;"
    "LongAsMax=yes;"
)

"""
In this example we use Azure Ollama to generate embeddings , however you can use different embeddings provided in LangChain.

You can deploy a version of Azure Ollama instance on Azure Portal following this [guide](https:\learn.microsoft.com\en-us\azure\ai-services\ollama\how-to\create-resource?pivots=web-portal). Once you have your instance running, make sure you have the name of your instance and key. You can find the key in the Azure Portal, under the "Keys and Endpoint" section of your instance.
"""
logger.info("In this example we use Azure Ollama to generate embeddings , however you can use different embeddings provided in LangChain.")

# !pip install langchain-ollama


azure_endpoint = "https://<YOUR_ENDPOINT>.ollama.azure.com/"
azure_deployment_name_embedding = "nomic-embed-text"
azure_deployment_name_chatcompletion = "chatcompletion"
azure_api_version = "2023-05-15"
azure_


llm = AzureChatOllama(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment_name_chatcompletion,
    ollama_api_version=azure_api_version,
    ollama_api_key=azure_api_key,
)

embeddings = AzureOllamaEmbeddings(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment_name_embedding,
    ollama_api_version=azure_api_version,
    ollama_api_key=azure_api_key,
)

"""
## Manage vector store
"""
logger.info("## Manage vector store")


vector_store = SQLServer_VectorStore(
    connection_string=_CONNECTION_STRING,
    # optional, if not provided, defaults to COSINE
    distance_strategy=DistanceStrategy.COSINE,
    # you can use different embeddings provided in LangChain
    embedding_function=embeddings,
    embedding_length=1536,
    table_name="langchain_test_table",  # using table with a custom name
)

"""
### Add items to vector store
"""
logger.info("### Add items to vector store")

query = [
    "I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.",
    "The candy is just red , No flavor . Just  plan and chewy .  I would never buy them again",
    "Arrived in 6 days and were so stale i could not eat any of the 6 bags!!",
    "Got these on sale for roughly 25 cents per cup, which is half the price of my local grocery stores, plus they rarely stock the spicy flavors. These things are a GREAT snack for my office where time is constantly crunched and sometimes you can't escape for a real meal. This is one of my favorite flavors of Instant Lunch and will be back to buy every time it goes on sale.",
    "If you are looking for a less messy version of licorice for the children, then be sure to try these!  They're soft, easy to chew, and they don't get your hands all sticky and gross in the car, in the summer, at the beach, etc. We love all the flavos and sometimes mix these in with the chocolate to have a very nice snack! Great item, great price too, highly recommend!",
    "We had trouble finding this locally - delivery was fast, no more hunting up and down the flour aisle at our local grocery stores.",
    "Too much of a good thing? We worked this kibble in over time, slowly shifting the percentage of Felidae to national junk-food brand until the bowl was all natural. By this time, the cats couldn't keep it in or down. What a mess. We've moved on.",
    "Hey, the description says 360 grams - that is roughly 13 ounces at under $4.00 per can. No way - that is the approximate price for a 100 gram can.",
    "The taste of these white cheddar flat breads is like a regular cracker - which is not bad, except that I bought them because I wanted a cheese taste.<br /><br />What was a HUGE disappointment? How misleading the packaging of the box is. The photo on the box (I bought these in store) makes it look like it is full of long flatbreads (expanding the length and width of the box). Wrong! The plastic tray that holds the crackers is about 2"
    " smaller all around - leaving you with about 15 or so small flatbreads.<br /><br />What is also bad about this is that the company states they use biodegradable and eco-friendly packaging. FAIL! They used a HUGE box for a ridiculously small amount of crackers. Not ecofriendly at all.<br /><br />Would I buy these again? No - I feel ripped off. The other crackers (like Sesame Tarragon) give you a little<br />more bang for your buck and have more flavor.",
    "I have used this product in smoothies for my son and he loves it. Additionally, I use this oil in the shower as a skin conditioner and it has made my skin look great. Some of the stretch marks on my belly has disappeared quickly. Highly recommend!!!",
    "Been taking Coconut Oil for YEARS.  This is the best on the retail market.  I wish it was in glass, but this is the one.",
]

query_metadata = [
    {"id": 1, "summary": "Good Quality Dog Food"},
    {"id": 8, "summary": "Nasty No flavor"},
    {"id": 4, "summary": "stale product"},
    {"id": 11, "summary": "Great value and convenient ramen"},
    {"id": 5, "summary": "Great for the kids!"},
    {"id": 2, "summary": "yum falafel"},
    {"id": 9, "summary": "Nearly killed the cats"},
    {"id": 6, "summary": "Price cannot be correct"},
    {"id": 3, "summary": "Taste is neutral, quantity is DECEITFUL!"},
    {"id": 7, "summary": "This stuff is great"},
    {"id": 10, "summary": "The reviews don't lie"},
]

vector_store.add_texts(texts=query, metadatas=query_metadata)

"""
## Query vector store
Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

simsearch_result = vector_store.similarity_search("Good reviews", k=3)
logger.debug(simsearch_result)

"""
### Filtering Support:

The vectorstore supports a set of filters that can be applied against the metadata fields of the documents.This feature enables developers and data analysts to refine their queries, ensuring that the search results are accurately aligned with their needs. By applying filters based on specific metadata attributes, users can limit the scope of their searches, concentrating only on the most relevant data subsets.
"""
logger.info("### Filtering Support:")

hybrid_simsearch_result = vector_store.similarity_search(
    "Good reviews", k=3, filter={"id": {"$ne": 1}}
)
logger.debug(hybrid_simsearch_result)

"""
### Similarity Search with Score:
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("### Similarity Search with Score:")

simsearch_with_score_result = vector_store.similarity_search_with_score(
    "Not a very good product", k=12
)
logger.debug(simsearch_with_score_result)

"""
For a full list of the different searches you can execute on a Azure SQL vector store, please refer to the [API reference](https://python.langchain.com/api_reference/sqlserver/index.html).

### Similarity Search when you already have embeddings you want to search on
"""
logger.info(
    "### Similarity Search when you already have embeddings you want to search on")

simsearch_by_vector = vector_store.similarity_search_by_vector(
    [-0.0033353185281157494, -0.017689190804958344, -0.01590404286980629, ...]
)
logger.debug(simsearch_by_vector)

simsearch_by_vector_with_score = vector_store.similarity_search_by_vector_with_score(
    [-0.0033353185281157494, -0.017689190804958344, -0.01590404286980629, ...]
)
logger.debug(simsearch_by_vector_with_score)

"""
## Delete items from vector store

### Delete Row by ID
"""
logger.info("## Delete items from vector store")

vector_store.delete(["3", "7"])

"""
### Drop Vector Store
"""
logger.info("### Drop Vector Store")

vector_store.drop()

"""
## Load a Document from Azure Blob Storage

Below is example of loading a file from Azure Blob Storage container into the SQL Vector store after splitting the document into chunks.
[Azure Blog Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction) is Microsoft's object storage solution for the cloud. Blob Storage is optimized for storing massive amounts of unstructured data.
"""
logger.info("## Load a Document from Azure Blob Storage")

pip install azure-storage-blob


conn_str = "DefaultEndpointsProtocol=https;AccountName=<YourBlobName>;AccountKey=<YourAccountKey>==;EndpointSuffix=core.windows.net"
container_name = "<YourContainerName"
blob_name = "01 Harry Potter and the Sorcerers Stone.txt"

loader = AzureBlobStorageFileLoader(
    conn_str=conn_str, container=container_name, blob_name=blob_name
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

logger.debug(f"Number of split documents: {len(split_documents)}")

"""
API Reference:[AzureBlobStorageContainerLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.azure_blob_storage_container.AzureBlobStorageContainerLoader.html)
"""
logger.info("API Reference:[AzureBlobStorageContainerLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.azure_blob_storage_container.AzureBlobStorageContainerLoader.html)")

vector_store = SQLServer_VectorStore(
    connection_string=_CONNECTION_STRING,
    distance_strategy=DistanceStrategy.COSINE,
    embedding_function=embeddings,
    embedding_length=1536,
    table_name="harrypotter",
)  # Replace with your actual vector store initialization

for i, doc in enumerate(split_documents):
    vector_store.add_documents(documents=[doc], ids=[f"doc_{i}"])

logger.debug("Documents added to the vector store successfully!")

"""
## Query directly
"""
logger.info("## Query directly")


query = "Why did the Dursleys not want Harry in their house?"
docs_with_score: List[Tuple[Document, float]] = (
    vector_store.similarity_search_with_score(query)
)

for doc, score in docs_with_score:
    logger.debug("-" * 60)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 60)

"""
## Usage for retrieval-augmented generation

#### Use Case 1: Q&A System based on the Story Book

The Q&A function allows users to ask specific questions about the story, characters, and events, and get concise, context-rich answers. This not only enhances their understanding of the books but also makes them feel like they're part of the magical universe.

## Query by turning into retriever

The LangChain Vector store simplifies building sophisticated Q&A systems by enabling efficient similarity searches to find the top 10 relevant documents based on the user's query. The **retriever** is created from the **vector\_store,** and the question-answer chain is built using the **create\_stuff\_documents\_chain** function. A prompt template is crafted using the **ChatPromptTemplate** class, ensuring structured and context-rich responses. Often in Q&A applications it's important to show users the sources that were used to generate the answer. LangChain's built-in **create\_retrieval\_chain** will propagate retrieved source documents to the output under the "context" key:

Read more about Langchain RAG tutorials & the terminologies mentioned above [here](/docs/tutorials/rag)
"""
logger.info("## Usage for retrieval-augmented generation")


def get_answer_and_sources(user_query: str):
    docs_with_score: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_score(
            user_query,
            k=10,
        )
    )

    context = "\n".join([doc.page_content for doc, score in docs_with_score])

    system_prompt = (
        "You are an assistant for question-answering tasks based on the story in the book. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know, but also suggest that the user can use the fan fiction function to generate fun stories. "
        "Use 5 sentences maximum and keep the answer concise by also providing some background context of 1-2 sentences."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = vector_store.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    input_data = {"input": user_query}

    response = rag_chain.invoke(input_data)

    logger.debug("Answer:", response["answer"])

    data = {
        "Doc ID": [
            doc.metadata.get("source", "N/A").split("/")[-1]
            for doc in response["context"]
        ],
        "Content": [
            doc.page_content[:50] + "..."
            if len(doc.page_content) > 100
            else doc.page_content
            for doc in response["context"]
        ],
    }

    df = pd.DataFrame(data)

    logger.debug("\nSources:")
    logger.debug(df.to_markdown(index=False))


user_query = "How did Harry feel when he first learnt that he was a Wizard?"

get_answer_and_sources(user_query)

user_query = "Did Harry have a pet? What was it"

get_answer_and_sources(user_query)

"""
## API reference 

For detailed documentation of SQLServer Vectorstore features and configurations head to the API reference: [https://python.langchain.com/api\_reference/sqlserver/index.html](https:\python.langchain.com\api_reference\sqlserver\index.html)

## Related
- Vector store [conceptual guide](https://python.langchain.com/docs/concepts/vectorstores/)
- Vector store [how-to guides](https://python.langchain.com/docs/how_to/#vector-stores)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
