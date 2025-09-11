from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain.tools import GooglePlacesTool # Or langchain_community.tools
from langchain.tools import YouTubeSearchTool # Or langchain_community.tools
from langchain.vectorstores import BigQueryVectorSearch # Or langchain_community.vectorstores
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.tools import GoogleSearchRun, GoogleSearchResults
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.vectorstores import ScaNN
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.messages import HumanMessage
from langchain_google_alloydb_pg import AlloyDBChatMessageHistory # AlloyDBEngine also available
from langchain_google_alloydb_pg import AlloyDBLoader # AlloyDBEngine also available
from langchain_google_alloydb_pg import AlloyDBVectorStore # AlloyDBEngine also available
from langchain_google_bigtable import BigtableChatMessageHistory
from langchain_google_bigtable import BigtableLoader
from langchain_google_cloud_sql_mssql import MSSQLChatMessageHistory # MSSQLEngine also available
from langchain_google_cloud_sql_mssql import MSSQLLoader # MSSQLEngine also available
from langchain_google_cloud_sql_mysql import MySQLChatMessageHistory # MySQLEngine also available
from langchain_google_cloud_sql_mysql import MySQLLoader # MySQLEngine also available
from langchain_google_cloud_sql_mysql import MySQLVectorStore # MySQLEngine also available
from langchain_google_cloud_sql_pg import PostgresChatMessageHistory # PostgresEngine also available
from langchain_google_cloud_sql_pg import PostgresLoader # PostgresEngine also available
from langchain_google_cloud_sql_pg import PostgresVectorStore # PostgresEngine also available
from langchain_google_community import BigQueryLoader
from langchain_google_community import DocAIParser
from langchain_google_community import GCSDirectoryLoader
from langchain_google_community import GCSFileLoader
from langchain_google_community import GMailLoader
from langchain_google_community import GmailToolkit
from langchain_google_community import GoogleDriveLoader
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_community import GoogleTranslateTransformer
from langchain_google_community import SpeechToTextLoader
from langchain_google_community import TextToSpeechTool
from langchain_google_community import VertexAIMultiTurnSearchRetriever
from langchain_google_community import VertexAISearchRetriever # Verify class name if needed
from langchain_google_community import VertexAISearchSummaryTool
from langchain_google_community.documentai_warehouse import DocumentAIWarehouseRetriever
from langchain_google_community.gmail.create_draft import GmailCreateDraft
from langchain_google_community.gmail.get_message import GmailGetMessage
from langchain_google_community.gmail.get_thread import GmailGetThread
from langchain_google_community.gmail.search import GmailSearch
from langchain_google_community.gmail.send_message import GmailSendMessage
from langchain_google_community.vision import CloudVisionLoader
from langchain_google_datastore import DatastoreChatMessageHistory
from langchain_google_datastore import DatastoreLoader
from langchain_google_el_carro import ElCarroChatMessageHistory
from langchain_google_el_carro import ElCarroLoader
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_firestore import FirestoreLoader
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_memorystore_redis import MemorystoreChatMessageHistory
from langchain_google_memorystore_redis import MemorystoreDocumentLoader
from langchain_google_memorystore_redis import RedisVectorStore
from langchain_google_spanner import SpannerChatMessageHistory
from langchain_google_spanner import SpannerLoader
from langchain_google_spanner import SpannerVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VectorSearchVectorStore
from langchain_google_vertexai import VectorSearchVectorStoreDatastore
from langchain_google_vertexai import VectorSearchVectorStoreGCS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAIModelGarden
from langchain_google_vertexai.callbacks import VertexAICallbackHandler
from langchain_google_vertexai.evaluators.evaluation import VertexPairWiseStringEvaluator
from langchain_google_vertexai.evaluators.evaluation import VertexStringEvaluator # Verify class name if needed
from langchain_google_vertexai.gemma import GemmaChatLocalHF
from langchain_google_vertexai.gemma import GemmaChatLocalKaggle
from langchain_google_vertexai.gemma import GemmaChatVertexAIModelGarden
from langchain_google_vertexai.gemma import GemmaLocalHF
from langchain_google_vertexai.gemma import GemmaLocalKaggle
from langchain_google_vertexai.gemma import GemmaVertexAIModelGarden
from langchain_google_vertexai.model_garden import ChatOllamaVertex
from langchain_google_vertexai.model_garden_maas.llama import VertexModelGardenLlama
from langchain_google_vertexai.model_garden_maas.mistral import VertexModelGardenMistral
from langchain_google_vertexai.vision_models import VertexAIImageCaptioning
from langchain_google_vertexai.vision_models import VertexAIImageCaptioningChat
from langchain_google_vertexai.vision_models import VertexAIImageEditorChat
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat
from langchain_google_vertexai.vision_models import VertexAIVisualQnAChat
from langchain_googledrive.retrievers import GoogleDriveRetriever
from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper
from toolbox_langchain import ToolboxClient
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
# Google

All functionality related to [Google Cloud](https://cloud.google.com/), [Google Gemini](https://ai.google.dev/gemini-api/docs) and other Google products.

1.  **Google Generative AI (Gemini API & AI Studio)**: Access Google Gemini models directly via the Gemini API. Use [Google AI Studio](https://aistudio.google.com/) for rapid prototyping and get started quickly with the `langchain-google-genai` package. This is often the best starting point for individual developers.
2.  **Google Cloud (Vertex AI & other services)**: Access Gemini models, Vertex AI Model Garden and a wide range of cloud services (databases, storage, document AI, etc.) via the [Google Cloud Platform](https://cloud.google.com/). Use the `langchain-google-vertexai` package for Vertex AI models and specific packages (e.g., `langchain-google-cloud-sql-pg`, `langchain-google-community`) for other cloud services. This is ideal for developers already using Google Cloud or needing enterprise features like MLOps, specific model tuning or enterprise support.

See Google's guide on [migrating from the Gemini API to Vertex AI](https://ai.google.dev/gemini-api/docs/migrate-to-cloud) for more details on the differences.

Integration packages for Gemini models and the Vertex AI platform are maintained in
the [langchain-google](https://github.com/langchain-ai/langchain-google) repository.
You can find a host of LangChain integrations with other Google APIs and services in the
[googleapis](https://github.com/googleapis?q=langchain-&type=all&language=&sort=)
Github organization and the `langchain-google-community` package.

## Google Generative AI (Gemini API & AI Studio)

Access Google Gemini models directly using the Gemini API, best suited for rapid development and experimentation. Gemini models are available in [Google AI Studio](https://aistudio.google.com/).
"""
logger.info("# Google")

pip install -U langchain-google-genai

"""
Start for free and get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
"""
logger.info("Start for free and get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).")

export GOOGLE_API_KEY="YOUR_API_KEY"

"""
### Chat Models

Use the `ChatGoogleGenerativeAI` class to interact with Gemini models. See
details in [this guide](/docs/integrations/chat/google_generative_ai).
"""
logger.info("### Chat Models")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("Sing a ballad of LangChain.")
logger.debug(result.content)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
result = llm.invoke([message])
logger.debug(result.content)

"""
The `image_url` can be a public URL, a GCS URI (`gs://...`), a local file path, a base64 encoded image string (`data:image/png;base64,...`), or a PIL Image object.


### Embedding Models

Generate text embeddings using models like `gemini-embedding-001` with the `GoogleGenerativeAIEmbeddings` class.

See a [usage example](/docs/integrations/text_embedding/google_generative_ai).
"""
logger.info("### Embedding Models")


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("What are embeddings?")
logger.debug(vector[:5])

"""
### LLMs

Access the same Gemini models using the ([legacy](/docs/concepts/text_llms)) LLM
interface with the `GoogleGenerativeAI` class.

See a [usage example](/docs/integrations/llms/google_ai).
"""
logger.info("### LLMs")


llm = GoogleGenerativeAI(model="gemini-2.5-flash")
result = llm.invoke("Sing a ballad of LangChain.")
logger.debug(result)

"""
## Google Cloud

Access Gemini models, Vertex AI Model Garden and other Google Cloud services via Vertex AI and specific cloud integrations.

Vertex AI models require the `langchain-google-vertexai` package. Other services might require additional packages like `langchain-google-community`, `langchain-google-cloud-sql-pg`, etc.
"""
logger.info("## Google Cloud")

pip install langchain-google-vertexai

"""
Google Cloud integrations typically use Application Default Credentials (ADC). Refer to the [Google Cloud authentication documentation](https://cloud.google.com/docs/authentication) for setup instructions (e.g., using `gcloud auth application-default login`).

### Chat Models

#### Vertex AI

Access chat models like Gemini via the Vertex AI platform.

See a [usage example](/docs/integrations/chat/google_vertex_ai_palm).
"""
logger.info("### Chat Models")


"""
#### Ollama on Vertex AI Model Garden

See a [usage example](/docs/integrations/llms/google_vertex_ai_palm).
"""
logger.info("#### Ollama on Vertex AI Model Garden")


"""
#### Llama on Vertex AI Model Garden
"""
logger.info("#### Llama on Vertex AI Model Garden")


"""
#### Mistral on Vertex AI Model Garden
"""
logger.info("#### Mistral on Vertex AI Model Garden")


"""
#### Gemma local from Hugging Face

>Local Gemma model loaded from HuggingFace. Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma local from Hugging Face")


"""
#### Gemma local from Kaggle

>Local Gemma model loaded from Kaggle. Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma local from Kaggle")


"""
#### Gemma on Vertex AI Model Garden

>Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma on Vertex AI Model Garden")


"""
#### Vertex AI image captioning

>Implementation of the Image Captioning model as a chat. Requires `langchain-google-vertexai`.
"""
logger.info("#### Vertex AI image captioning")


"""
#### Vertex AI image editor

>Given an image and a prompt, edit the image. Currently only supports mask-free editing. Requires `langchain-google-vertexai`.
"""
logger.info("#### Vertex AI image editor")


"""
#### Vertex AI image generator

>Generates an image from a prompt. Requires `langchain-google-vertexai`.
"""
logger.info("#### Vertex AI image generator")


"""
#### Vertex AI visual QnA

>Chat implementation of a visual QnA model. Requires `langchain-google-vertexai`.
"""
logger.info("#### Vertex AI visual QnA")


"""
### LLMs

You can also use the ([legacy](/docs/concepts/text_llms)) string-in, string-out LLM
interface.

#### Vertex AI Model Garden

Access Gemini, and hundreds of OSS models via Vertex AI Model Garden service. Requires `langchain-google-vertexai`.

See a [usage example](/docs/integrations/llms/google_vertex_ai_palm#vertex-model-garden).
"""
logger.info("### LLMs")


"""
#### Gemma local from Hugging Face

>Local Gemma model loaded from HuggingFace. Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma local from Hugging Face")


"""
#### Gemma local from Kaggle

>Local Gemma model loaded from Kaggle. Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma local from Kaggle")


"""
#### Gemma on Vertex AI Model Garden

>Requires `langchain-google-vertexai`.
"""
logger.info("#### Gemma on Vertex AI Model Garden")


"""
#### Vertex AI image captioning

>Implementation of the Image Captioning model as an LLM. Requires `langchain-google-vertexai`.
"""
logger.info("#### Vertex AI image captioning")


"""
### Embedding Models

#### Vertex AI

Generate embeddings using models deployed on Vertex AI. Requires `langchain-google-vertexai`.

See a [usage example](/docs/integrations/text_embedding/google_vertex_ai_palm).
"""
logger.info("### Embedding Models")


"""
### Document Loaders

Load documents from various Google Cloud sources.
#### AlloyDB for PostgreSQL

> [Google Cloud AlloyDB](https://cloud.google.com/alloydb) is a fully managed PostgreSQL-compatible database service.

Install the python package:
"""
logger.info("### Document Loaders")

pip install langchain-google-alloydb-pg

"""
See [usage example](/docs/integrations/document_loaders/google_alloydb).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_alloydb).")


"""
#### BigQuery

> [Google Cloud BigQuery](https://cloud.google.com/bigquery) is a serverless data warehouse.

Install with BigQuery dependencies:
"""
logger.info("#### BigQuery")

pip install langchain-google-community[bigquery]

"""
See a [usage example](/docs/integrations/document_loaders/google_bigquery).
"""
logger.info("See a [usage example](/docs/integrations/document_loaders/google_bigquery).")


"""
#### Bigtable

> [Google Cloud Bigtable](https://cloud.google.com/bigtable/docs) is a fully managed NoSQL Big Data database service.

Install the python package:
"""
logger.info("#### Bigtable")

pip install langchain-google-bigtable

"""
See [usage example](/docs/integrations/document_loaders/google_bigtable).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_bigtable).")


"""
#### Cloud SQL for MySQL

> [Google Cloud SQL for MySQL](https://cloud.google.com/sql) is a fully-managed MySQL database service.

Install the python package:
"""
logger.info("#### Cloud SQL for MySQL")

pip install langchain-google-cloud-sql-mysql

"""
See [usage example](/docs/integrations/document_loaders/google_cloud_sql_mysql).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_cloud_sql_mysql).")


"""
#### Cloud SQL for SQL Server

> [Google Cloud SQL for SQL Server](https://cloud.google.com/sql) is a fully-managed SQL Server database service.

Install the python package:
"""
logger.info("#### Cloud SQL for SQL Server")

pip install langchain-google-cloud-sql-mssql

"""
See [usage example](/docs/integrations/document_loaders/google_cloud_sql_mssql).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_cloud_sql_mssql).")


"""
#### Cloud SQL for PostgreSQL

> [Google Cloud SQL for PostgreSQL](https://cloud.google.com/sql) is a fully-managed PostgreSQL database service.

Install the python package:
"""
logger.info("#### Cloud SQL for PostgreSQL")

pip install langchain-google-cloud-sql-pg

"""
See [usage example](/docs/integrations/document_loaders/google_cloud_sql_pg).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_cloud_sql_pg).")


"""
#### Cloud Storage

>[Cloud Storage](https://en.wikipedia.org/wiki/Google_Cloud_Storage) is a managed service for storing unstructured data.

Install with GCS dependencies:
"""
logger.info("#### Cloud Storage")

pip install langchain-google-community[gcs]

"""
Load from a directory or a specific file:

See [directory usage example](/docs/integrations/document_loaders/google_cloud_storage_directory).
"""
logger.info("Load from a directory or a specific file:")


"""
See [file usage example](/docs/integrations/document_loaders/google_cloud_storage_file).
"""
logger.info("See [file usage example](/docs/integrations/document_loaders/google_cloud_storage_file).")


"""
#### Cloud Vision loader

Load data using Google Cloud Vision API.

Install with Vision dependencies:
"""
logger.info("#### Cloud Vision loader")

pip install langchain-google-community[vision]

"""

"""


"""
#### El Carro for Oracle Workloads

> Google [El Carro Oracle Operator](https://github.com/GoogleCloudPlatform/elcarro-oracle-operator) runs Oracle databases in Kubernetes.

Install the python package:
"""
logger.info("#### El Carro for Oracle Workloads")

pip install langchain-google-el-carro

"""
See [usage example](/docs/integrations/document_loaders/google_el_carro).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_el_carro).")


"""
#### Firestore (Native Mode)

> [Google Cloud Firestore](https://cloud.google.com/firestore/docs/) is a NoSQL document database.

Install the python package:
"""
logger.info("#### Firestore (Native Mode)")

pip install langchain-google-firestore

"""
See [usage example](/docs/integrations/document_loaders/google_firestore).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_firestore).")


"""
#### Firestore (Datastore Mode)

> [Google Cloud Firestore in Datastore mode](https://cloud.google.com/datastore/docs).

Install the python package:
"""
logger.info("#### Firestore (Datastore Mode)")

pip install langchain-google-datastore

"""
See [usage example](/docs/integrations/document_loaders/google_datastore).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_datastore).")


"""
#### Memorystore for Redis

> [Google Cloud Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis) is a fully managed Redis service.

Install the python package:
"""
logger.info("#### Memorystore for Redis")

pip install langchain-google-memorystore-redis

"""
See [usage example](/docs/integrations/document_loaders/google_memorystore_redis).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_memorystore_redis).")


"""
#### Spanner

> [Google Cloud Spanner](https://cloud.google.com/spanner/docs) is a fully managed, globally distributed relational database service.

Install the python package:
"""
logger.info("#### Spanner")

pip install langchain-google-spanner

"""
See [usage example](/docs/integrations/document_loaders/google_spanner).
"""
logger.info("See [usage example](/docs/integrations/document_loaders/google_spanner).")


"""
#### Speech-to-Text

> [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text) transcribes audio files.

Install with Speech-to-Text dependencies:
"""
logger.info("#### Speech-to-Text")

pip install langchain-google-community[speech]

"""
See [usage example and authorization instructions](/docs/integrations/document_loaders/google_speech_to_text).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/document_loaders/google_speech_to_text).")


"""
### Document Transformers

Transform documents using Google Cloud services.

#### Document AI

>[Google Cloud Document AI](https://cloud.google.com/document-ai/docs/overview) is a Google Cloud
> service that transforms unstructured data from documents into structured data, making it easier
> to understand, analyze, and consume.

We need to set up a [`GCS` bucket and create your own OCR processor](https://cloud.google.com/document-ai/docs/create-processor)
The `GCS_OUTPUT_PATH` should be a path to a folder on GCS (starting with `gs://`)
and a processor name should look like `projects/PROJECT_NUMBER/locations/LOCATION/processors/PROCESSOR_ID`.
We can get it either programmatically or copy from the `Prediction endpoint` section of the `Processor details`
tab in the Google Cloud Console.
"""
logger.info("### Document Transformers")

pip install langchain-google-community[docai]

"""
See a [usage example](/docs/integrations/document_transformers/google_docai).
"""
logger.info("See a [usage example](/docs/integrations/document_transformers/google_docai).")


"""
#### Google Translate

> [Google Translate](https://translate.google.com/) is a multilingual neural machine
> translation service developed by Google to translate text, documents and websites
> from one language into another.

The `GoogleTranslateTransformer` allows you to translate text and HTML with the [Google Cloud Translation API](https://cloud.google.com/translate).

First, we need to install the `langchain-google-community` with translate dependencies.
"""
logger.info("#### Google Translate")

pip install langchain-google-community[translate]

"""
See [usage example and authorization instructions](/docs/integrations/document_transformers/google_translate).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/document_transformers/google_translate).")


"""
### Vector Stores

Store and search vectors using Google Cloud databases and Vertex AI Vector Search.

#### AlloyDB for PostgreSQL

> [Google Cloud AlloyDB](https://cloud.google.com/alloydb) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability on Google Cloud. AlloyDB is 100% compatible with PostgreSQL.

Install the python package:
"""
logger.info("### Vector Stores")

pip install langchain-google-alloydb-pg

"""
See [usage example](/docs/integrations/vectorstores/google_alloydb).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_alloydb).")


"""
#### BigQuery Vector Search

> [Google Cloud BigQuery](https://cloud.google.com/bigquery),
> BigQuery is a serverless and cost-effective enterprise data warehouse in Google Cloud.
>
> [Google Cloud BigQuery Vector Search](https://cloud.google.com/bigquery/docs/vector-search-intro)
> BigQuery vector search lets you use GoogleSQL to do semantic search, using vector indexes for fast but approximate results, or using brute force for exact results.

> It can calculate Euclidean or Cosine distance. With LangChain, we default to use Euclidean distance.

We need to install several python packages.
"""
logger.info("#### BigQuery Vector Search")

pip install google-cloud-bigquery

"""
See [usage example](/docs/integrations/vectorstores/google_bigquery_vector_search).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_bigquery_vector_search).")


"""
#### Memorystore for Redis

> Vector store using [Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis).

Install the python package:
"""
logger.info("#### Memorystore for Redis")

pip install langchain-google-memorystore-redis

"""
See [usage example](/docs/integrations/vectorstores/google_memorystore_redis).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_memorystore_redis).")


"""
#### Spanner

> Vector store using [Cloud Spanner](https://cloud.google.com/spanner/docs).

Install the python package:
"""
logger.info("#### Spanner")

pip install langchain-google-spanner

"""
See [usage example](/docs/integrations/vectorstores/google_spanner).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_spanner).")


"""
#### Firestore (Native Mode)

> Vector store using [Firestore](https://cloud.google.com/firestore/docs/).

Install the python package:
"""
logger.info("#### Firestore (Native Mode)")

pip install langchain-google-firestore

"""
See [usage example](/docs/integrations/vectorstores/google_firestore).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_firestore).")


"""
#### Cloud SQL for MySQL

> Vector store using [Cloud SQL for MySQL](https://cloud.google.com/sql).

Install the python package:
"""
logger.info("#### Cloud SQL for MySQL")

pip install langchain-google-cloud-sql-mysql

"""
See [usage example](/docs/integrations/vectorstores/google_cloud_sql_mysql).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_cloud_sql_mysql).")


"""
#### Cloud SQL for PostgreSQL

> Vector store using [Cloud SQL for PostgreSQL](https://cloud.google.com/sql).

Install the python package:
"""
logger.info("#### Cloud SQL for PostgreSQL")

pip install langchain-google-cloud-sql-pg

"""
See [usage example](/docs/integrations/vectorstores/google_cloud_sql_pg).
"""
logger.info("See [usage example](/docs/integrations/vectorstores/google_cloud_sql_pg).")


"""
#### Vertex AI Vector Search

> [Google Cloud Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) from Google Cloud,
> formerly known as `Vertex AI Matching Engine`, provides the industry's leading high-scale
> low latency vector database. These vector databases are commonly
> referred to as vector similarity-matching or an approximate nearest neighbor (ANN) service.

Install the python package:
"""
logger.info("#### Vertex AI Vector Search")

pip install langchain-google-vertexai

"""
See a [usage example](/docs/integrations/vectorstores/google_vertex_ai_vector_search).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/google_vertex_ai_vector_search).")


"""
##### With DataStore Backend

> Vector search using Datastore for document storage.

See [usage example](/docs/integrations/vectorstores/google_vertex_ai_vector_search/#optional--you-can-also-create-vectore-and-store-chunks-in-a-datastore).
"""
logger.info("##### With DataStore Backend")


"""
##### With GCS Backend

> Alias for `VectorSearchVectorStore` storing documents/index in GCS.
"""
logger.info("##### With GCS Backend")


"""
### Retrievers

Retrieve information using Google Cloud services.

#### Vertex AI Search

> Build generative AI powered search engines using [Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/introduction).
> from Google Cloud allows developers to quickly build generative AI powered search engines for customers and employees.

See a [usage example](/docs/integrations/retrievers/google_vertex_ai_search).

Note: `GoogleVertexAISearchRetriever` is deprecated. Use the components below from `langchain-google-community`.

Install the `google-cloud-discoveryengine` package for underlying access.
"""
logger.info("### Retrievers")

pip install google-cloud-discoveryengine langchain-google-community

"""
##### VertexAIMultiTurnSearchRetriever
"""
logger.info("##### VertexAIMultiTurnSearchRetriever")


"""
##### VertexAISearchRetriever
"""
logger.info("##### VertexAISearchRetriever")


"""
##### VertexAISearchSummaryTool
"""
logger.info("##### VertexAISearchSummaryTool")


"""
#### Document AI Warehouse

> Search, store, and manage documents using [Document AI Warehouse](https://cloud.google.com/document-ai-warehouse).

Note: `GoogleDocumentAIWarehouseRetriever` (from `langchain`) is deprecated. Use `DocumentAIWarehouseRetriever` from `langchain-google-community`.

Requires installation of relevant Document AI packages (check specific docs).
"""
logger.info("#### Document AI Warehouse")

pip install langchain-google-community # Add specific docai dependencies if needed

"""

"""


"""
### Tools

Integrate agents with various Google services.

#### Text-to-Speech

>[Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech) is a Google Cloud service that enables developers to
> synthesize natural-sounding speech with 100+ voices, available in multiple languages and variants.
> It applies DeepMind's groundbreaking research in WaveNet and Google's powerful neural networks
> to deliver the highest fidelity possible.

Install required packages:
"""
logger.info("### Tools")

pip install google-cloud-text-to-speech langchain-google-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_cloud_texttospeech).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_cloud_texttospeech).")


"""
#### Google Drive

Tools for interacting with Google Drive.

Install required packages:
"""
logger.info("#### Google Drive")

pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain-googledrive

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_drive).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_drive).")


"""
#### Google Finance

Query financial data. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Finance")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_finance).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_finance).")


"""
#### Google Jobs

Query job listings. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Jobs")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_jobs).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_jobs).")


"""
#### Google Lens

Perform visual searches. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Lens")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_lens).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_lens).")


"""
#### Google Places

Search for places information. Requires `googlemaps` package and a Google Maps API key.
"""
logger.info("#### Google Places")

pip install googlemaps langchain # Requires base langchain

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_places).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_places).")


"""
#### Google Scholar

Search academic papers. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Scholar")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_scholar).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_scholar).")


"""
#### Google Search

Perform web searches using Google Custom Search Engine (CSE). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.

Install `langchain-google-community`:
"""
logger.info("#### Google Search")

pip install langchain-google-community

"""
Wrapper:
"""
logger.info("Wrapper:")


"""
Tools:
"""
logger.info("Tools:")


"""
Agent Loading:
"""
logger.info("Agent Loading:")

tools = load_tools(["google-search"])

"""
See [detailed notebook](/docs/integrations/tools/google_search).

#### Google Trends

Query Google Trends data. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Trends")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_trends).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_trends).")


"""
### Toolkits

Collections of tools for specific Google services.

#### GMail


> [Google Gmail](https://en.wikipedia.org/wiki/Gmail) is a free email service provided by Google.
This toolkit works with emails through the `Gmail API`.
"""
logger.info("### Toolkits")

pip install langchain-google-community[gmail]

"""
See [usage example and authorization instructions](/docs/integrations/tools/gmail).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/gmail).")



"""
### MCP Toolbox

[MCP Toolbox](https://github.com/googleapis/genai-toolbox) provides a simple and efficient way to connect to your databases, including those on Google Cloud like [Cloud SQL](https://cloud.google.com/sql/docs) and [AlloyDB](https://cloud.google.com/alloydb/docs/overview). With MCP Toolbox, you can seamlessly integrate your database with LangChain to build powerful, data-driven applications.

#### Installation

To get started, [install the Toolbox server and client](https://github.com/googleapis/genai-toolbox/releases/).


[Configure](https://googleapis.github.io/genai-toolbox/getting-started/configure/) a `tools.yaml` to define your tools, and then execute toolbox to start the server:
"""
logger.info("### MCP Toolbox")

toolbox --tools-file "tools.yaml"

"""
Then, install the Toolbox client:
"""
logger.info("Then, install the Toolbox client:")

pip install toolbox-langchain

"""
#### Getting Started

Here is a quick example of how to use MCP Toolbox to connect to your database:
"""
logger.info("#### Getting Started")


async with ToolboxClient("http://127.0.0.1:5000") as client:
    
        tools = client.load_toolset()
logger.success(format_json(result))

"""
See [usage example and setup instructions](/docs/integrations/tools/toolbox).

### Memory

Store conversation history using Google Cloud databases.

#### AlloyDB for PostgreSQL

> Chat memory using [AlloyDB](https://cloud.google.com/alloydb).

Install the python package:
"""
logger.info("### Memory")

pip install langchain-google-alloydb-pg

"""
See [usage example](/docs/integrations/memory/google_alloydb).
"""
logger.info("See [usage example](/docs/integrations/memory/google_alloydb).")


"""
#### Cloud SQL for PostgreSQL

> Chat memory using [Cloud SQL for PostgreSQL](https://cloud.google.com/sql).

Install the python package:
"""
logger.info("#### Cloud SQL for PostgreSQL")

pip install langchain-google-cloud-sql-pg

"""
See [usage example](/docs/integrations/memory/google_sql_pg).
"""
logger.info("See [usage example](/docs/integrations/memory/google_sql_pg).")


"""
#### Cloud SQL for MySQL

> Chat memory using [Cloud SQL for MySQL](https://cloud.google.com/sql).

Install the python package:
"""
logger.info("#### Cloud SQL for MySQL")

pip install langchain-google-cloud-sql-mysql

"""
See [usage example](/docs/integrations/memory/google_sql_mysql).
"""
logger.info("See [usage example](/docs/integrations/memory/google_sql_mysql).")


"""
#### Cloud SQL for SQL Server

> Chat memory using [Cloud SQL for SQL Server](https://cloud.google.com/sql).

Install the python package:
"""
logger.info("#### Cloud SQL for SQL Server")

pip install langchain-google-cloud-sql-mssql

"""
See [usage example](/docs/integrations/memory/google_sql_mssql).
"""
logger.info("See [usage example](/docs/integrations/memory/google_sql_mssql).")


"""
#### Spanner

> Chat memory using [Cloud Spanner](https://cloud.google.com/spanner/docs).

Install the python package:
"""
logger.info("#### Spanner")

pip install langchain-google-spanner

"""
See [usage example](/docs/integrations/memory/google_spanner).
"""
logger.info("See [usage example](/docs/integrations/memory/google_spanner).")


"""
#### Memorystore for Redis

> Chat memory using [Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis).

Install the python package:
"""
logger.info("#### Memorystore for Redis")

pip install langchain-google-memorystore-redis

"""
See [usage example](/docs/integrations/memory/google_memorystore_redis).
"""
logger.info("See [usage example](/docs/integrations/memory/google_memorystore_redis).")


"""
#### Bigtable

> Chat memory using [Cloud Bigtable](https://cloud.google.com/bigtable/docs).

Install the python package:
"""
logger.info("#### Bigtable")

pip install langchain-google-bigtable

"""
See [usage example](/docs/integrations/memory/google_bigtable).
"""
logger.info("See [usage example](/docs/integrations/memory/google_bigtable).")


"""
#### Firestore (Native Mode)

> Chat memory using [Firestore](https://cloud.google.com/firestore/docs/).

Install the python package:
"""
logger.info("#### Firestore (Native Mode)")

pip install langchain-google-firestore

"""
See [usage example](/docs/integrations/memory/google_firestore).
"""
logger.info("See [usage example](/docs/integrations/memory/google_firestore).")


"""
#### Firestore (Datastore Mode)

> Chat memory using [Firestore in Datastore mode](https://cloud.google.com/datastore/docs).

Install the python package:
"""
logger.info("#### Firestore (Datastore Mode)")

pip install langchain-google-datastore

"""
See [usage example](/docs/integrations/memory/google_firestore_datastore).
"""
logger.info("See [usage example](/docs/integrations/memory/google_firestore_datastore).")


"""
#### El Carro: The Oracle Operator for Kubernetes

> Chat memory using Oracle databases run via [El Carro](https://github.com/GoogleCloudPlatform/elcarro-oracle-operator).

Install the python package:
"""
logger.info("#### El Carro: The Oracle Operator for Kubernetes")

pip install langchain-google-el-carro

"""
See [usage example](/docs/integrations/memory/google_el_carro).
"""
logger.info("See [usage example](/docs/integrations/memory/google_el_carro).")


"""
### Callbacks

Track LLM/Chat model usage.

#### Vertex AI callback handler

>Callback Handler that tracks `VertexAI` usage info.

Requires `langchain-google-vertexai`.
"""
logger.info("### Callbacks")


"""
### Evaluators

Evaluate model outputs using Vertex AI.

Requires `langchain-google-vertexai`.

#### VertexPairWiseStringEvaluator

>Pair-wise evaluation using Vertex AI models.
"""
logger.info("### Evaluators")


"""
#### VertexStringEvaluator

>Evaluate a single prediction string using Vertex AI models.
"""
logger.info("#### VertexStringEvaluator")


"""
## Other Google Products

Integrations with various Google services beyond the core Cloud Platform.

### Document Loaders

#### Google Drive

>[Google Drive](https://en.wikipedia.org/wiki/Google_Drive) file storage. Currently supports Google Docs.

Install with Drive dependencies:
"""
logger.info("## Other Google Products")

pip install langchain-google-community[drive]

"""
See [usage example and authorization instructions](/docs/integrations/document_loaders/google_drive).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/document_loaders/google_drive).")


"""
### Vector Stores

#### ScaNN (Local Index)

>[Google ScaNN](https://github.com/google-research/google-research/tree/master/scann)
> (Scalable Nearest Neighbors) is a python package.
>
>`ScaNN` is a method for efficient vector similarity search at scale.

>`ScaNN` includes search space pruning and quantization for Maximum Inner
> Product Search and also supports other distance functions such as
> Euclidean distance. The implementation is optimized for x86 processors
> with AVX2 support. See its [Google Research github](https://github.com/google-research/google-research/tree/master/scann)
> for more details.

Install the `scann` package:
"""
logger.info("### Vector Stores")

pip install scann langchain-community # Requires langchain-community

"""
See a [usage example](/docs/integrations/vectorstores/scann).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/scann).")


"""
### Retrievers

#### Google Drive

Retrieve documents from Google Drive.

Install required packages:
"""
logger.info("### Retrievers")

pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain-googledrive

"""
See [usage example and authorization instructions](/docs/integrations/retrievers/google_drive).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/retrievers/google_drive).")


"""
### Tools

#### Google Drive

Tools for interacting with Google Drive.

Install required packages:
"""
logger.info("### Tools")

pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain-googledrive

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_drive).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_drive).")


"""
#### Google Finance

Query financial data. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Finance")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_finance).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_finance).")


"""
#### Google Jobs

Query job listings. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Jobs")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_jobs).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_jobs).")


"""
#### Google Lens

Perform visual searches. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Lens")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_lens).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_lens).")


"""
#### Google Places

Search for places information. Requires `googlemaps` package and a Google Maps API key.
"""
logger.info("#### Google Places")

pip install googlemaps langchain # Requires base langchain

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_places).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_places).")


"""
#### Google Scholar

Search academic papers. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Scholar")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_scholar).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_scholar).")


"""
#### Google Search

Perform web searches using Google Custom Search Engine (CSE). Requires `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.

Install `langchain-google-community`:
"""
logger.info("#### Google Search")

pip install langchain-google-community

"""
Wrapper:
"""
logger.info("Wrapper:")


"""
Tools:
"""
logger.info("Tools:")


"""
Agent Loading:
"""
logger.info("Agent Loading:")

tools = load_tools(["google-search"])

"""
See [detailed notebook](/docs/integrations/tools/google_search).

#### Google Trends

Query Google Trends data. Requires `google-search-results` package and SerpApi key.
"""
logger.info("#### Google Trends")

pip install google-search-results langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/tools/google_trends).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/google_trends).")


"""
### Toolkits

#### GMail


> [Google Gmail](https://en.wikipedia.org/wiki/Gmail) is a free email service provided by Google.
This toolkit works with emails through the `Gmail API`.
"""
logger.info("### Toolkits")

pip install langchain-google-community[gmail]

"""
See [usage example and authorization instructions](/docs/integrations/tools/gmail).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/tools/gmail).")



"""
### Chat Loaders

#### GMail

> Load chat history from Gmail threads.

Install with GMail dependencies:
"""
logger.info("### Chat Loaders")

pip install langchain-google-community[gmail]

"""
See [usage example and authorization instructions](/docs/integrations/chat_loaders/gmail).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/chat_loaders/gmail).")


"""
## 3rd Party Integrations

Access Google services via third-party APIs.

### SearchApi

>[SearchApi](https://www.searchapi.io/) provides API access to Google search, YouTube, etc. Requires `langchain-community`.

See [usage examples and authorization instructions](/docs/integrations/tools/searchapi).
"""
logger.info("## 3rd Party Integrations")


"""
### SerpApi

>[SerpApi](https://serpapi.com/) provides API access to Google search results. Requires `langchain-community`.

See a [usage example and authorization instructions](/docs/integrations/tools/serpapi).
"""
logger.info("### SerpApi")


"""
### Serper.dev

>[Google Serper](https://serper.dev/) provides API access to Google search results. Requires `langchain-community`.

See a [usage example and authorization instructions](/docs/integrations/tools/google_serper).
"""
logger.info("### Serper.dev")


"""
### YouTube

#### YouTube Search Tool

>Search YouTube videos without the official API. Requires `youtube_search` package.
"""
logger.info("### YouTube")

pip install youtube_search langchain # Requires base langchain

"""
See a [usage example](/docs/integrations/tools/youtube).
"""
logger.info("See a [usage example](/docs/integrations/tools/youtube).")


"""
#### YouTube Audio Loader

>Download audio from YouTube videos. Requires `yt_dlp`, `pydub`, `librosa`.
"""
logger.info("#### YouTube Audio Loader")

pip install yt_dlp pydub librosa langchain-community # Requires langchain-community

"""
See [usage example and authorization instructions](/docs/integrations/document_loaders/youtube_audio).
"""
logger.info("See [usage example and authorization instructions](/docs/integrations/document_loaders/youtube_audio).")


"""
#### YouTube Transcripts Loader

>Load video transcripts. Requires `youtube-transcript-api`.
"""
logger.info("#### YouTube Transcripts Loader")

pip install youtube-transcript-api langchain-community # Requires langchain-community

"""
See a [usage example](/docs/integrations/document_loaders/youtube_transcript).
"""
logger.info("See a [usage example](/docs/integrations/document_loaders/youtube_transcript).")


logger.info("\n\n[DONE]", bright=True)