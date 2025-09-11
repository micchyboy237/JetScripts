from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import (
AlibabaCloudOpenSearch,
AlibabaCloudOpenSearchSettings,
)
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
# Alibaba Cloud OpenSearch

>[Alibaba Cloud Opensearch](https://www.alibabacloud.com/product/opensearch) is a one-stop platform to develop intelligent search services. `OpenSearch` was built on the large-scale distributed search engine developed by `Alibaba`. `OpenSearch` serves more than 500 business cases in Alibaba Group and thousands of Alibaba Cloud customers. `OpenSearch` helps develop search services in different search scenarios, including e-commerce, O2O, multimedia, the content industry, communities and forums, and big data query in enterprises.

>`OpenSearch` helps you develop high-quality, maintenance-free, and high-performance intelligent search services to provide your users with high search efficiency and accuracy.

>`OpenSearch` provides the vector search feature. In specific scenarios, especially test question search and image search scenarios, you can use the vector search feature together with the multimodal search feature to improve the accuracy of search results.

This notebook shows how to use functionality related to the `Alibaba Cloud OpenSearch Vector Search Edition`.

## Setting up


### Purchase an instance and configure it

Purchase OpenSearch Vector Search Edition from [Alibaba Cloud](https://opensearch.console.aliyun.com) and configure the instance according to the help [documentation](https://help.aliyun.com/document_detail/463198.html?spm=a2c4g.465092.0.0.2cd15002hdwavO).

To run, you should have an [OpenSearch Vector Search Edition](https://opensearch.console.aliyun.com) instance up and running.

  
### Alibaba Cloud OpenSearch Vector Store class
                                                                                                                `AlibabaCloudOpenSearch` class supports functions:
- `add_texts`
- `add_documents`
- `from_texts`
- `from_documents`
- `similarity_search`
- `asimilarity_search`
- `similarity_search_by_vector`
- `asimilarity_search_by_vector`
- `similarity_search_with_relevance_scores`
- `delete_doc_by_texts`


Read the [help document](https://www.alibabacloud.com/help/en/opensearch/latest/vector-search) to quickly familiarize and configure OpenSearch Vector Search Edition instance.

If you encounter any problems during use, please feel free to contact xingshaomin.xsm@alibaba-inc.com, and we will do our best to provide you with assistance and support.

After the instance is up and running, follow these steps to split documents, get embeddings, connect to the alibaba cloud opensearch instance, index documents, and perform vector retrieval.

We need to install the following Python packages first.
"""
logger.info("# Alibaba Cloud OpenSearch")

# %pip install --upgrade --quiet  langchain-community alibabacloud_ha3engine_vector

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info("We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
## Example
"""
logger.info("## Example")


"""
Split documents and get embeddings.
"""
logger.info("Split documents and get embeddings.")


loader = TextLoader("../../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
Create opensearch settings.
"""
logger.info("Create opensearch settings.")

settings = AlibabaCloudOpenSearchSettings(
    endpoint=" The endpoint of opensearch instance, You can find it from the console of Alibaba Cloud OpenSearch.",
    instance_id="The identify of opensearch instance, You can find it from the console of Alibaba Cloud OpenSearch.",
    protocol="Communication Protocol between SDK and Server, default is http.",
    username="The username specified when purchasing the instance.",
    password="The password specified when purchasing the instance.",
    namespace="The instance data will be partitioned based on the namespace field. If the namespace is enabled, you need to specify the namespace field name during initialization. Otherwise, the queries cannot be executed correctly.",
    tablename="The table name specified during instance configuration.",
    embedding_field_separator="Delimiter specified for writing vector field data, default is comma.",
    output_fields="Specify the field list returned when invoking OpenSearch, by default it is the value list of the field mapping field.",
    field_name_mapping={
        "id": "id",  # The id field name mapping of index document.
        "document": "document",  # The text field name mapping of index document.
        "embedding": "embedding",  # The embedding field name mapping of index document.
        "name_of_the_metadata_specified_during_search": "opensearch_metadata_field_name,=",
    },
)

"""
Create an opensearch access instance by settings.
"""
logger.info("Create an opensearch access instance by settings.")

opensearch = AlibabaCloudOpenSearch.from_texts(
    texts=docs, embedding=embeddings, config=settings
)

"""
or
"""
logger.info("or")

opensearch = AlibabaCloudOpenSearch(embedding=embeddings, config=settings)

"""
Add texts and build index.
"""
logger.info("Add texts and build index.")

metadatas = [
    {"string_field": "value1", "int_field": 1, "float_field": 1.0, "double_field": 2.0},
    {"string_field": "value2", "int_field": 2, "float_field": 3.0, "double_field": 4.0},
    {"string_field": "value3", "int_field": 3, "float_field": 5.0, "double_field": 6.0},
]
opensearch.add_texts(texts=docs, ids=[], metadatas=metadatas)

"""
Query and retrieve data.
"""
logger.info("Query and retrieve data.")

query = "What did the president say about Ketanji Brown Jackson"
docs = opensearch.similarity_search(query)
logger.debug(docs[0].page_content)

"""
Query and retrieve data with metadata.
"""
logger.info("Query and retrieve data with metadata.")

query = "What did the president say about Ketanji Brown Jackson"
metadata = {
    "string_field": "value1",
    "int_field": 1,
    "float_field": 1.0,
    "double_field": 2.0,
}
docs = opensearch.similarity_search(query, filter=metadata)
logger.debug(docs[0].page_content)

"""
If you encounter any problems during use, please feel free to contact xingshaomin.xsm@alibaba-inc.com, and we will do our best to provide you with assistance and support.
"""
logger.info("If you encounter any problems during use, please feel free to contact xingshaomin.xsm@alibaba-inc.com, and we will do our best to provide you with assistance and support.")

logger.info("\n\n[DONE]", bright=True)