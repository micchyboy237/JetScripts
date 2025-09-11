from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hippo import Hippo
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
# Hippo

>[Transwarp Hippo](https://www.transwarp.cn/en/subproduct/hippo) is an enterprise-level cloud-native distributed vector database that supports storage, retrieval, and management of massive vector-based datasets. It efficiently solves problems such as vector similarity search and high-density vector clustering. `Hippo` features high availability, high performance, and easy scalability. It has many functions, such as multiple vector search indexes, data partitioning and sharding, data persistence, incremental data ingestion, vector scalar field filtering, and mixed queries. It can effectively meet the high real-time search demands of enterprises for massive vector data

## Getting Started

The only prerequisite here is an API key from the Ollama website. Make sure you have already started a Hippo instance.

## Installing Dependencies

Initially, we require the installation of certain dependencies, such as Ollama, Langchain, and Hippo-API. Please note, that you should install the appropriate versions tailored to your environment.
"""
logger.info("# Hippo")

# %pip install --upgrade --quiet  langchain langchain_community tiktoken langchain-ollama
# %pip install --upgrade --quiet  hippo-api==1.1.0.rc3

"""
Note: Python version needs to be >=3.8.

## Best Practices
### Importing Dependency Packages
"""
logger.info("## Best Practices")


"""
### Loading Knowledge Documents
"""
logger.info("### Loading Knowledge Documents")

# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI KEY"
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

"""
### Segmenting the Knowledge Document

Here, we use Langchain's CharacterTextSplitter for segmentation. The delimiter is a period. After segmentation, the text segment does not exceed 1000 characters, and the number of repeated characters is 0.
"""
logger.info("### Segmenting the Knowledge Document")

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

"""
### Declaring the Embedding Model
Below, we create the Ollama or Azure embedding model using the OllamaEmbeddings method from Langchain.
"""
logger.info("### Declaring the Embedding Model")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
### Declaring Hippo Client
"""
logger.info("### Declaring Hippo Client")

HIPPO_CONNECTION = {"host": "IP", "port": "PORT"}

"""
### Storing the Document
"""
logger.info("### Storing the Document")

logger.debug("input...")
vector_store = Hippo.from_documents(
    docs,
    embedding=embeddings,
    table_name="langchain_test",
    connection_args=HIPPO_CONNECTION,
)
logger.debug("success")

"""
### Conducting Knowledge-based Question and Answer
#### Creating a Large Language Question-Answering Model
Below, we create the Ollama or Azure large language question-answering model respectively using the AzureChatOllama and ChatOllama methods from Langchain.
"""
logger.info("### Conducting Knowledge-based Question and Answer")

llm = ChatOllama(model="llama3.2")

"""
### Acquiring Related Knowledge Based on the Question：
"""
logger.info("### Acquiring Related Knowledge Based on the Question：")

query = "Please introduce COVID-19"


res = vector_store.similarity_search(query, 2)
content_list = [item.page_content for item in res]
text = "".join(content_list)

"""
### Constructing a Prompt Template
"""
logger.info("### Constructing a Prompt Template")

prompt = f"""
Please use the content of the following [Article] to answer my question. If you don't know, please say you don't know, and the answer should be concise."
[Article]:{text}
Please answer this question in conjunction with the above article:{query}
"""

"""
### Waiting for the Large Language Model to Generate an Answer
"""
logger.info("### Waiting for the Large Language Model to Generate an Answer")

response_with_hippo = llm.predict(prompt)
logger.debug(f"response_with_hippo:{response_with_hippo}")
response = llm.predict(query)
logger.debug("==========================================")
logger.debug(f"response_without_hippo:{response}")

logger.info("\n\n[DONE]", bright=True)
