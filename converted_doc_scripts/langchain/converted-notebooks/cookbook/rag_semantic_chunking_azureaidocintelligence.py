from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import AzureChatOllama, AzureOllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from operator import itemgetter
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
# Retrieval Augmented Generation (RAG)

This notebook demonstrates an example of using [LangChain](https://www.langchain.com/) to delvelop a Retrieval Augmented Generation (RAG) pattern. It uses Azure AI Document Intelligence as document loader, which can extracts tables, paragraphs, and layout information from pdf, image, office and html files. The output markdown can be used in LangChain's markdown header splitter, which enables semantic chunking of the documents. Then the chunked documents are indexed into Azure AI Search vectore store. Given a user query, it will use Azure AI Search to get the relevant chunks, then feed the context into the prompt with the query to generate the answer.

![semantic-chunking-rag.png](attachment:semantic-chunking-rag.png)

## Prerequisites
- An Azure AI Document Intelligence resource in one of the 3 preview regions: **East US**, **West US2**, **West Europe** - follow [this document](https://learn.microsoft.com/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0) to create one if you don't have.
- An Azure AI Search resource - follow [this document](https://learn.microsoft.com/azure/search/search-create-service-portal) to create one if you don't have.
- An Azure Ollama resource and deployments for embeddings model and chat model - follow [this document](https://learn.microsoft.com/azure/ai-services/ollama/how-to/create-resource?pivots=web-portal) to create one if you don't have.

We’ll use an Azure Ollama chat model and embeddings and Azure AI Search in this walkthrough, but everything shown here works with any ChatModel or LLM, Embeddings, and VectorStore or Retriever.

## Setup
"""
logger.info("# Retrieval Augmented Generation (RAG)")

# ! pip install python-dotenv langchain langchain-community langchain-ollama langchainhub ollama tiktoken azure-ai-documentintelligence azure-identity azure-search-documents==11.4.0b8

"""
This code loads environment variables using the `dotenv` library and sets the necessary environment variables for Azure services.
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""


load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")


"""
## Load a document and split it into semantic chunks
"""
logger.info("## Load a document and split it into semantic chunks")

loader = AzureAIDocumentIntelligenceLoader(
    file_path="<path to your file>",
    api_key=doc_intelligence_key,
    api_endpoint=doc_intelligence_endpoint,
    api_model="prebuilt-layout",
)
docs = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

docs_string = docs[0].page_content
splits = text_splitter.split_text(docs_string)

logger.debug("Length of splits: " + str(len(splits)))

"""
## Embed and index the chunks
"""
logger.info("## Embed and index the chunks")

aoai_embeddings = AzureOllamaEmbeddings(
    azure_deployment="<Azure Ollama embeddings model>",
    ollama_api_version="<Azure Ollama API version>",  # e.g., "2023-07-01-preview"
)

vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")

index_name: str = "<your index name>"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=aoai_embeddings.embed_query,
)

vector_store.add_documents(documents=splits)

"""
## Retrive relevant chunks based on a question
"""
logger.info("## Retrive relevant chunks based on a question")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retrieved_docs = retriever.invoke("<your question>")

logger.debug(retrieved_docs[0].page_content)

prompt = hub.pull("rlm/rag-prompt")
llm = AzureChatOllama(
    ollama_api_version="<Azure Ollama API version>",  # e.g., "2023-07-01-preview"
    azure_deployment="<your chat model deployment name>",
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

"""
## Document Q&A
"""
logger.info("## Document Q&A")

rag_chain.invoke("<your question>")

"""
## Doucment Q&A with references
"""
logger.info("## Doucment Q&A with references")



rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

rag_chain_with_source.invoke("<your question>")

logger.info("\n\n[DONE]", bright=True)