from dotenv import load_dotenv
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PagedCSVReader
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_csv_rag_with_llamaindex.ipynb)

# Simple RAG (Retrieval-Augmented Generation) System for CSV Files

## Overview

This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying CSV documents. The system encodes the document content into a vector store, which can then be queried to retrieve relevant information.

# CSV File Structure and Use Case
The CSV file contains dummy customer data, comprising various attributes like first name, last name, company, etc. This dataset will be utilized for a RAG use case, facilitating the creation of a customer information Q&A system.

## Key Components

1. Loading and spliting csv files.
2. Vector store creation using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) and Ollama embeddings
3. Query engine setup for querying the processed documents
4. Creating a question and answer over the csv data.

## Method Details

### Document Preprocessing

1. The csv is loaded using LlamaIndex's [PagedCSVReader](https://docs.llamaindex.ai/en/stable/api_reference/readers/file/#llama_index.readers.file.PagedCSVReader)
2. This reader converts each row into a LlamaIndex Document along with the respective column names of the table. No further splitting applied.


### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the text chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Query Engine Setup

1. A query engine is configured to fetch the most relevant chunks for a given query then answer the question.

## Benefits of this Approach

1. Scalability: Can handle large documents by processing them in chunks.
2. Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.
3. Efficiency: Utilizes FAISS for fast similarity search in high-dimensional spaces.
4. Integration with Advanced NLP: Uses Ollama embeddings for state-of-the-art text representation.

## Conclusion

This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. This approach is particularly useful for applications requiring quick access to specific information within a CSV file.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Simple RAG (Retrieval-Augmented Generation) System for CSV Files")

# !pip install faiss-cpu llama-index pandas python-dotenv



load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


EMBED_DIMENSION=512
Settings.llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

"""
### CSV File Structure and Use Case
The CSV file contains dummy customer data, comprising various attributes like first name, last name, company, etc. This dataset will be utilized for a RAG use case, facilitating the creation of a customer information Q&A system.
"""
logger.info("### CSV File Structure and Use Case")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/customers-100.csv https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/customers-100.csv

file_path = ('data/customers-100.csv') # insert the path of the csv file
data = pd.read_csv(file_path)

data.head()

"""
### Vector Store
"""
logger.info("### Vector Store")

fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)

"""
### Load and Process CSV Data as Document
"""
logger.info("### Load and Process CSV Data as Document")

csv_reader = PagedCSVReader()

reader = SimpleDirectoryReader(
    input_files=[file_path],
    file_extractor= {".csv": csv_reader}
    )

docs = reader.load_data()

logger.debug(docs[0].text)

"""
### Ingestion Pipeline
"""
logger.info("### Ingestion Pipeline")

pipeline = IngestionPipeline(
    vector_store=vector_store,
    documents=docs
)

nodes = pipeline.run()

"""
### Create Query Engine
"""
logger.info("### Create Query Engine")

vector_store_index = VectorStoreIndex(nodes)
query_engine = vector_store_index.as_query_engine(similarity_top_k=2)

"""
### Query the rag bot with a question based on the CSV data
"""
logger.info("### Query the rag bot with a question based on the CSV data")

response = query_engine.query("which company does sheryl Baxter work for?")
response.response

logger.info("\n\n[DONE]", bright=True)