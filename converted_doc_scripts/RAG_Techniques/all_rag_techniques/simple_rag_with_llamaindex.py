from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv
from evaluation.evalute_rag import create_deep_eval_test_cases
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import List
import faiss
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag_with_llamaindex.ipynb)

# Simple RAG (Retrieval-Augmented Generation) System

## Overview

This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying PDF document(s). The system uses a pipeline that encodes the documents and creates nodes. These nodes then can be used to build a vector index to retrieve relevant information.

## Key Components

1. PDF processing and text extraction
2. Text chunking for manageable processing
3. Ingestion pipeline creation using FAISS as vector store and Ollama embeddings
4. Retriever setup for querying the processed documents
5. Evaluation of the RAG system

## Method Details

### Document Preprocessing

1. The PDF is loaded using [SimpleDirectoryReader](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/).
2. The text is split into [nodes/chunks](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/) using [SentenceSplitter](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencesplitter) with specified chunk size and overlap.

### Text Cleaning

A custom transformation `TextCleaner` is applied to clean the texts. This likely addresses specific formatting issues in the PDF.

### Ingestion Pipeline Creation

1. Ollama embeddings are used to create vector representations of the text nodes.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Retriever Setup

1. A retriever is configured to fetch the top 2 most relevant chunks for a given query.


## Key Features

1. Modular Design: The ingestion process is encapsulated in a single function for easy reuse.
2. Configurable Chunking: Allows adjustment of chunk size and overlap.
3. Efficient Retrieval: Uses FAISS for fast similarity search.
4. Evaluation: Includes a function to evaluate the RAG system's performance.

## Usage Example

The code includes a test query: "What is the main cause of climate change?". This demonstrates how to use the retriever to fetch relevant context from the processed document.

## Evaluation

The system includes an `evaluate_rag` function to assess the performance of the retriever, though the specific metrics used are not detailed in the provided code.

## Benefits of this Approach

1. Scalability: Can handle large documents by processing them in chunks.
2. Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.
3. Efficiency: Utilizes FAISS for fast similarity search in high-dimensional spaces.
4. Integration with Advanced NLP: Uses Ollama embeddings for state-of-the-art text representation.

## Conclusion

This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. This approach is particularly useful for applications requiring quick access to specific information within large documents or document collections.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Simple RAG (Retrieval-Augmented Generation) System")

# !pip install faiss-cpu llama-index python-dotenv

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')



EMBED_DIMENSION = 512

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

"""
### Read Docs
"""
logger.info("### Read Docs")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/q_a.json https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/q_a.json

path = f"{GENERATED_DIR}/"
node_parser = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = node_parser.load_data()
logger.debug(documents[0])

"""
### Vector Store
"""
logger.info("### Vector Store")

faiss_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=faiss_index)

"""
### Text Cleaner Transformation
"""
logger.info("### Text Cleaner Transformation")

class TextCleaner(TransformComponent):
    """
    Transformation to be used within the ingestion pipeline.
    Cleans clutters from texts.
    """
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:

        for node in nodes:
            node.text = node.text.replace('\t', ' ') # Replace tabs with spaces
            node.text = node.text.replace(' \n', ' ') # Replace paragraph seperator with spacaes

        return nodes

"""
### Ingestion Pipeline
"""
logger.info("### Ingestion Pipeline")

text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
    ],
    vector_store=vector_store,
)

nodes = pipeline.run(documents=documents)

"""
### Create retriever
"""
logger.info("### Create retriever")

vector_store_index = VectorStoreIndex(nodes)
retriever = vector_store_index.as_retriever(similarity_top_k=2)

"""
### Test retriever
"""
logger.info("### Test retriever")

def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        logger.debug(f"Context {i+1}:")
        logger.debug(c.text)
        logger.debug("\n")

test_query = "What is the main cause of climate change?"
context = retriever.retrieve(test_query)
show_context(context)

"""
### Let's see how well does it perform:
"""
logger.info("### Let's see how well does it perform:")


LLM_MODEL = "gpt-4o"

correctness_metric = GEval(
    name="Correctness",
    model=LLM_MODEL,
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=LLM_MODEL,
    include_reason=False
)

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model=LLM_MODEL,
    include_reason=True
)

def evaluate_rag(query_engine, num_questions: int = 5) -> None:
    """
    Evaluate the RAG system using predefined metrics.

    Args:
        query_engine: Query engine to ask questions and get answers along with retrieved context.
        num_questions (int): Number of questions to evaluate (default: 5).
    """


    q_a_file_name = f"{GENERATED_DIR}/q_a.json"
    with open(q_a_file_name, "r", encoding="utf-8") as json_file:
        q_a = json.load(json_file)

    questions = [qa["question"] for qa in q_a][:num_questions]
    ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]
    generated_answers = []
    retrieved_documents = []

    for question in questions:
        response = query_engine.query(question)
        context = [doc.text for doc in response.source_nodes]
        retrieved_documents.append(context)
        generated_answers.append(response.response)

    test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevance_metric]
    )

"""
### Evaluate results
"""
logger.info("### Evaluate results")

query_engine  = vector_store_index.as_query_engine(similarity_top_k=2)
evaluate_rag(query_engine, num_questions=1)

logger.info("\n\n[DONE]", bright=True)