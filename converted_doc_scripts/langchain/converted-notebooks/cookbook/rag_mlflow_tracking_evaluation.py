from jet.adapters.langchain.chat_ollama import OllamaEmbeddings, ChatOllama
from jet.logger import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from mlflow.genai.scorers import RelevanceToQuery, Correctness, ExpectationsGuidelines
import mlflow
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
# RAG Pipeline with MLflow Tracking, Tracing & Evaluation

This notebook demonstrates how to build a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain and integrate it with MLflow for experiment tracking, tracing, and evaluation.


- **RAG Pipeline Construction**: Build a complete RAG system using LangChain components
- **MLflow Integration**: Track experiments, parameters, and artifacts
- **Tracing**: Monitor inputs, outputs, retrieved documents, scores, prompts, and timings
- **Evaluation**: Use MLflow's built-in scorers to assess RAG performance
- **Best Practices**: Implement proper configuration management and reproducible experiments

We'll build a RAG system that can answer questions about academic papers by:
1. Loading and chunking documents from ArXiv
2. Creating embeddings and a vector store
3. Setting up a retrieval-augmented generation chain
4. Tracking all experiments with MLflow
5. Evaluating the system's performance

![System Diagram](https://miro.medium.com/v2/resize:fit:720/format:webp/1*eiw86PP4hrBBxhjTjP0JUQ.png)

#### Setup
"""
logger.info("# RAG Pipeline with MLflow Tracking, Tracing & Evaluation")

# %pip install -U langchain mlflow langchain-community arxiv pymupdf langchain-text-splitters langchain-ollama


# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

mlflow.set_experiment("LangChain-RAG-MLflow")
mlflow.langchain.autolog()

"""
Define all hyperparameters and configuration in a centralized dictionary. This makes it easy to:
- Track different experiment configurations
- Reproduce results
- Perform hyperparameter tuning

**Key Parameters**:
- `chunk_size`: Size of text chunks for document splitting
- `chunk_overlap`: Overlap between consecutive chunks
- `retriever_k`: Number of documents to retrieve
- `embeddings_model`: Ollama embedding model
- `llm`: Language model for generation
- `temperature`: Sampling temperature for the LLM
"""
logger.info("Define all hyperparameters and configuration in a centralized dictionary. This makes it easy to:")

CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 80,
    "retriever_k": 3,
    "embeddings_model": "nomic-embed-text",
    "system_prompt": "You are a helpful assistant. Use the following context to answer the question. Use three sentences maximum and keep the answer concise.",
    "llm": "gpt-5-nano",
    "temperature": 0,
}

"""
#### ArXiv Dcoument Loading and Processing
"""
logger.info("#### ArXiv Dcoument Loading and Processing")

loader = ArxivLoader(
    query="1706.03762",
    load_max_docs=1,
)
docs = loader.load()
logger.debug(docs[0].metadata)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONFIG["chunk_size"],
    chunk_overlap=CONFIG["chunk_overlap"],
)
chunks = splitter.split_documents(docs)


def join_chunks(chunks):
    return "\n\n".join([chunk.page_content for chunk in chunks])

"""
#### Vector Store and Retriever Setup
"""
logger.info("#### Vector Store and Retriever Setup")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = InMemoryVectorStore.from_documents(
    chunks,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retriever_k"]})

"""
#### RAG Chain Construction using [LCEL](https://python.langchain.com/docs/concepts/lcel/)

Flow:
1. Query → Retriever (finds relevant chunks)
2. Chunks → join_chunks (creates context)
3. Context + Query → Prompt Template
4. Prompt → Language Model → Response
"""
logger.info("#### RAG Chain Construction using [LCEL](https://python.langchain.com/docs/concepts/lcel/)")

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONFIG["system_prompt"] + "\n\nContext:\n{context}\n\n"),
        ("human", "\n{question}\n"),
    ]
)

rag_chain = (
    {
        "context": retriever | RunnableLambda(join_chunks),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

"""
#### Prediction Function with MLflow Tracing

Create a prediction function decorated with `@mlflow.trace` to automatically log:
- Input queries
- Retrieved documents
- Generated responses
- Execution time
- Chain intermediate steps
"""
logger.info("#### Prediction Function with MLflow Tracing")

@mlflow.trace
def predict_fn(question: str) -> str:
    return rag_chain.invoke(question)


sample_question = "What is the main idea of the paper?"
response = predict_fn(sample_question)
logger.debug(f"Question: {sample_question}")
logger.debug(f"Response: {response}")

"""
#### Evaluation Dataset and Scoring

Define an evaluation dataset and run systematic evaluation using [MLflow's built-in scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/#available-scorers):

<u>Evaluation Components:</u>
- **Dataset**: Questions with expected concepts and facts
- **Scorers**: 
  - `RelevanceToQuery`: Measures how relevant the response is to the question
  - `Correctness`: Evaluates factual accuracy of the response
  - `ExpectationsGuidelines`: Checks that output matches expectation guidelines

<u>Best Practices:</u>
- Create diverse test cases covering different query types
- Include expected concepts to guide evaluation
- Use multiple scoring metrics for comprehensive assessment
"""
logger.info("#### Evaluation Dataset and Scoring")

eval_dataset = [
    {
        "inputs": {"question": "What is the main idea of the paper?"},
        "expectations": {
            "key_concepts": ["attention mechanism", "transformer", "neural network"],
            "expected_facts": [
                "attention mechanism is a key component of the transformer model"
            ],
            "guidelines": ["The response must be factual and concise"],
        },
    },
    {
        "inputs": {
            "question": "What's the difference between a transformer and a recurrent neural network?"
        },
        "expectations": {
            "key_concepts": ["sequential", "attention mechanism", "hidden state"],
            "expected_facts": [
                "transformer processes data in parallel while RNN processes data sequentially"
            ],
            "guidelines": [
                "The response must be factual and focus on the difference between the two models"
            ],
        },
    },
    {
        "inputs": {"question": "What does the attention mechanism do?"},
        "expectations": {
            "key_concepts": ["query", "key", "value", "relationship", "similarity"],
            "expected_facts": [
                "attention allows the model to weigh the importance of different parts of the input sequence when processing it"
            ],
            "guidelines": [
                "The response must be factual and explain the concept of attention"
            ],
        },
    },
]

with mlflow.start_run(run_name="baseline_eval") as run:
    mlflow.log_params(CONFIG)

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[RelevanceToQuery(), Correctness(), ExpectationsGuidelines()],
    )

"""
#### Launch MLflow UI to check out the results

<u>What you'll see in the UI:</u>
- **Experiments**: Compare different RAG configurations
- **Runs**: Individual experiment runs with metrics and parameters
- **Traces**: Detailed execution traces showing retrieval and generation steps
- **Evaluation Results**: Scoring metrics and detailed comparisons
- **Artifacts**: Saved models, datasets, and other files

Navigate to `http://localhost:5000` after running the command below.
"""
logger.info("#### Launch MLflow UI to check out the results")

# !mlflow ui

"""
You should see something like this

![MLflow UI image](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Cx7MMy53pAP7150x_hvztA.png)
"""
logger.info("You should see something like this")

logger.info("\n\n[DONE]", bright=True)