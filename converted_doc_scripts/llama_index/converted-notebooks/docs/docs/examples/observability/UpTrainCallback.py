from jet.logger import CustomLogger
from llama_index.callbacks.uptrain.base import UpTrainCallbackHandler
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.readers.web import SimpleWebPageReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/UpTrainCallback.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# UpTrain Callback Handler

UpTrain ([github](https://github.com/uptrain-ai/uptrain) || [website](https://github.com/uptrain-ai/uptrain/) || [docs](https://docs.uptrain.ai/)) is an open-source platform to evaluate and improve GenAI applications. It provides grades for 20+ preconfigured checks (covering language, code, embedding use cases), performs root cause analysis on failure cases and gives insights on how to resolve them. 

This notebook showcases how to use UpTrain Callback Handler to evaluate different components of your RAG pipelines.

## 1. **RAG Query Engine Evaluations**:
The RAG query engine plays a crucial role in retrieving context and generating responses. To ensure its performance and response quality, we conduct the following evaluations:

- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.

## 2. **Sub-Question Query Generation Evaluation**:
The SubQuestionQueryGeneration operator decomposes a question into sub-questions, generating responses for each using an RAG query engine. To measure it's accuracy, we use:

- **[Sub Query Completeness](https://docs.uptrain.ai/predefined-evaluations/query-quality/sub-query-completeness)**: Assures that the sub-questions accurately and comprehensively cover the original query.

## 3. **Re-Ranking Evaluations**:
Re-ranking involves reordering nodes based on relevance to the query and choosing the top nodes. Different evaluations are performed based on the number of nodes returned after re-ranking.

a. Same Number of Nodes
- **[Context Reranking](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-reranking)**: Checks if the order of re-ranked nodes is more relevant to the query than the original order.

b. Different Number of Nodes:
- **[Context Conciseness](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-conciseness)**: Examines whether the reduced number of nodes still provides all the required information.

These evaluations collectively ensure the robustness and effectiveness of the RAG query engine, SubQuestionQueryGeneration operator, and the re-ranking process in the LlamaIndex pipeline.

#### **Note:** 
- We have performed evaluations using basic RAG query engine, the same evaluations can be performed using the advanced RAG query engine as well.
- Same is true for Re-Ranking evaluations, we have performed evaluations using SentenceTransformerRerank, the same evaluations can be performed using other re-rankers as well.

## Install Dependencies and Import Libraries

Install notebook dependencies.
"""
logger.info("# UpTrain Callback Handler")

# %pip install llama-index-readers-web
# %pip install llama-index-callbacks-uptrain
# %pip install -q html2text llama-index pandas tqdm uptrain torch sentence-transformers

"""
Import libraries.
"""
logger.info("Import libraries.")

# from getpass import getpass



"""
## Setup

UpTrain provides you with:
1. Dashboards with advanced drill-down and filtering options
1. Insights and common topics among failing cases
1. Observability and real-time monitoring of production data
1. Regression testing via seamless integration with your CI/CD pipelines

You can choose between the following options for evaluating using UpTrain:
### 1. **UpTrain's Open-Source Software (OSS)**: 
You can use the open-source evaluation service to evaluate your model. In this case, you will need to provide an MLX API key. You can get yours [here](https://platform.openai.com/account/api-keys).

In order to view your evaluations in the UpTrain dashboard, you will need to set it up by running the following commands in your terminal:

```bash
git clone https://github.com/uptrain-ai/uptrain
cd uptrain
bash run_uptrain.sh
```

This will start the UpTrain dashboard on your local machine. You can access it at `http://localhost:3000/dashboard`.

Parameters:
- key_type="openai"
# - api_key="OPENAI_API_KEY"
- project_name="PROJECT_NAME"


### 2. **UpTrain Managed Service and Dashboards**:
Alternatively, you can use UpTrain's managed service to evaluate your model. You can create a free UpTrain account [here](https://uptrain.ai/) and get free trial credits. If you want more trial credits, [book a call with the maintainers of UpTrain here](https://calendly.com/uptrain-sourabh/30min).

The benefits of using the managed service are:
1. No need to set up the UpTrain dashboard on your local machine.
1. Access to many LLMs without needing their API keys.

Once you perform the evaluations, you can view them in the UpTrain dashboard at `https://dashboard.uptrain.ai/dashboard`

Parameters:
- key_type="uptrain"
- api_key="UPTRAIN_API_KEY"
- project_name="PROJECT_NAME"


**Note:** The `project_name` will be the project name under which the evaluations performed will be shown in the UpTrain dashboard.

## Create the UpTrain Callback Handler
"""
logger.info("## Setup")

# os.environ["OPENAI_API_KEY"] = getpass()

callback_handler = UpTrainCallbackHandler(
    key_type="openai",
#     api_key=os.environ["OPENAI_API_KEY"],
    project_name="uptrain_llamaindex",
)

Settings.callback_manager = CallbackManager([callback_handler])

"""
## Load and Parse Documents

Load documents from Paul Graham's essay "What I Worked On".
"""
logger.info("## Load and Parse Documents")

documents = SimpleWebPageReader().load_data(
    [
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
    ]
)

"""
Parse the document into nodes.
"""
logger.info("Parse the document into nodes.")

parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)

"""
# 1. RAG Query Engine Evaluation

UpTrain callback handler will automatically capture the query, context and response once generated and will run the following three evaluations *(Graded from 0 to 1)* on the response:
- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.
"""
logger.info("# 1. RAG Query Engine Evaluation")

index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()

max_characters_per_line = 80
queries = [
    "What did Paul Graham do growing up?",
    "When and how did Paul Graham's mother die?",
    "What, in Paul Graham's opinion, is the most distinctive thing about YC?",
    "When and how did Paul Graham meet Jessica Livingston?",
    "What is Bel, and when and where was it written?",
]
for query in queries:
    response = query_engine.query(query)

"""
# 2. Sub-Question Query Engine Evaluation

The **sub-question query engine** is used to tackle the problem of answering a complex query using multiple data sources. It first breaks down the complex query into sub-questions for each relevant data source, then gathers all the intermediate responses and synthesizes a final response.

UpTrain callback handler will automatically capture the sub-question and the responses for each of them once generated and will run the following three evaluations *(Graded from 0 to 1)* on the response:
- **[Context Relevance](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-relevance)**: Determines if the retrieved context has sufficient information to answer the user query or not.
- **[Factual Accuracy](https://docs.uptrain.ai/predefined-evaluations/context-awareness/factual-accuracy)**: Assesses if the LLM's response can be verified via the retrieved context.
- **[Response Completeness](https://docs.uptrain.ai/predefined-evaluations/response-quality/response-completeness)**: Checks if the response contains all the information required to answer the user query comprehensively.

In addition to the above evaluations, the callback handler will also run the following evaluation:
- **[Sub Query Completeness](https://docs.uptrain.ai/predefined-evaluations/query-quality/sub-query-completeness)**: Assures that the sub-questions accurately and comprehensively cover the original query.
"""
logger.info("# 2. Sub-Question Query Engine Evaluation")

vector_query_engine = VectorStoreIndex.from_documents(
    documents=documents,
    use_async=True,
).as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="documents",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)

"""
# 3. Re-ranking 

Re-ranking is the process of reordering the nodes based on their relevance to the query. There are multiple classes of re-ranking algorithms offered by Llamaindex. We have used LLMRerank for this example.

The re-ranker allows you to enter the number of top n nodes that will be returned after re-ranking. If this value remains the same as the original number of nodes, the re-ranker will only re-rank the nodes and not change the number of nodes. Otherwise, it will re-rank the nodes and return the top n nodes.

We will perform different evaluations based on the number of nodes returned after re-ranking.

## 3a. Re-ranking (With same number of nodes)

If the number of nodes returned after re-ranking is the same as the original number of nodes, the following evaluation will be performed:

- **[Context Reranking](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-reranking)**: Checks if the order of re-ranked nodes is more relevant to the query than the original order.
"""
logger.info("# 3. Re-ranking")

callback_handler = UpTrainCallbackHandler(
    key_type="openai",
#     api_key=os.environ["OPENAI_API_KEY"],
    project_name="uptrain_llamaindex",
)
Settings.callback_manager = CallbackManager([callback_handler])

rerank_postprocessor = SentenceTransformerRerank(
    top_n=3,  # number of nodes after reranking
    keep_retrieval_score=True,
)

index = VectorStoreIndex.from_documents(
    documents=documents,
)

query_engine = index.as_query_engine(
    similarity_top_k=3,  # number of nodes before reranking
    node_postprocessors=[rerank_postprocessor],
)

response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

"""
# 3b. Re-ranking (With different number of nodes)

If the number of nodes returned after re-ranking is the lesser as the original number of nodes, the following evaluation will be performed:

- **[Context Conciseness](https://docs.uptrain.ai/predefined-evaluations/context-awareness/context-conciseness)**: Examines whether the reduced number of nodes still provides all the required information.
"""
logger.info("# 3b. Re-ranking (With different number of nodes)")

callback_handler = UpTrainCallbackHandler(
    key_type="openai",
#     api_key=os.environ["OPENAI_API_KEY"],
    project_name="uptrain_llamaindex",
)
Settings.callback_manager = CallbackManager([callback_handler])

rerank_postprocessor = SentenceTransformerRerank(
    top_n=2,  # Number of nodes after re-ranking
    keep_retrieval_score=True,
)

index = VectorStoreIndex.from_documents(
    documents=documents,
)
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Number of nodes before re-ranking
    node_postprocessors=[rerank_postprocessor],
)

response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

"""
# UpTrain's Dashboard and Insights

Here's a short video showcasing the dashboard and the insights:

![llamaindex_uptrain.gif](https://uptrain-assets.s3.ap-south-1.amazonaws.com/images/llamaindex/llamaindex_uptrain.gif)
"""
logger.info("# UpTrain's Dashboard and Insights")

logger.info("\n\n[DONE]", bright=True)