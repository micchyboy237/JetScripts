from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import (
Settings,
set_global_handler,
)
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from phoenix.evals import (
HallucinationEvaluator,
OllamaFunctionCallingAdapterModel,
QAEvaluator,
RelevanceEvaluator,
run_evals,
)
from phoenix.session.evaluation import (
get_qa_with_reference,
get_retrieved_documents,
)
from phoenix.trace import DocumentEvaluations, SpanEvaluations
from tqdm import tqdm
from urllib.request import urlopen
import json
import openai
import os
import pandas as pd
import phoenix as px
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Observability with Arize Phoenix - Tracing and Evaluating a LlamaIndex Application

LlamaIndex provides high-level APIs that enable users to build powerful applications in a few lines of code. However, it can be challenging to understand what is going on under the hood and to pinpoint the cause of issues. Phoenix makes your LLM applications *observable* by visualizing the underlying structure of each call to your query engine and surfacing problematic `spans`` of execution based on latency, token count, or other evaluation metrics.

In this tutorial, you will:
- Build a simple query engine using LlamaIndex that uses retrieval-augmented generation to answer questions over the Paul Graham Essay,
- Record trace data in [OpenInference tracing](https://github.com/Arize-ai/openinference) format using the global `arize_phoenix` handler
- Inspect the traces and spans of your application to identify sources of latency and cost,
- Export your trace data as a pandas dataframe and run an [LLM Evals](https://docs.arize.com/phoenix/concepts/llm-evals).

ℹ️ This notebook requires an OllamaFunctionCallingAdapter API key.

[Observability Documentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/)

## 1. Install Dependencies and Import Libraries

Install Phoenix, LlamaIndex, and OllamaFunctionCallingAdapter.
"""
logger.info("# Observability with Arize Phoenix - Tracing and Evaluating a LlamaIndex Application")

# !pip install llama-index
# !pip install llama-index-callbacks-arize-phoenix
# !pip install arize-phoenix[evals]
# !pip install "openinference-instrumentation-llama-index>=1.0.0"

# from getpass import getpass

# import nest_asyncio

# nest_asyncio.apply()
pd.set_option("display.max_colwidth", 1000)

"""
## 2. Launch Phoenix

You can run Phoenix in the background to collect trace data emitted by any LlamaIndex application that has been instrumented with the `OpenInferenceTraceCallbackHandler`. Phoenix supports LlamaIndex's [one-click observability](https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/one_click_observability.html) which will automatically instrument your LlamaIndex application! You can consult our [integration guide](https://docs.arize.com/phoenix/integrations/llamaindex) for a more detailed explanation of how to instrument your LlamaIndex application.

Launch Phoenix and follow the instructions in the cell output to open the Phoenix UI (the UI should be empty because we have yet to run the LlamaIndex application).
"""
logger.info("## 2. Launch Phoenix")

session = px.launch_app()

"""
## 3. Configure Your OllamaFunctionCallingAdapter API Key

Set your OllamaFunctionCallingAdapter API key if it is not already set as an environment variable.
"""
logger.info("## 3. Configure Your OllamaFunctionCallingAdapter API Key")


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## 4. Build Index and Create QueryEngine

a. Download Data

b. Load Data

c. Setup Phoenix Tracing

d. Setup LLM And Embedding Model

e. Create Index

f. Create Query Engine

### Download Data
"""
logger.info("## 4. Build Index and Create QueryEngine")

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

"""
### Load Data
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

"""
### Setup Phoenix Tracing

Enable Phoenix tracing within LlamaIndex by setting `arize_phoenix` as the global handler. This will mount Phoenix's [OpenInferenceTraceCallback](https://docs.arize.com/phoenix/integrations/llamaindex) as the global handler. Phoenix uses OpenInference traces - an open-source standard for capturing and storing LLM application traces that enables LLM applications to seamlessly integrate with LLM observability solutions such as Phoenix.
"""
logger.info("### Setup Phoenix Tracing")

set_global_handler("arize_phoenix")

"""
### Setup LLM and Embedding Model
"""
logger.info("### Setup LLM and Embedding Model")


llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.2)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

Settings.llm = llm
Settings.embed_model = embed_model

"""
### Create Index
"""
logger.info("### Create Index")


index = VectorStoreIndex.from_documents(documents)

"""
### Create Query Engine.
"""
logger.info("### Create Query Engine.")

query_engine = index.as_query_engine(similarity_top_k=5)

"""
## 5. Run Your Query Engine and View Your Traces in Phoenix
"""
logger.info("## 5. Run Your Query Engine and View Your Traces in Phoenix")

queries = [
    "what did paul graham do growing up?",
    "why did paul graham start YC?",
]

for query in tqdm(queries):
    query_engine.query(query)

logger.debug(query_engine.query("Who is Paul Graham?"))

logger.debug(f"🚀 Open the Phoenix UI if you haven't already: {session.url}")

"""
## 6. Export and Evaluate Your Trace Data

You can export your trace data as a pandas dataframe for further analysis and evaluation.

In this case, we will export our `retriever` spans into two separate dataframes:
- `queries_df`, in which the retrieved documents for each query are concatenated into a single column,
- `retrieved_documents_df`, in which each retrieved document is "exploded" into its own row to enable the evaluation of each query-document pair in isolation.

This will enable us to compute multiple kinds of evaluations, including:
- relevance: Are the retrieved documents grounded in the response?
- Q&A correctness: Are your application's responses grounded in the retrieved context?
- hallucinations: Is your application making up false information?
"""
logger.info("## 6. Export and Evaluate Your Trace Data")

queries_df = get_qa_with_reference(px.Client())
retrieved_documents_df = get_retrieved_documents(px.Client())

"""
Next, define your evaluation model and your evaluators.

Evaluators are built on top of language models and prompt the LLM to assess the quality of responses, the relevance of retrieved documents, etc., and provide a quality signal even in the absence of human-labeled data. Pick an evaluator type and instantiate it with the language model you want to use to perform evaluations using our battle-tested evaluation templates.
"""
logger.info("Next, define your evaluation model and your evaluators.")

eval_model = OllamaFunctionCallingAdapterModel(
    model="llama3.2", request_timeout=300.0, context_window=4096,
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df, qa_correctness_eval_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df = run_evals(
    dataframe=retrieved_documents_df,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]

px.Client().log_evaluations(
    SpanEvaluations(
        eval_name="Hallucination", dataframe=hallucination_eval_df
    ),
    SpanEvaluations(
        eval_name="QA Correctness", dataframe=qa_correctness_eval_df
    ),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
)

"""
For more details on Phoenix, LLM Tracing, and LLM Evals, checkout the [documentation](https://docs.arize.com/phoenix/).
"""
logger.info("For more details on Phoenix, LLM Tracing, and LLM Evals, checkout the [documentation](https://docs.arize.com/phoenix/).")

logger.info("\n\n[DONE]", bright=True)