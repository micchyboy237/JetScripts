from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from tqdm import tqdm
from phoenix.trace import DocumentEvaluations, SpanEvaluations
from phoenix.session.evaluation import (
    get_qa_with_reference,
    get_retrieved_documents,
)
from phoenix.evals import (
    HallucinationEvaluator,
    OllamaModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import (
    Settings,
    set_global_handler,
)
import phoenix as px
import pandas as pd
import openai
import nest_asyncio
from urllib.request import urlopen
from getpass import getpass
import os
import json
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <h1 align="center">Observability with Arize Phoenix - Tracing and Evaluating a LlamaIndex Application</h1>
#
# LlamaIndex provides high-level APIs that enable users to build powerful applications in a few lines of code. However, it can be challenging to understand what is going on under the hood and to pinpoint the cause of issues. Phoenix makes your LLM applications *observable* by visualizing the underlying structure of each call to your query engine and surfacing problematic `spans`` of execution based on latency, token count, or other evaluation metrics.
#
# In this tutorial, you will:
# - Build a simple query engine using LlamaIndex that uses retrieval-augmented generation to answer questions over the Paul Graham Essay,
# - Record trace data in [OpenInference tracing](https://github.com/Arize-ai/openinference) format using the global `arize_phoenix` handler
# - Inspect the traces and spans of your application to identify sources of latency and cost,
# - Export your trace data as a pandas dataframe and run an [LLM Evals](https://docs.arize.com/phoenix/concepts/llm-evals).
#
# ℹ️ This notebook requires an Ollama API key.
#
# [Observability Documentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/)

# 1. Install Dependencies and Import Libraries
#
# Install Phoenix, LlamaIndex, and Ollama.

# !pip install llama-index
# !pip install llama-index-callbacks-arize-phoenix
# !pip install arize-phoenix[evals]
# !pip install "openinference-instrumentation-llama-index>=1.0.0"


nest_asyncio.apply()
pd.set_option("display.max_colwidth", 1000)

# 2. Launch Phoenix
#
# You can run Phoenix in the background to collect trace data emitted by any LlamaIndex application that has been instrumented with the `OpenInferenceTraceCallbackHandler`. Phoenix supports LlamaIndex's [one-click observability](https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/one_click_observability.html) which will automatically instrument your LlamaIndex application! You can consult our [integration guide](https://docs.arize.com/phoenix/integrations/llamaindex) for a more detailed explanation of how to instrument your LlamaIndex application.
#
# Launch Phoenix and follow the instructions in the cell output to open the Phoenix UI (the UI should be empty because we have yet to run the LlamaIndex application).

session = px.launch_app()

# 3. Configure Your Ollama API Key
#
# Set your Ollama API key if it is not already set as an environment variable.


# os.environ["OPENAI_API_KEY"] = "sk-..."

# 4. Build Index and Create QueryEngine
#
# a. Download Data
#
# b. Load Data
#
# c. Setup Phoenix Tracing
#
# d. Setup LLM And Embedding Model
#
# e. Create Index
#
# f. Create Query Engine
#

# Download Data

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

# Load Data


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

# Setup Phoenix Tracing
#
# Enable Phoenix tracing within LlamaIndex by setting `arize_phoenix` as the global handler. This will mount Phoenix's [OpenInferenceTraceCallback](https://docs.arize.com/phoenix/integrations/llamaindex) as the global handler. Phoenix uses OpenInference traces - an open-source standard for capturing and storing LLM application traces that enables LLM applications to seamlessly integrate with LLM observability solutions such as Phoenix.

set_global_handler("arize_phoenix")

# Setup LLM and Embedding Model


llm = Ollama(model="llama3.2", request_timeout=300.0,
             context_window=4096, temperature=0.2)
embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

Settings.llm = llm
Settings.embed_model = embed_model

# Create Index


index = VectorStoreIndex.from_documents(documents)

# Create Query Engine.

query_engine = index.as_query_engine(similarity_top_k=5)

# 5. Run Your Query Engine and View Your Traces in Phoenix
#

queries = [
    "what did paul graham do growing up?",
    "why did paul graham start YC?",
]

for query in tqdm(queries):
    query_engine.query(query)

print(query_engine.query("Who is Paul Graham?"))

print(f"🚀 Open the Phoenix UI if you haven't already: {session.url}")

# 6. Export and Evaluate Your Trace Data
#
# You can export your trace data as a pandas dataframe for further analysis and evaluation.
#
# In this case, we will export our `retriever` spans into two separate dataframes:
# - `queries_df`, in which the retrieved documents for each query are concatenated into a single column,
# - `retrieved_documents_df`, in which each retrieved document is "exploded" into its own row to enable the evaluation of each query-document pair in isolation.
#
# This will enable us to compute multiple kinds of evaluations, including:
# - relevance: Are the retrieved documents grounded in the response?
# - Q&A correctness: Are your application's responses grounded in the retrieved context?
# - hallucinations: Is your application making up false information?

queries_df = get_qa_with_reference(px.Client())
retrieved_documents_df = get_retrieved_documents(px.Client())

# Next, define your evaluation model and your evaluators.
#
# Evaluators are built on top of language models and prompt the LLM to assess the quality of responses, the relevance of retrieved documents, etc., and provide a quality signal even in the absence of human-labeled data. Pick an evaluator type and instantiate it with the language model you want to use to perform evaluations using our battle-tested evaluation templates.

eval_model = OllamaModel(
    model="llama3.1", request_timeout=300.0, context_window=4096,
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

# For more details on Phoenix, LLM Tracing, and LLM Evals, checkout the [documentation](https://docs.arize.com/phoenix/).

logger.info("\n\n[DONE]", bright=True)
