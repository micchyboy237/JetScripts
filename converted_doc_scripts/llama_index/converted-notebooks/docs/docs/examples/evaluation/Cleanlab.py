from jet.models.config import MODELS_CACHE_DIR
from cleanlab_tlm import TrustworthyRAG, Eval, get_default_evals
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMPredictStartEvent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, ClassVar
import os
import os, re
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/Cleanlab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Trustworthy RAG with LlamaIndex and Cleanlab

LLMs occasionally hallucinate incorrect answers, especially for questions not well-supported within their training data. While organizations are adopting Retrieval Augmented Generation (RAG) to power LLMs with proprietary data, incorrect RAG responses remain a problem.

This tutorial shows how to build **trustworthy** RAG applications: use [Cleanlab](https://help.cleanlab.ai/tlm/) to score the trustworthiness of every LLM response, and diagnose *why* responses are untrustworthy via evaluations of specific RAG components.

Powered by [state-of-the-art uncertainty estimation](https://cleanlab.ai/blog/trustworthy-language-model/), Cleanlab trustworthiness scores help you automatically catch incorrect responses from any LLM application. Trust scoring happens in real-time and does not require any data labeling or model training work. Cleanlab provides additional real-time Evals for specific RAG components like the retrieved context, which help you root cause *why* RAG responses were incorrect. Cleanlab makes it easy to prevent inaccurate responses from your RAG app, and avoid losing your users' trust.

## Setup

This tutorial requires a:
- Cleanlab API Key: Sign up at [tlm.cleanlab.ai/](https://tlm.cleanlab.ai/) to get a free key
- OllamaFunctionCalling API Key: To make completion requests to an LLM

Start by installing the required dependencies.
"""
logger.info("# Trustworthy RAG with LlamaIndex and Cleanlab")

# %pip install llama-index cleanlab-tlm




"""
Initialize the OllamaFunctionCalling client using its API key.
"""
logger.info("Initialize the OllamaFunctionCalling client using its API key.")

# os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"

llm = OllamaFunctionCalling(model="llama3.2")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
Now, we initialize Cleanlab's client with default configurations. You can achieve better detection accuracy and latency by adjusting [optional configurations](https://help.cleanlab.ai/tlm/tutorials/tlm_advanced/).
"""
logger.info("Now, we initialize Cleanlab's client with default configurations. You can achieve better detection accuracy and latency by adjusting [optional configurations](https://help.cleanlab.ai/tlm/tutorials/tlm_advanced/).")

os.environ["CLEANLAB_TLM_API_KEY"] = "<your-cleanlab-api-key"

trustworthy_rag = (
    TrustworthyRAG()
)  # Optional configurations can improve accuracy/latency

"""
## Read data

This tutorial uses Nvidia’s Q1 FY2024 earnings report as an example data source for populating the RAG application's knowledge base.
"""
logger.info("## Read data")

# !wget -nc 'https://cleanlab-public.s3.amazonaws.com/Datasets/NVIDIA_Financial_Results_Q1_FY2024.md'
# !mkdir -p ./data
# !mv NVIDIA_Financial_Results_Q1_FY2024.md data/

with open(
    "data/NVIDIA_Financial_Results_Q1_FY2024.md", "r", encoding="utf-8"
) as file:
    data = file.read()

logger.debug(data[:200])

"""
## Build a RAG pipeline

Now let's build a simple RAG pipeline with LlamaIndex. We have already initialized the OllamaFunctionCalling API for both as LLM and Embedding model.
"""
logger.info("## Build a RAG pipeline")


Settings.llm = llm
Settings.embed_model = embed_model

"""
### Load Data and Create Index + Query Engine

Let's create an index from the document we just pulled above. We stick with the default index from LlamaIndex for this tutorial.
"""
logger.info("### Load Data and Create Index + Query Engine")

documents = SimpleDirectoryReader("data").load_data()
for doc in documents:
    doc.excluded_llm_metadata_keys.append(
        "file_path"
    )  # file_path wouldn't be a useful metadata to add to LLM's context since our datasource contains just 1 file
index = VectorStoreIndex.from_documents(documents)

"""
The generated index is used to power a query engine over the data.
"""
logger.info("The generated index is used to power a query engine over the data.")

query_engine = index.as_query_engine()

"""
Note that Cleanlab is agnostic to the index and the query engine used for RAG, and is compatible with any choices you make for these components of your system.

In addition, you can just use Cleanlab in an existing custom-built RAG pipeline (using any other LLM generator, streaming or not). <br>
Cleanlab just needs the prompt sent to your LLM (including system instructions, retrieved context, user query, etc.) and the generated response.

We define an event handler that stores the prompt that LlamaIndex sends to the LLM. Refer to the [instrumentation documentation](https://docs.llamaindex.ai/en/stable/examples/instrumentation/basic_usage/) for more details.
"""
logger.info("Note that Cleanlab is agnostic to the index and the query engine used for RAG, and is compatible with any choices you make for these components of your system.")



class PromptEventHandler(BaseEventHandler):
    events: ClassVar[List[BaseEvent]] = []
    PROMPT_TEMPLATE: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "PromptEventHandler"

    def handle(self, event) -> None:
        if isinstance(event, LLMPredictStartEvent):
            self.PROMPT_TEMPLATE = event.template.default_template.template
            self.events.append(event)


root_dispatcher = get_dispatcher()

event_handler = PromptEventHandler()
root_dispatcher.add_event_handler(event_handler)

"""
For each query, we can fetch the prompt from `event_handler.PROMPT_TEMPLATE`. Let's see it in action.

## Use our RAG application

Now that the vector database is loaded with text chunks and their corresponding embeddings, we can start querying it to answer questions.
"""
logger.info("## Use our RAG application")

query = "What was NVIDIA's total revenue in the first quarter of fiscal 2024?"

response = query_engine.query(query)
logger.debug(response)

"""
This response is indeed correct for our simple query. Let's see the document chunks that LlamaIndex retrieved for this query, from which we can easy verify this response was right.
"""
logger.info("This response is indeed correct for our simple query. Let's see the document chunks that LlamaIndex retrieved for this query, from which we can easy verify this response was right.")

def get_retrieved_context(response, print_chunks=False):
    if isinstance(response, list):
        texts = [node.text for node in response]
    else:
        texts = [src.node.text for src in response.source_nodes]

    if print_chunks:
        for idx, text in enumerate(texts):
            logger.debug(f"--- Chunk {idx + 1} ---\n{text[:200]}...")
    return "\n".join(texts)

context_str = get_retrieved_context(response, True)

"""
## Add a Trust Layer with Cleanlab

Let's add a detection layer to flag untrustworthy RAG responses in real-time. TrustworthyRAG runs Cleanlab's state-of-the-art uncertainty estimator, the [Trustworthy Language Model](https://cleanlab.ai/tlm/), to provide a **trustworthiness score** indicating overall confidence that your RAG's response is *correct*. 

To diagnose *why* responses are untrustworthy, TrustworthyRAG can run additional evaluations of specific RAG components. Let's see what Evals it runs by default:
"""
logger.info("## Add a Trust Layer with Cleanlab")

default_evals = get_default_evals()
for eval in default_evals:
    logger.debug(f"{eval.name}")

"""
Each Eval returns a score between 0-1 (higher is better) that assesses a different aspect of your RAG system:

1. **context_sufficiency**: Evaluates whether the retrieved context contains sufficient information to completely answer the query. A low score indicates that key information is missing from the context (perhaps due to poor retrieval or missing documents).

2. **response_groundedness**: Evaluates whether claims/information stated in the response are explicitly supported by the provided context.

3. **response_helpfulness**: Evaluates whether the response attempts to answer the user query in a helpful manner.

4. **query_ease**: Evaluates whether the user query seems easy for an AI system to properly handle. Complex, vague, tricky, or disgruntled-sounding queries receive lower scores.

To run TrustworthyRAG, we need the prompt sent to the LLM, which includes the system message, retrieved chunks, the user's query, and the LLM's response.
The event handler defined above provides this prompt.
Let's define a helper function to run Cleanlab's detection.
"""
logger.info("Each Eval returns a score between 0-1 (higher is better) that assesses a different aspect of your RAG system:")

def get_eval(query, response, event_handler, evaluator):
    context = get_retrieved_context(response)
    pt = event_handler.PROMPT_TEMPLATE
    full_prompt = pt.format(context_str=context, query_str=query)

    eval_result = evaluator.score(
        query=query,
        context=context,
        response=response.response,
        prompt=full_prompt,
    )
    logger.debug("### Evaluation results:")
    for metric, value in eval_result.items():
        logger.debug(f"{metric}: {value['score']}")


def get_answer(query, evaluator=trustworthy_rag, event_handler=event_handler):
    response = query_engine.query(query)

    logger.debug(
        f"### Query:\n{query}\n\n### Trimmed Context:\n{get_retrieved_context(response)[:300]}..."
    )
    logger.debug(f"\n### Generated response:\n{response.response}\n")

    get_eval(query, response, event_handler, evaluator)

get_eval(query, response, event_handler, trustworthy_rag)

"""
**Analysis:** The high `trustworthiness_score` indicates this response is very trustworthy, i.e. non-hallucinated and likely correct. The context that was retrieved here is sufficient to answer this query, as reflected by the high `context_sufficiency` score. The high `query_ease` score indicates this is a straightforward query as well.

Now let’s run a *challenging* query that **cannot** be answered using the only document in our RAG application's knowledge base.
"""
logger.info("Now let’s run a *challenging* query that **cannot** be answered using the only document in our RAG application's knowledge base.")

get_answer(
    "How does the report explain why NVIDIA's Gaming revenue decreased year over year?"
)

"""
**Analysis:** The generator LLM avoids conjecture by providing a reliable response, as seen in the high `trustworthiness_score`. The low `context_sufficiency` score reflects that the retrieved context was lacking, and the response doesn’t actually answer the user’s query, as indicated by the low `response_helpfulness`.

Let’s see how our RAG system responds to another *challenging* question.
"""
logger.info("Let’s see how our RAG system responds to another *challenging* question.")

get_answer(
    "How much did Nvidia's revenue decrease this quarter vs last quarter, in dollars?"
)

"""
**Analysis**: The generated response incorrectly states that NVIDIA's revenue decreased this quarter, when in fact the referenced report notes a 19% increase quarter-over-quarter. 

Cleanlab's low trustworthiness score helps us automatically catch this incorrect RAG response in real-time!  To root-cause why this response was untrustworthy, we see the `response_groundedness` score is low, which indicates our LLM model is to blame for fabricating this false information. 

Let's try another one:
"""
logger.info("Cleanlab's low trustworthiness score helps us automatically catch this incorrect RAG response in real-time!  To root-cause why this response was untrustworthy, we see the `response_groundedness` score is low, which indicates our LLM model is to blame for fabricating this false information.")

get_answer(
    "If NVIDIA's Data Center segment maintains its Q1 FY2024 quarter-over-quarter growth rate for the next four quarters, what would be its projected annual revenue?"
)

"""
**Analysis**: Reviewing the generated response, we find it overstates (sums up the financials of Q1) the projected revenue. Again Cleanlab helps us automatically catch this incorrect response via its low `trustworthiness_score`.  Based on the additional Evals, the root cause of this issue again appears to be the LLM model failing to ground its response in the retrieved context.

### Custom Evals

You can also specify custom evaluations to assess specific criteria, and combine them with the default evaluations for comprehensive/tailored assessment of your RAG system.

For instance, here's how to create and run a custom eval that checks the conciseness of the generated response.
"""
logger.info("### Custom Evals")

conciseness_eval = Eval(
    name="response_conciseness",
    criteria="Evaluate whether the Generated response is concise and to the point without unnecessary verbosity or repetition. A good response should be brief but comprehensive, covering all necessary information without extra words or redundant explanations.",
    response_identifier="Generated Response",
)

combined_evals = get_default_evals() + [conciseness_eval]

combined_trustworthy_rag = TrustworthyRAG(evals=combined_evals)

get_answer(
    "What significant transitions did Jensen comment on?",
    evaluator=combined_trustworthy_rag,
)

"""
### Replace your LLM with Cleanlab's

Beyond evaluating responses already generated from your LLM, Cleanlab can also generate responses and evaluate them simultaneously (using one of many [supported models](https://help.cleanlab.ai/tlm/api/python/tlm/#class-tlmoptions)). <br />
You can do this by calling `trustworthy_rag.generate(query=query, context=context, prompt=full_prompt)` <br />
This replaces your own LLM within your RAG system and can be more convenient/accurate/faster.

Let's replace our OllamaFunctionCalling LLM to call Cleanlab's endpoint instead:
"""
logger.info("### Replace your LLM with Cleanlab's")

query = "How much did Nvidia's revenue decrease this quarter vs last quarter, in dollars?"
relevant_chunks = query_engine.retrieve(query)
context = get_retrieved_context(relevant_chunks)
logger.debug(f"### Query:\n{query}\n\n### Trimmed Context:\n{context[:300]}")

pt = event_handler.PROMPT_TEMPLATE
full_prompt = pt.format(context_str=context, query_str=query)

result = trustworthy_rag.generate(
    query=query, context=context, prompt=full_prompt
)
logger.debug(f"\n### Generated Response:\n{result['response']}\n")
logger.debug("### Evaluation Scores:")
for metric, value in result.items():
    if metric != "response":
        logger.debug(f"{metric}: {value['score']}")

"""
While it remains hard to achieve a RAG application that will accurately answer *any* possible question, you can easily use Cleanlab to deploy a *trustworthy* RAG application which at least flags answers that are likely inaccurate.  Learn more about optional configurations you can adjust to improve accuracy/latency in the [Cleanlab documentation](https://help.cleanlab.ai/tlm/).
"""
logger.info("While it remains hard to achieve a RAG application that will accurately answer *any* possible question, you can easily use Cleanlab to deploy a *trustworthy* RAG application which at least flags answers that are likely inaccurate.  Learn more about optional configurations you can adjust to improve accuracy/latency in the [Cleanlab documentation](https://help.cleanlab.ai/tlm/).")

logger.info("\n\n[DONE]", bright=True)