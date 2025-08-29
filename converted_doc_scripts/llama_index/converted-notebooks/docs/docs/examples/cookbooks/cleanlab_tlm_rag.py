from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMCompletionEndEvent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cleanlab import CleanlabTLM
from typing import Dict, List, ClassVar
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/cleanlab_tlm_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Trustworthy RAG with the Trustworthy Language Model

This tutorial demonstrates how to use Cleanlab's [Trustworthy Language Model](https://cleanlab.ai/blog/trustworthy-language-model/) (TLM) in any RAG system, to score the trustworthiness of answers and **automatically catch incorrect/hallucinated responses in real-time**.

Today's RAG and Agent applications often produce unreliable responses, because they depend on LLMs which are fundamentally unreliable. Cleanlab’s [Trustworthy Language Model](https://cleanlab.ai/blog/trustworthy-language-model/) scores the trustworthiness of every LLM response in real-time, using state-of-the-art uncertainty estimates for LLMs. Cleanlab works effectively **no matter your RAG architecture or retrieval and indexing processes**. 

To diagnose when RAG answers cannot be trusted, this tutorial demonstrates how to replace your LLM with Cleanlab's to [generate responses and score their trustworthiness](https://docs.llamaindex.ai/en/stable/examples/llm/cleanlab/). You can alternatively use Cleanlab only to score responses from your unmodified RAG system and run other real-time Evals, see [our Evaluation tutorial](https://docs.llamaindex.ai/en/stable/examples/evaluation/Cleanlab/).

## Setup

RAG is all about connecting LLMs to data, to better inform their answers. This tutorial uses Nvidia's Q1 FY2024 earnings report as an example dataset.
Use the following commands to download the data (earnings report) and store it in a directory named `data/`.
"""
logger.info("# Trustworthy RAG with the Trustworthy Language Model")

# !wget -nc 'https://cleanlab-public.s3.amazonaws.com/Datasets/NVIDIA_Financial_Results_Q1_FY2024.md'
# !mkdir -p ./data
# !mv NVIDIA_Financial_Results_Q1_FY2024.md data/

"""
Now let's install the required dependencies.
"""
logger.info("Now let's install the required dependencies.")

# %pip install llama-index-llms-cleanlab llama-index llama-index-embeddings-huggingface

"""
We then initialize Cleanlab's TLM. Here we initialize a CleanlabTLM object with default settings.
"""
logger.info("We then initialize Cleanlab's TLM. Here we initialize a CleanlabTLM object with default settings.")



llm = CleanlabTLM(api_key="your_api_key")

"""
Note: If you encounter `ValidationError` during the above import, please upgrade your python version to >= 3.11

You can achieve better results by playing with the TLM configurations outlined in this [advanced TLM tutorial](https://help.cleanlab.ai/tlm/tutorials/tlm_advanced/).

For example, if your application requires OllamaFunctionCallingAdapter's GPT-4 model and restrict the output tokens to 256, you can configure it using the `options` argument:

```python
options = {
    "model": "gpt-4",
    "max_tokens": 256,
}
llm = CleanlabTLM(api_key="your_api_key", options=options)
```

Let's start by asking the LLM a simple question.
"""
logger.info("Note: If you encounter `ValidationError` during the above import, please upgrade your python version to >= 3.11")

response = llm.complete("What is NVIDIA's ticker symbol?")
logger.debug(response)

"""
TLM not only provides a response but also includes a **trustworthiness score** indicating the confidence that this response is good/accurate. You can access this score from the response itself.
"""
logger.info("TLM not only provides a response but also includes a **trustworthiness score** indicating the confidence that this response is good/accurate. You can access this score from the response itself.")

response.additional_kwargs

"""
## Build a RAG pipeline with TLM

Now let's integrate TLM into a RAG pipeline.
"""
logger.info("## Build a RAG pipeline with TLM")


Settings.llm = llm

"""
### Specify Embedding Model

RAG uses an embedding model to match queries against document chunks to retrieve the most relevant data. Here we opt for a no-cost, local embedding model from Hugging Face. You can use any other embedding model by referring to this [LlamaIndex guide](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#embeddings).
"""
logger.info("### Specify Embedding Model")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

"""
### Load Data and Create Index + Query Engine

Let's create an index from the documents stored in the data directory. The system can index multiple files within the same folder, although for this tutorial, we'll use just one document.
We stick with the default index from LlamaIndex for this tutorial.
"""
logger.info("### Load Data and Create Index + Query Engine")

documents = SimpleDirectoryReader("data").load_data()
for doc in documents:
    doc.excluded_llm_metadata_keys.append("file_path")
index = VectorStoreIndex.from_documents(documents)

"""
The generated index is used to power a query engine over the data.
"""
logger.info("The generated index is used to power a query engine over the data.")

query_engine = index.as_query_engine()

"""
Note that TLM is agnostic to the index and the query engine used for RAG, and is compatible with any choices you make for these components of your system.

In addition, you can just use TLM's trustworthiness score in an existing custom-built RAG pipeline (using any other LLM generator, streaming or not). <br>
To achieve this, you'd need to fetch the prompt sent to LLM (including system instructions, retrieved context, user query, etc.) and the returned response. TLM requires both to predict trustworthiness.

Details about this approach and example code are available [here](https://docs.llamaindex.ai/en/stable/examples/evaluation/Cleanlab/).

### Extract Trustworthiness Score from LLM response

As we saw earlier, Cleanlab's TLM also provides the `trustworthiness_score` in addition to the text, in its response to the prompt. 

To get this score out when TLM is used in a RAG pipeline, Llamaindex provides an [instrumentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/#instrumentation) tool that allows us to observe the events running behind the scenes in RAG. <br> 
We can utilise this tooling to extract `trustworthiness_score` from LLM's response.

Let's define a simple event handler that stores this score for every request sent to the LLM. You can refer to [Llamaindex's](https://docs.llamaindex.ai/en/stable/examples/instrumentation/basic_usage/) documentation for more details on instrumentation.
"""
logger.info("### Extract Trustworthiness Score from LLM response")



class GetTrustworthinessScore(BaseEventHandler):
    events: ClassVar[List[BaseEvent]] = []
    trustworthiness_score: float = 0.0

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "GetTrustworthinessScore"

    def handle(self, event: BaseEvent) -> Dict:
        if isinstance(event, LLMCompletionEndEvent):
            self.trustworthiness_score = event.response.additional_kwargs[
                "trustworthiness_score"
            ]
            self.events.append(event)


root_dispatcher = get_dispatcher()

event_handler = GetTrustworthinessScore()
root_dispatcher.add_event_handler(event_handler)

"""
For each query, we can fetch this score from `event_handler.trustworthiness_score`. Let's see it in action.

## Answering queries with our RAG system

Let's try out our RAG pipeline based on TLM. Here we pose questions with differing levels of complexity.
"""
logger.info("## Answering queries with our RAG system")

def display_response(response):
    response_str = response.response
    trustworthiness_score = event_handler.trustworthiness_score
    logger.debug(f"Response: {response_str}")
    logger.debug(f"Trustworthiness score: {round(trustworthiness_score, 2)}")

"""
### Easy Questions

We first pose straightforward questions that can be directly answered by the provided data and can be easily located within a few lines of text.
"""
logger.info("### Easy Questions")

response = query_engine.query(
    "What was NVIDIA's total revenue in the first quarter of fiscal 2024?"
)
display_response(response)

response = query_engine.query(
    "What was the GAAP earnings per diluted share for the quarter?"
)
display_response(response)

response = query_engine.query(
    "What significant transitions did Jensen Huang, NVIDIA's CEO, comment on?"
)
display_response(response)

"""
TLM returns high trustworthiness scores for these responses, indicating high confidence they are accurate. After doing a quick fact-check (reviewing the original earnings report), we can confirm that TLM indeed accurately answered these questions. In case you're curious, here are relevant excerpts from the data context for these questions:

> NVIDIA (NASDAQ: NVDA) today reported revenue for the first quarter ended April 30, 2023, of $7.19 billion, ...

> GAAP earnings per diluted share for the quarter were $0.82, up 28% from a year ago and up 44% from the previous quarter.

> Jensen Huang, founder and CEO of NVIDIA, commented on the significant transitions the computer industry is undergoing, particularly accelerated computing and generative AI, ...

### Questions without Available Context 

Now let's see how TLM responds to queries that *cannot* be answered using the provided data.
"""
logger.info("### Questions without Available Context")

response = query_engine.query(
    "What factors as per the report were responsible to the decline in NVIDIA's proviz revenue?"
)
display_response(response)

"""
The lower TLM trustworthiness score indicates a bit more uncertainty about the response, which aligns with the lack of information available. Let's try some more questions.
"""
logger.info("The lower TLM trustworthiness score indicates a bit more uncertainty about the response, which aligns with the lack of information available. Let's try some more questions.")

response = query_engine.query(
    "How does the report explain why NVIDIA's Gaming revenue decreased year over year?"
)
display_response(response)

response = query_engine.query(
    "How does NVIDIA's dividend payout for this quarter compare to the industry average?",
)
display_response(response)

"""
We observe that TLM demonstrates the ability to recognize the limitations of the available information. It refrains from generating speculative responses or hallucinations, thereby maintaining the reliability of the question-answering system. This behavior showcases an understanding of the boundaries of the context and prioritizes accuracy over conjecture. 

### Challenging Questions

Let's see how our RAG system responds to harder questions, some of which may be misleading.
"""
logger.info("### Challenging Questions")

response = query_engine.query(
    "How much did Nvidia's revenue decrease this quarter vs last quarter, in terms of $?"
)
display_response(response)

response = query_engine.query(
    "This report focuses on Nvidia's Q1FY2024 financial results. There are mentions of other companies in the report like Microsoft, Dell, ServiceNow, etc. Can you name them all here?",
)
display_response(response)

response = query_engine.query(
    "How many RTX GPU models, including all custom versions released by third-party manufacturers and all revisions across different series, were officially announced in NVIDIA's Q1 FY2024 financial results?",
)
display_response(response)

response = query_engine.query(
    "If NVIDIA's Data Center segment maintains its Q1 FY2024 quarter-over-quarter growth rate for the next four quarters, what would be its projected annual revenue?",
)
display_response(response)

"""
TLM automatically alerts us that these answers are unreliable, by the low trustworthiness score. RAG systems with TLM help you properly exercise caution when you see low trustworthiness scores. Here are the correct answers to the aforementioned questions:

> NVIDIA's revenue increased by $1.14 billion this quarter compared to last quarter.

> Google, Amazon Web Services, Microsoft, Oracle, ServiceNow, Medtronic, Dell Technologies.

> There is not a specific total count of RTX GPUs mentioned.

> Projected annual revenue if this growth rate is maintained for the next four quarters: approximately $26.34 billion.

With TLM, you can easily increase trust in any RAG system! 

Read [TLM's performance benchmarks](https://cleanlab.ai/blog/trustworthy-language-model/) to learn about the effectiveness of the trustworthiness scoring. <br />
Rather than replacing your LLM with Cleanlab's (as done in this tutorial), you can alternatively use Cleanlab only to detect incorrect responses from your existing unmodified RAG system; check out [our real-time Evaluation tutorial](https://docs.llamaindex.ai/en/stable/examples/evaluation/Cleanlab/).
"""
logger.info("TLM automatically alerts us that these answers are unreliable, by the low trustworthiness score. RAG systems with TLM help you properly exercise caution when you see low trustworthiness scores. Here are the correct answers to the aforementioned questions:")

logger.info("\n\n[DONE]", bright=True)