from IPython.display import display, Markdown
from jet.logger import CustomLogger
from llama_index.indices.managed.vectara import VectaraIndex
from vectara_agentic.agent import Agent
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/managed/vectaraDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vectara Managed Index
In this notebook we are going to show how to use [Vectara](https://vectara.com) with LlamaIndex. Please note that this notebook is for Vectara ManagedIndex versions >=0.4.0.

[Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications. 

Vectara provides an end-to-end managed service for Retrieval Augmented Generation or [RAG](https://vectara.com/grounded-generation/), which includes:

1. An integrated API for processing input data, including text extraction from documents and ML-based chunking.

2. The state-of-the-art [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model. Each text chunk is encoded into a vector embedding using Boomerang, and stored in the Vectara internal knowledge (vector+text) store. Thus, when using Vectara with LlamaIndex you do not need to call a separate embedding model - this happens automatically within the Vectara backend.

3. A query service that automatically encodes the query into embeddings and retrieves the most relevant text segmentsthrough [hybrid search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) and a variety of [reranking](https://docs.vectara.com/docs/api-reference/search-apis/reranking) strategies, including a [multilingual reranker](https://docs.vectara.com/docs/learn/vectara-multi-lingual-reranker), [maximal marginal relevance (MMR) reranker](https://docs.vectara.com/docs/learn/mmr-reranker), [user-defined function reranker](https://docs.vectara.com/docs/learn/user-defined-function-reranker), and a [chain reranker](https://docs.vectara.com/docs/learn/chain-reranker) that provides a way to chain together multiple reranking methods to achieve better control over the reranking, combining the strengths of various reranking methods.

4. An option to create a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview) with a wide selection of LLM summarizers (including Vectara's [Mockingbird](https://vectara.com/blog/mockingbird-is-a-rag-specific-llm-that-beats-gpt-4-gemini-1-5-pro-in-rag-output-quality/), trained specifically for RAG-based tasks), based on the retrieved documents, including citations.

See the [Vectara API documentation](https://docs.vectara.com/docs/) for more information on how to use the API.

The main benefits of using Vectara RAG-as-a-service to build your application are:
* **Accuracy and Quality**: Vectara provides an end-to-end platform that focuses on eliminating hallucinations, reducing bias, and safeguarding copyright integrity.
* **Security**: Vectara's platform provides acess control--protecting against prompt injection attacks--and meets SOC2 and HIPAA compliance.
* **Explainability**: Vectara makes it easy to troubleshoot bad results by clearly explaining rephrased queries, LLM prompts, retrieved results, and agent actions.

## Getting Started

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Vectara Managed Index")

# !pip install llama-index llama-index-indices-managed-vectara

"""
To get started with Vectara, [sign up](https://vectara.com/integrations/llamaindex) (if you haven't already) and follow our [quickstart guide](https://docs.vectara.com/docs/quickstart) to create a corpus and an API key.

Once you have these, you can provide them as environment variables `VECTARA_CORPUS_KEY`, and `VECTARA_API_KEY`. Make sure your API key has both query and index permissions.

## RAG with LlamaIndex and Vectara

There are a few ways you can index your data into Vectara, including:
1. With the `from_documents()` or `insert_file()` methods of `VectaraIndex`
2. Uploading files directly in the [Vectara console](https://console.vectara.com/)
3. Using Vectara's [file upload](https://docs.vectara.com/docs/rest-api/upload-file) or [document index](https://docs.vectara.com/docs/rest-api/create-corpus-document) APIs
4. Using [vectara-ingest](https://github.com/vectara/vectara-ingest), an open source crawler/indexer project
5. Using one of our ingest integration partners like Airbyte, Unstructured or DataVolo.

For this purpose, we will use a simple set of small documents, so using `VectaraIndex` directly for the ingest is good enough.

Let's ingest the "AI bill of rights" document into our new corpus.
"""
logger.info("## RAG with LlamaIndex and Vectara")


url = "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf"
response = requests.get(url)
local_path = "ai-bill-of-rights.pdf"
with open(local_path, "wb") as file:
    file.write(response.content)

index = VectaraIndex()
index.insert_file(
    local_path, metadata={"name": "AI bill of rights", "year": 2022}
)

"""
### Running single queries with Vectara Query Engine
Now that we've uploaded the document (or if documents have been uploaded previously) we can go and ask questions directly in LlamaIndex. This activates Vectara's RAG pipeline. 

To use Vectara's internal LLM for summarization, make sure you specify `summary_enabled=True` when you generate the Query engine. Here's an example:
"""
logger.info("### Running single queries with Vectara Query Engine")

questions = [
    "What are the risks of AI?",
    "What should we do to prevent bad actors from using AI?",
    "What are the benefits?",
]

qe = index.as_query_engine(
    n_sentences_before=1,
    n_sentences_after=1,
    summary_enabled=True,
    summary_prompt_name="mockingbird-1.0-2024-07-16",
)
qe.query(questions[0]).response

"""
If you want the response to be returned in streaming mode, simply set `streaming=True`
"""
logger.info("If you want the response to be returned in streaming mode, simply set `streaming=True`")

qe = index.as_query_engine(
    n_sentences_before=1,
    n_sentences_after=1,
    summary_enabled=True,
    summary_prompt_name="mockingbird-1.0-2024-07-16",
    streaming=True,
)
response = qe.query(questions[0])

response.print_response_stream()

"""
### Using Vectara Chat

Vectara also supports a simple chat mode. In this mode the chat history is maintained by Vectara and so you don't have to worry about it. To use it simple call `as_chat_engine`.

(Chat mode always uses Vectara's summarization so you don't have to explicitly specify `summary_enabled=True` like before)
"""
logger.info("### Using Vectara Chat")

ce = index.as_chat_engine(n_sentences_before=1, n_sentences_after=1)

for q in questions:
    logger.debug(f"Question: {q}\n")
    response = ce.chat(q).response
    logger.debug(f"Response: {response}\n")

"""
Of course streaming works as well with Chat:
"""
logger.info("Of course streaming works as well with Chat:")

ce = index.as_chat_engine(
    n_sentences_before=1, n_sentences_after=1, streaming=True
)

response = ce.stream_chat("Will artificial intelligence rule the government?")

response.print_response_stream()

"""
### Agentic RAG

Vectara also has its own package, [vectara-agentic](https://github.com/vectara/py-vectara-agentic), built on top of many features from LlamaIndex to easily implement agentic RAG applications. It allows you to create your own AI assistant with RAG query tools and other custom tools, such as making API calls to retrieve information from financial websites. You can find the full documentation for vectara-agentic [here](https://vectara.github.io/vectara-agentic-docs/).

Let's create a ReAct Agent with a single RAG tool using vectara-agentic (to create a ReAct agent, specify `VECTARA_AGENTIC_AGENT_TYPE` as `"REACT"` in your environment).

Vectara does not yet have an LLM capable of acting as an agent for planning and tool use, so we will need to use another LLM as the driver of the agent resoning.

# In this demo, we are using MLX's GPT4o. Please make sure you have `OPENAI_API_KEY` defined in your environment or specify another LLM with the corresponding key (for the full list of supported LLMs, check out our [documentation](https://vectara.github.io/vectara-agentic-docs/introduction.html#try-it-yourself) for setting up your environment).
"""
logger.info("### Agentic RAG")

# !pip install -U vectara-agentic


agent = Agent.from_corpus(
    tool_name="query_ai",
    data_description="AI regulations",
    assistant_specialty="artificial intelligence",
    vectara_reranker="mmr",
    vectara_rerank_k=50,
    vectara_summary_num_results=5,
    vectara_summarizer="mockingbird-1.0-2024-07-16",
    verbose=True,
)

response = agent.chat(
    "What are the risks of AI? What are the benefits? Compare and contrast and provide a summary with arguments for and against from experts."
)

display(Markdown(response))

logger.info("\n\n[DONE]", bright=True)