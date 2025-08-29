from jet.models.config import MODELS_CACHE_DIR
from googleapiclient import discovery
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.agent.introspective import (
    ToolInteractiveReflectionAgentWorker,
)
from llama_index.agent.introspective import IntrospectiveAgentWorker
from llama_index.agent.multi_hop.planner import MultiHopPlannerAgent
from llama_index.agent.openai import OllamaFunctionCallingAdapterAgent
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.bridge.pydantic import Field
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from numpy import random
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typing import Dict, Optional, Tuple
from typing import List
import os
import qdrant_client
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
**NOTE:** This notebook was written in 2024, and is not guaranteed to work with the latest version of llama-index. It is presented here for reference only.

![Slide One](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/1.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Two](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/2.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Three](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/3.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Four](https://d3ddy8balm3goa.cloudfront.net/vector-oss-tools/draft/4.svg)

## Observability: Arize AI

Follow the quickstart guide found [here](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-llama-index#quickstart).
"""
logger.info("## Observability: Arize AI")

# %pip install --upgrade \
openinference-instrumentation-llama-index \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    "opentelemetry-proto>=1.12.0" \
    arize-phoenix - q


get_ipython().system = os.system

# !python -m phoenix.server.main serve > arize.log 2>&1 &


endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint))
)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

"""
## Example: A Gang of LLMs Tell A Story
"""
logger.info("## Example: A Gang of LLMs Tell A Story")

# %pip install llama-index-llms-ollama -q
# %pip install llama-index-llms-cohere -q
# %pip install llama-index-llms-anthropic -q
# %pip install llama-index-llms-mistralai -q
# %pip install llama-index-vector-stores-qdrant -q
# %pip install llama-index-agent-openai -q
# %pip install llama-index-agent-introspective -q
# %pip install google-api-python-client -q
# %pip install llama-index-program-openai -q
# %pip install llama-index-readers-file -q

# %pip install pyvis -q

# import nest_asyncio

# nest_asyncio.apply()


anthropic_llm = Anthropic(model="claude-3-opus-20240229")
cohere_llm = Cohere(model="command")
mistral_llm = MistralAI(model="mistral-large-latest")
openai_llm = OllamaFunctionCallingAdapter(
    model="llama3.2", request_timeout=300.0, context_window=4096)

theme = "over-the-top pizza toppings"
start = anthropic_llm.complete(
    f"Please start a random story around {theme}. Limit your response to 20 words."
)
logger.debug(start)

middle = cohere_llm.complete(
    f"Please continue the provided story. Limit your response to 20 words.\n\n {start.text}"
)
climax = mistral_llm.complete(
    f"Please continue the attached story. Your part is the climax of the story, so make it exciting! Limit your response to 20 words.\n\n {start.text + middle.text}"
)
ending = openai_llm.complete(
    f"Please continue the attached story. Your part is the end of the story, so wrap it up! Limit your response to 20 words.\n\n {start.text + middle.text + climax.text}"
)

logger.debug(f"{start}\n\n{middle}\n\n{climax}\n\n{ending}")

logger.debug(f"{start}\n\n{middle}\n\n{climax}\n\n{ending}")

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Five](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/5.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Six](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/6.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Seven](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/7.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Eight](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/8.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Nine](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/9.svg)

## Example: LLMs Lack Access To Updated Data
"""
logger.info("## Example: LLMs Lack Access To Updated Data")

response = mistral_llm.complete(
    "What can you tell me about Georgian Partners?"
)

logger.debug(response)

query = "According to the 2022 Annual Purpose Report, what percentage of customers participated in 2022 ESG survey?"

response = mistral_llm.complete(query)
logger.debug(response)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Ten](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/10.svg)

## Example: RAG Yields More Accurate Responses
"""
logger.info("## Example: RAG Yields More Accurate Responses")

# !mkdir data
# !wget "https://cdn.pathfactory.com/assets/preprocessed/10580/b81532f1-95f3-4a1c-ba0d-80a56726e833/b81532f1-95f3-4a1c-ba0d-80a56726e833.pdf" -O "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/gp-purpose-report-2022.pdf"


loader = SimpleDirectoryReader(
    input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp")
documents = loader.load_data()
index = VectorStoreIndex.from_documents(documents)
rag = index.as_query_engine(llm=mistral_llm)

response = rag.query(query)

logger.debug(response)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Eleven](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/11.svg)

## Example: 3 Steps For Basic RAG (Unpacking the previous Example RAG)

### Step 1: Build Knowledge Store
"""
logger.info(
    "## Example: 3 Steps For Basic RAG (Unpacking the previous Example RAG)")

"""Load the data.

With llama-index, before any transformations are applied,
data is loaded in the `Document` abstraction, which is
a container that holds the text of the document.
"""


loader = SimpleDirectoryReader(
    input_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp")
documents = loader.load_data()

documents[1].text

"""Chunk, Encode, and Store into a Vector Store.

To streamline the process, we can make use of the IngestionPipeline
class that will apply your specified transformations to the
Document's.
"""


client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    ],
    vector_store=vector_store,
)
_nodes = pipeline.run(documents=documents, num_workers=4)

"""Create a llama-index... wait for it... Index.

After uploading your encoded documents into your vector
store of choice, you can connect to it with a VectorStoreIndex
which then gives you access to all of the llama-index functionality.
"""


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
### Step 2: Retrieve Against A Query
"""
logger.info("### Step 2: Retrieve Against A Query")

"""Retrieve relevant documents against a query.

With our Index ready, we can now query it to
retrieve the most relevant document chunks.
"""

retriever = index.as_retriever(similarity_top_k=2)
retrieved_nodes = retriever.retrieve(query)

retrieved_nodes

"""
### Step 3: Generate Final Response
"""
logger.info("### Step 3: Generate Final Response")

"""Context-Augemented Generation.

With our Index ready, we can create a QueryEngine
that handles the retrieval and context augmentation
in order to get the final response.
"""

query_engine = index.as_query_engine(llm=mistral_llm)

logger.debug(
    query_engine.get_prompts()[
        "response_synthesizer:text_qa_template"
    ].default_template.template
)

response = query_engine.query(query)
logger.debug(response)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Twelve](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/12.svg)

[Hi-Resolution Cheat Sheet](https://d3ddy8balm3goa.cloudfront.net/llamaindex/rag-cheat-sheet-final.svg)

## Example: Graph RAG
"""
logger.info("## Example: Graph RAG")


index = PropertyGraphIndex.from_documents(
    documents[10:20],
    llm=openai_llm,
    embed_model=HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    show_progress=True,
)

index.property_graph_store.save_networkx_graph(name="./kg.html")

retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve(query)

for node in nodes:
    logger.debug(node.text)

query_engine = index.as_query_engine(
    include_text=True,
)

response = query_engine.query(query)

logger.debug(str(response))

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Thirteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/13.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Fourteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/14.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Fifteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/15.svg)

## Example: Agent Ingredients — Tool Use

**Note:** LLMs are not very good pseudo-random number generators (see my [LinkedIn post](https://www.linkedin.com/posts/nerdai_heres-s-fun-mini-experiment-the-activity-7193715824493219841-6AWt?utm_source=share&utm_medium=member_desktop) about this)
"""
logger.info("## Example: Agent Ingredients — Tool Use")


def uniform_random_sample(n: int) -> List[float]:
    """Generate a list a of uniform random numbers of size n between 0 and 1."""
    return random.rand(n).tolist()


rs_tool = FunctionTool.from_defaults(fn=uniform_random_sample)

agent = OllamaFunctionCallingAdapterAgent.from_tools(
    [rs_tool], llm=openai_llm, verbose=True)

response = agent.chat(
    "Can you please give me a sample of 10 uniformly random numbers?"
)
logger.debug(str(response))

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Sixteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/16.svg)

## Example: Agent Ingredients — Composable Memory
"""
logger.info("## Example: Agent Ingredients — Composable Memory")


vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    retriever_kwargs={"similarity_top_k": 2},
)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer."""
    return a * b


def mystery(a: int, b: int) -> int:
    """Mystery function on two numbers."""
    return a**2 - b**2


multiply_tool = FunctionTool.from_defaults(fn=multiply)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, mystery_tool], llm=openai_llm, verbose=True
)
agent = agent_worker.as_agent(memory=composable_memory)

"""
### Execute some function calls
"""
logger.info("### Execute some function calls")

response = agent.chat("What is the mystery function on 5 and 6?")

response = agent.chat("What happens if you multiply 2 and 3?")

"""
### New Agent Session

#### Without memory
"""
logger.info("### New Agent Session")

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, mystery_tool], llm=openai_llm, verbose=True
)
agent_without_memory = agent_worker.as_agent()

response = agent_without_memory.chat(
    "What was the output of the mystery function on 5 and 6 again? Don't recompute."
)

"""
#### With memory
"""
logger.info("#### With memory")

llm = OllamaFunctionCallingAdapter(
    model="llama3.2", request_timeout=300.0, context_window=4096)
agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, mystery_tool], llm=openai_llm, verbose=True
)
composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=ChatMemoryBuffer.from_defaults(),
    secondary_memory_sources=[
        vector_memory.copy(
            deep=True
        )  # using a copy here for illustration purposes
    ],
)
agent_with_memory = agent_worker.as_agent(memory=composable_memory)

agent_with_memory.chat_history  # an empty chat history

response = agent_with_memory.chat(
    "What was the output of the mystery function on 5 and 6 again? Don't recompute."
)

response = agent_with_memory.chat(
    "What was the output of the multiply function on 2 and 3 again? Don't recompute."
)

"""
#### Under the hood

Calling `.chat()` will invoke `memory.get()`. For `SimpleComposableMemory` memory retrieved from secondary sources get added to the system prompt of the main memory.
"""
logger.info("#### Under the hood")

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=ChatMemoryBuffer.from_defaults(),
    secondary_memory_sources=[
        vector_memory.copy(
            deep=True
        )  # copy for illustrative purposes to explain what
    ],
)
agent_with_memory = agent_worker.as_agent(memory=composable_memory)

logger.debug(
    agent_with_memory.memory.get(
        "What was the output of the mystery function on 5 and 6 again? Don't recompute."
    )[0]
)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Seventeen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/17.svg)

## Example: Reflection Toxicity Reduction

Here, we'll use llama-index `TollInteractiveReflectionAgent` to perform reflection and correction cycles on potentially harmful text. See the full demo [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/introspective_agent_toxicity_reduction.ipynb).

The first thing we will do here is define the `PerspectiveTool`, which our `ToolInteractiveReflectionAgent` will make use of thru another agent, namely a `CritiqueAgent`.

To use Perspecive's API, you will need to do the following steps:

1. Enable the Perspective API in your Google Cloud projects
2. Generate a new set of credentials (i.e. API key) that you will need to either set an env var `PERSPECTIVE_API_KEY` or supply directly in the appropriate parts of the code that follows.

To perform steps 1. and 2., you can follow the instructions outlined here: https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US.

### Perspective API as Tool
"""
logger.info("## Example: Reflection Toxicity Reduction")


class Perspective:
    """Custom class to interact with Perspective API."""

    attributes = [
        "toxicity",
        "severe_toxicity",
        "identity_attack",
        "insult",
        "profanity",
        "threat",
        "sexually_explicit",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            try:
                api_key = os.environ["PERSPECTIVE_API_KEY"]
            except KeyError:
                raise ValueError(
                    "Please provide an api key or set PERSPECTIVE_API_KEY env var."
                )

        self._client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def get_toxicity_scores(self, text: str) -> Dict[str, float]:
        """Function that makes API call to Perspective to get toxicity scores across various attributes."""
        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {
                att.upper(): {} for att in self.attributes
            },
        }

        response = (
            self._client.comments().analyze(body=analyze_request).execute()
        )
        try:
            return {
                att: response["attributeScores"][att.upper()]["summaryScore"][
                    "value"
                ]
                for att in self.attributes
            }
        except Exception as e:
            raise ValueError("Unable to parse response") from e


perspective = Perspective()


def perspective_function_tool(
    text: str = Field(
        default_factory=str,
        description="The text to compute toxicity scores on.",
    ),
) -> Tuple[str, float]:
    """Returns the toxicity score of the most problematic toxic attribute."""
    scores = perspective.get_toxicity_scores(text=text)
    max_key = max(scores, key=scores.get)
    return (max_key, scores[max_key] * 100)


pespective_tool = FunctionTool.from_defaults(
    perspective_function_tool,
)

perspective_function_tool(text="friendly greetings from python")

"""
### Build Agent To Reduce Toxicity of Harmful Text
"""
logger.info("### Build Agent To Reduce Toxicity of Harmful Text")


verbose = True
critique_agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[pespective_tool], llm=OllamaFunctionCallingAdapter("gpt-3.5-turbo"), verbose=verbose
)
correction_llm = OllamaFunctionCallingAdapter("gpt-4-turbo-preview")


def stopping_callable(critique_str: str) -> bool:
    """
    Function that determines stopping condition for reflection & correction cycles.

    critique_str [str]: The response string provided by the critique agent.
    """
    return "[PASS]" in critique_str


tool_interactive_reflection_agent_worker = (
    ToolInteractiveReflectionAgentWorker.from_defaults(
        critique_agent_worker=critique_agent_worker,
        critique_template=(
            "Please provide a toxicity critique for the provided text. "
            "If the toxicity score is less than 3.0 then please "
            "write '[PASS]' otherwise write '[FAIL]'. "
            "Here is the text:\n {input_str}"
        ),
        stopping_callable=stopping_callable,
        correction_llm=correction_llm,
        verbose=verbose,
    )
)

introspective_agent_worker = IntrospectiveAgentWorker.from_defaults(
    reflective_agent_worker=tool_interactive_reflection_agent_worker,
    verbose=verbose,
)

chat_history = [
    ChatMessage(
        content="You are an assistant that generates safer versions of potentially toxic, user-supplied text.",
        role=MessageRole.SYSTEM,
    )
]

introspective_agent = introspective_agent_worker.as_agent(
    chat_history=chat_history, verbose=verbose
)

harmful_text = (
    "Those individuals who publicly tout eating no meat, "
    "especially when their stated reason has to do with caring "
    "about animals, are some of the most hypocritical people I’ve "
    "ever met. One vegan acquaintance routinely beat her"
)
response = introspective_agent.chat(harmful_text)

logger.debug(response)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Eighteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/18.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Nineteen](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/19.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide Twenty](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/20.svg)

## Example: Agentic RAG
"""
logger.info("## Example: Agentic RAG")


# !mkdir vector_data
# !wget "https://vectorinstitute.ai/wp-content/uploads/2024/02/Vector-Annual-Report-2022-23_accessible_rev0224-1.pdf" -O "./vector_data/Vector-Annual-Report-2022-23_accessible_rev0224-1.pdf"

vector_loader = SimpleDirectoryReader(input_dir="./vector_data")
vector_documents = vector_loader.load_data()
vector_index = VectorStoreIndex.from_documents(vector_documents)
vector_query_engine = vector_index.as_query_engine(llm=mistral_llm)

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="georgian_partners_annual_purpose_report_2022",
            description=(
                "Provides information on purpose initiatives for Georgian Partners in the year 2022."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_annual_report_2023",
            description=(
                "Provides information about Vector in the year 2023."
            ),
        ),
    ),
]

agent = OllamaFunctionCallingAdapterAgent.from_tools(
    query_engine_tools, verbose=True)

response = agent.chat(query)

logger.debug(response)

response = agent.chat(
    "According to Vector Institute's Annual Report 2022-2023, "
    "how many AI jobs were created in Ontario?"
)

logger.debug(response)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide TwentyOne](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/21.svg)

## Example: Multi-hop Agent (WIP)

At the time of this presentation, this is still ongoing work, but despite its unfinished status, it demonstrates the flexibility and advantages for using an agentic interface over extneral knowledge bases (i.e., RAG).

With the multi-hop agent, we aim to solve query's by first planning out the required data elements that should be retrieved in order to be able to answer the question. And so, we're really combining here a few concepts:

- planning
- structured data extraction (using a RAG tool)
- reflection/correction

![multi-hop agent](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/multi-hop-agent.excalidraw.svg)
"""
logger.info("## Example: Multi-hop Agent (WIP)")


index = VectorStoreIndex.from_documents([Document.example()])
tool = QueryEngineTool.from_defaults(
    index.as_query_engine(),
    name="dummy",
    description="dummy",
)


worker = FunctionCallingAgentWorker.from_tools([tool], verbose=True)

agent = MultiHopPlannerAgent(worker, tools=[tool], verbose=True)

agent.create_plan(
    input="Who is more than just a film director, Gene Kelly or Yannis Smaragdis?"
)

"""
![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide TwentyThree](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/23.svg)

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
![Slide TwentyTwo](https://d3ddy8balm3goa.cloudfront.net/georgian-partners-genai-bootcamp/v3/22.svg)
"""

logger.info("\n\n[DONE]", bright=True)
