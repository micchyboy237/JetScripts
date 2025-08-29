import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import (
    AgentStream,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import (
    ObjectIndex,
    ObjectRetriever,
)
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from jet.llm.mlx.adapters.rerankers.mlx_llama_index_cohere_rerank_adapter import CohereRerank
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from tqdm.notebook import tqdm
from typing import Callable
import os
import pickle
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/multi_document_agents-v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Document Agents (V1)

In this guide, you learn towards setting up a multi-document agent over the LlamaIndex documentation.

This is an extension of V0 multi-document agents with the additional features:
- Reranking during document (tool) retrieval
- Query planning tool that the agent can use to plan 


We do this with the following architecture:

- setup a "document agent" over each Document: each doc agent can do QA/summarization within its doc
- setup a top-level agent over this set of document agents. Do tool retrieval and then do CoT over the set of tools to answer a question.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Multi-Document Agents (V1)")

# %pip install llama-index-core
# %pip install llama-index-agent-openai
# %pip install llama-index-readers-file
# %pip install llama-index-postprocessor-cohere-rerank
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-ollama
# %pip install unstructured[html]

# %load_ext autoreload
# %autoreload 2

"""
## Setup and Download Data

In this section, we'll load in the LlamaIndex documentation.

**NOTE:** This command will take a while to run, it will download the entire LlamaIndex documentation. In my testing, this took about 15 minutes.
"""
logger.info("## Setup and Download Data")

domain = "docs.llamaindex.ai"
docs_url = "https://docs.llamaindex.ai/en/latest/"
# !wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains {domain} --no-parent {docs_url}


reader = UnstructuredReader()


all_files_gen = Path("./docs.llamaindex.ai/").rglob("*")
all_files = [f.resolve() for f in all_files_gen]

all_html_files = [f for f in all_files if f.suffix.lower() == ".html"]

len(all_html_files)

useful_files = [
    x
    for x in all_html_files
    if "understanding" in str(x).split(".")[-2]
    or "examples" in str(x).split(".")[-2]
]
logger.debug(len(useful_files))


doc_limit = 100

docs = []
for idx, f in enumerate(useful_files):
    if idx > doc_limit:
        break
    logger.debug(f"Idx {idx}/{len(useful_files)}")
    loaded_docs = reader.load_data(file=f, split_documents=True)

    loaded_doc = Document(
        text="\n\n".join([d.get_content() for d in loaded_docs]),
        metadata={"path": str(f)},
    )
    logger.debug(loaded_doc.metadata["path"])
    docs.append(loaded_doc)

logger.debug(len(docs))

"""
Define Global LLM + Embeddings
"""
logger.info("Define Global LLM + Embeddings")


# os.environ["OPENAI_API_KEY"] = "sk-..."


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit")
Settings.llm = llm
Settings.embed_model = MLXEmbedding(
    model="mxbai-embed-large", embed_batch_size=256
)

"""
## Building Multi-Document Agents

In this section we show you how to construct the multi-document agent. We first build a document agent for each document, and then define the top-level parent agent with an object index.

### Build Document Agent for each Document

In this section we define "document agents" for each document.

We define both a vector index (for semantic search) and summary index (for summarization) for each document. The two query engines are then converted into tools that are passed to an MLX function calling agent.

This document agent can dynamically choose to perform semantic search or summarization within a given document.

We create a separate document agent for each city.
"""
logger.info("## Building Multi-Document Agents")


async def build_agent_per_doc(nodes, file_base):
    vi_out_path = f"./data/llamaindex_docs/{file_base}"
    summary_out_path = f"./data/llamaindex_docs/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/llamaindex_docs/").mkdir(parents=True, exist_ok=True)
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
        )

    summary_index = SummaryIndex(nodes)

    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            summary_query_engine.query(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name=f"vector_tool_{file_base}",
            description=f"Useful for questions related to specific facts",
        ),
        QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            name=f"summary_tool_{file_base}",
            description=f"Useful for summarization questions",
        ),
    ]

    function_llm = MLXLlamaIndexLLMAdapter(
        model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
    agent = FunctionAgent(
        tools=query_engine_tools,
        llm=function_llm,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the LlamaIndex docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary


async def build_agents(docs):
    node_parser = SentenceSplitter()

    agents_dict = {}
    extra_info_dict = {}

    for idx, doc in enumerate(tqdm(docs)):
        nodes = node_parser.get_nodes_from_documents([doc])

        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)

        async def run_async_code_a3623563():
            async def run_async_code_24569e98():
                agent, summary = await build_agent_per_doc(nodes, file_base)
                return agent, summary
            agent, summary = asyncio.run(run_async_code_24569e98())
            logger.success(format_json(agent, summary))
            return agent, summary
        agent, summary = asyncio.run(run_async_code_a3623563())
        logger.success(format_json(agent, summary))

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict


async def run_async_code_a78a7590():
    async def run_async_code_7cb7b201():
        agents_dict, extra_info_dict = await build_agents(docs)
        return agents_dict, extra_info_dict
    agents_dict, extra_info_dict = asyncio.run(run_async_code_7cb7b201())
    logger.success(format_json(agents_dict, extra_info_dict))
    return agents_dict, extra_info_dict
agents_dict, extra_info_dict = asyncio.run(run_async_code_a78a7590())
logger.success(format_json(agents_dict, extra_info_dict))

"""
### Build Retriever-Enabled MLX Agent

We build a top-level agent that can orchestrate across the different document agents to answer any user query.

This agent will use a tool retriever to retrieve the most relevant tools for the query.

**Improvements from V0**: We make the following improvements compared to the "base" version in V0.

- Adding in reranking: we use Cohere reranker to better filter the candidate set of documents.
- Adding in a query planning tool: we add an explicit query planning tool that's dynamically created based on the set of retrieved tools.
"""
logger.info("### Build Retriever-Enabled MLX Agent")


def get_agent_tool_callable(agent: FunctionAgent) -> Callable:
    async def query_agent(query: str) -> str:
        async def run_async_code_3e3a5871():
            async def run_async_code_7733d1e5():
                response = await agent.run(query)
                return response
            response = asyncio.run(run_async_code_7733d1e5())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_3e3a5871())
        logger.success(format_json(response))
        return str(response)

    return query_agent


all_tools = []
for file_base, agent in agents_dict.items():
    summary = extra_info_dict[file_base]["summary"]
    async_fn = get_agent_tool_callable(agent)
    doc_tool = FunctionTool.from_defaults(
        async_fn,
        name=f"tool_{file_base}",
        description=summary,
    )
    all_tools.append(doc_tool)

logger.debug(all_tools[0].metadata)


llm = MLXLlamaIndexLLMAdapter(model_name="qwen3-1.7b-4bit")

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
vector_node_retriever = obj_index.as_node_retriever(
    similarity_top_k=10,
)


class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or MLXLlamaIndexLLMAdapter(
            model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, query_bundle):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)

        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_agent = FunctionAgent(
            name="compare_tool",
            description=f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
""",
            tools=tools,
            llm=self._llm,
            system_prompt="""You are an expert at comparing documents. Given a query, use the tools provided to compare the documents and return a summary of the results.""",
        )

        async def query_sub_agent(query: str) -> str:
            async def run_async_code_e34a6f1c():
                async def run_async_code_fefe4604():
                    response = await sub_agent.run(query)
                    return response
                response = asyncio.run(run_async_code_fefe4604())
                logger.success(format_json(response))
                return response
            response = asyncio.run(run_async_code_e34a6f1c())
            logger.success(format_json(response))
            return str(response)

        sub_question_tool = FunctionTool.from_defaults(
            query_sub_agent,
            name=sub_agent.name,
            description=sub_agent.description,
        )
        return tools + [sub_question_tool]


custom_obj_retriever = CustomObjectRetriever(
    vector_node_retriever,
    obj_index.object_node_mapping,
    node_postprocessors=[CohereRerank(top_n=5, model="rerank-v3.5")],
    llm=llm,
)

tmps = custom_obj_retriever.retrieve("hello")

logger.debug(len(tmps))


top_agent = FunctionAgent(
    tool_retriever=custom_obj_retriever,
    system_prompt=""" \
You are an agent designed to answer queries about the documentation.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    llm=llm,
)

"""
### Define Baseline Vector Store Index

As a point of comparison, we define a "naive" RAG pipeline which dumps all docs into a single vector index collection.

We set the top_k = 4
"""
logger.info("### Define Baseline Vector Store Index")

all_nodes = [
    n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]
]

base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

"""
## Running Example Queries

Let's run some example queries, ranging from QA / summaries over a single document to QA / summarization over multiple documents.
"""
logger.info("## Running Example Queries")


handler = top_agent.run(
    "What can you build with LlamaIndex?",
)
async for ev in handler.stream_events():
    if isinstance(ev, ToolCallResult):
        logger.debug(
            f"\nCalling tool {ev.tool_name} with args {ev.tool_kwargs}\n Got response: {str(ev.tool_output)[:200]}"
        )
    elif isinstance(ev, ToolCall):
        logger.debug(f"\nTool call: {ev.tool_name} with args {ev.tool_kwargs}")


async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.debug(str(response))


response = base_query_engine.query(
    "What can you build with LlamaIndex?",
)
logger.debug(str(response))


async def run_async_code_96488610():
    async def run_async_code_6f32d7af():
        response = await top_agent.run("Compare workflows to query engines")
        return response
    response = asyncio.run(run_async_code_6f32d7af())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_96488610())
logger.success(format_json(response))
logger.debug(str(response))


async def async_func_31():
    response = await top_agent.run(
        "Can you compare the compact and tree_summarize response synthesizer response modes at a very high-level?"
    )
    return response
response = asyncio.run(async_func_31())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
