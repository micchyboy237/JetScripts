import asyncio
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Callable, Tuple
from tqdm import tqdm
from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, ObjectRetriever
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import UnstructuredReader
from jet.models.embeddings.adapters.rerank_cross_encoder_llama_index_adapter import CrossEncoderRerank
from jet.logger import CustomLogger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)
Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
logger.info("Initialized embedding model and LLM")


async def load_documents(doc_limit: int = 100) -> List[Document]:
    """Load and process markdown documents from the resume data directory."""
    reader = UnstructuredReader()
    data_dir = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data")
    all_files_gen = data_dir.rglob("*.md")
    all_files = [f.resolve() for f in all_files_gen]
    logger.debug(f"Found {len(all_files)} markdown files in {data_dir}")

    docs = []
    for idx, f in enumerate(all_files):
        if idx >= doc_limit:
            break
        logger.debug(f"Processing file {idx + 1}/{len(all_files)}: {f}")
        try:
            loaded_docs = reader.load_data(file=f, split_documents=True)
            loaded_doc = Document(
                text="\n\n".join([d.get_content() for d in loaded_docs]),
                metadata={"path": str(f)},
            )
            docs.append(loaded_doc)
            logger.debug(f"Loaded document: {loaded_doc.metadata['path']}")
        except Exception as e:
            logger.error(f"Failed to load document {f}: {str(e)}")
    logger.debug(f"Total documents loaded: {len(docs)}")
    return docs


async def build_agent_per_doc(nodes: List, file_base: str) -> Tuple[FunctionAgent, str]:
    """Build a document agent for a single document with vector and summary query engines."""
    vi_out_path = f"./data/resume_docs/{file_base}"
    summary_out_path = f"./data/resume_docs/{file_base}_summary.pkl"
    try:
        if not os.path.exists(vi_out_path):
            os.makedirs(os.path.dirname(vi_out_path), exist_ok=True)
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=vi_out_path)
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=vi_out_path))
    except Exception as e:
        logger.error(
            f"Failed to load or create vector index for {file_base}: {str(e)}")
        raise
    summary_index = SummaryIndex(nodes)
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=Settings.llm)
    try:
        if not os.path.exists(summary_out_path):
            os.makedirs(os.path.dirname(summary_out_path), exist_ok=True)
            summary = str(summary_query_engine.query(
                "Extract a concise 1-2 line summary of this document"))
            with open(summary_out_path, "wb") as f:
                pickle.dump(summary, f)
        else:
            with open(summary_out_path, "rb") as f:
                summary = pickle.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load or generate summary for {file_base}: {str(e)}")
        raise
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name=f"vector_tool_{file_base}",
            description="Useful for questions related to specific facts",
        ),
        QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            name=f"summary_tool_{file_base}",
            description="Useful for summarization questions",
        ),
    ]
    function_llm = OllamaFunctionCallingAdapter(model="llama3.2")
    agent = FunctionAgent(
        tools=query_engine_tools,
        llm=function_llm,
        system_prompt=f"""You are a specialized agent designed to answer queries about the `{file_base}.md` part of the resume data.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.""",
    )
    return agent, summary


async def build_agents(docs: List[Document]) -> Tuple[Dict[str, FunctionAgent], Dict[str, Dict]]:
    """Build document agents for all documents."""
    node_parser = SentenceSplitter()
    agents_dict = {}
    extra_info_dict = {}
    for idx, doc in enumerate(tqdm(docs, desc="Building agents")):
        nodes = node_parser.get_nodes_from_documents([doc])
        file_path = Path(doc.metadata["path"])
        file_base = f"{file_path.parent.stem}_{file_path.stem}"
        try:
            agent, summary = await build_agent_per_doc(nodes, file_base)
            agents_dict[file_base] = agent
            extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}
            logger.success(
                f"Built agent for {file_base}: {format_json({'agent': str(agent), 'summary': summary})}")
        except Exception as e:
            logger.error(f"Failed to build agent for {file_base}: {str(e)}")
    return agents_dict, extra_info_dict


def get_agent_tool_callable(agent: FunctionAgent) -> Callable:
    """Create a callable for querying a document agent."""
    async def query_agent(query: str) -> str:
        try:
            response = await agent.run(query)
            logger.success(
                f"Agent query response: {format_json(str(response))}")
            return str(response)
        except Exception as e:
            logger.error(f"Agent query failed: {str(e)}")
            return f"Error: {str(e)}"
    return query_agent


class CustomObjectRetriever(ObjectRetriever):
    """Custom retriever for selecting tools with reranking."""

    def __init__(self, retriever, object_node_mapping, node_postprocessors=None, llm=None):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or OllamaFunctionCallingAdapter(model="llama3.2")
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, query_bundle: str | QueryBundle) -> List:
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)
        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=query_bundle)
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]
        sub_agent = FunctionAgent(
            name="compare_tool",
            description="""Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this tool with the original query. Do NOT use the other tools for any queries involving multiple documents.""",
            tools=tools,
            llm=self._llm,
            system_prompt="You are an expert at comparing documents. Given a query, use the tools provided to compare the documents and return a summary of the results.",
        )

        async def query_sub_agent(query: str) -> str:
            try:
                response = await sub_agent.run(query)
                logger.success(
                    f"Sub-agent query response: {format_json(str(response))}")
                return str(response)
            except Exception as e:
                logger.error(f"Sub-agent query failed: {str(e)}")
                return f"Error: {str(e)}"
        sub_question_tool = FunctionTool.from_defaults(
            query_sub_agent,
            name=sub_agent.name,
            description=sub_agent.description,
        )
        return tools + [sub_question_tool]


async def main():
    """Main function to orchestrate multi-document agent setup and queries."""
    logger.info("Starting multi-document agent setup")
    docs = await load_documents(doc_limit=100)
    logger.info("Documents loaded successfully")
    agents_dict, extra_info_dict = await build_agents(docs)
    logger.info("Agents built successfully")
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
    logger.debug(
        f"First tool metadata: {all_tools[0].metadata if all_tools else 'No tools'}")
    obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
    vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)
    custom_obj_retriever = CustomObjectRetriever(
        vector_node_retriever,
        obj_index.object_node_mapping,
        node_postprocessors=[CrossEncoderRerank(
            top_n=5, model="ms-marco-MiniLM-L12-v2")],
        llm=Settings.llm,
    )
    top_agent = FunctionAgent(
        tool_retriever=custom_obj_retriever,
        system_prompt="""You are an agent designed to answer queries about the resume data.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.""",
        llm=Settings.llm,
    )
    logger.info("Top agent initialized")
    all_nodes = [n for extra_info in extra_info_dict.values()
                 for n in extra_info["nodes"]]
    base_index = VectorStoreIndex(all_nodes)
    base_query_engine = base_index.as_query_engine(similarity_top_k=4)
    logger.info("Base query engine initialized")
    handler = top_agent.run("What skills are listed in the resume?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            logger.debug(
                f"Calling tool {ev.tool_name} with args {ev.tool_kwargs}\nGot response: {str(ev.tool_output)[:200]}")
        elif isinstance(ev, ToolCall):
            logger.debug(
                f"Tool call: {ev.tool_name} with args {ev.tool_kwargs}")
    response = await handler
    logger.success(f"Query response: {format_json(str(response))}")
    logger.debug(f"Response: {str(response)}")
    response = base_query_engine.query("What skills are listed in the resume?")
    logger.debug(f"Baseline query response: {str(response)}")
    response = await top_agent.run("Compare the mobile and web app projects")
    logger.success(f"Comparison query response: {format_json(str(response))}")
    logger.debug(f"Response: {str(response)}")
    response = await top_agent.run(
        "Can you compare the skills and work history at a very high-level?"
    )
    logger.success(
        f"Skills and work history comparison response: {format_json(str(response))}")
    logger.debug(f"Response: {str(response)}")
    logger.info("\n\n[DONE]", bright=True)

if __name__ == "__main__":
    asyncio.run(main())
