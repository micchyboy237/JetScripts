from typing import List, TypedDict
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    ServiceContext,
    download_loader,
    SimpleDirectoryReader,
)
from llama_index.core.query_engine.transform_query_engine import (
    TransformQueryEngine,
)
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, QueryBundle, BaseNode, NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core.node_parser import NodeParser, SentenceSplitter

from llama_index.core.response.notebook_utils import display_source_node

import os
import json
from jet.logger import logger
from jet.file import save_json
from jet.transformers import make_serializable

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


class SettingsDict(TypedDict):
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


class SettingsManager:
    @staticmethod
    def create(settings: SettingsDict):
        Settings.chunk_size = settings["chunk_size"]
        Settings.chunk_overlap = settings["chunk_overlap"]
        Settings.embed_model = OllamaEmbedding(
            model_name=settings["embedding_model"],
            base_url=settings["base_url"],
        )
        Settings.llm = Ollama(
            temperature=0,
            request_timeout=120.0,
            model=settings["llm_model"],
            base_url=settings["base_url"],
        )
        return Settings


class IndexManager:
    @staticmethod
    def create_nodes(documents: List[Document], parser: NodeParser):
        nodes = parser.get_nodes_from_documents(documents, show_progress=True)
        return nodes

    @staticmethod
    def create_index(embed_model: OllamaEmbedding, nodes: list[BaseNode]) -> VectorStoreIndex:
        # build index
        return VectorStoreIndex(
            nodes=nodes, embed_model=embed_model, show_progress=True)

    @staticmethod
    def create_retriever(index: VectorStoreIndex, similarity_top_k: int):
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever

    @staticmethod
    def create_query_engine(index: VectorStoreIndex, similarity_top_k: int):
        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
        return query_engine


# Download the documents from Wikipedia and load them
WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
pages = ['Emma_Stone', 'La_La_Land', 'Ryan_Gosling']
documents = loader.load_data(pages=pages, auto_suggest=False, redirect=False)

logger.log("Documents:", len(documents), colors=["GRAY", "DEBUG"])
save_json(documents, file_path="generated/wiki/documents.json")


# Example 1
# # Transform chunks into numerical vectors using the embedding model
# service_context_gpt3 = ServiceContext.from_defaults(
#     llm=gpt3, chunk_size=256, chunk_overlap=0, embed_model=embed_model)
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context_gpt3)

settings = SettingsDict(
    llm_model="llama3.1",
    embedding_model="nomic-embed-text",
    chunk_size=256,
    chunk_overlap=0,
    base_url="http://localhost:11434",
)

results = {
    "settings": settings,
    "initial_prompts": [],
    "initial_queries": [],
    "questions": [],
    "chats": [],
    "rerank": {},
    "hyde": {},
}

settings_manager = SettingsManager.create(settings)
# Merge settings
logger.log("Settings:", json.dumps(settings), colors=["GRAY", "DEBUG"])
save_json(results, file_path="generated/wiki/results.json")


# Initialize the gpt3.5 model
# gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct",
#               api_key=OPENAI_API_KEY)
gpt3 = settings_manager.llm
# Initialize the embedding model
# embed_model = OpenAIEmbedding(
#     model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002, api_key=OPENAI_API_KEY)
embed_model = settings_manager.embed_model

# Create nodes
logger.debug("Creating nodes...")
nodes = IndexManager.create_nodes(
    documents=documents, parser=settings_manager.node_parser)

# Create index
logger.debug("Creating index...")
index = IndexManager.create_index(
    embed_model=settings_manager.embed_model,
    nodes=nodes,
)

# Create retriever
logger.debug("Creating retriever...")
retriever = IndexManager.create_retriever(
    index, similarity_top_k=3)

# Build a prompt template to only provide answers based on the loaded documents
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "Don't give an answer unless it is supported by the context above.\n"
)

qa_template = PromptTemplate(template)
# Create a prompt for the model
question = "What is the plot of the film that led Emma Stone to win her first Academy Award?"

# Retrieve the context from the model
contexts: list[NodeWithScore] = retriever.retrieve(question)
context_list = [n.get_content() for n in contexts]
prompt = qa_template.format(
    context_str="\n\n".join(context_list), query_str=question)

# Generate the response
response = gpt3.complete(prompt)
results["initial_prompts"].append({
    "prompt": prompt,
    "response": response,
})
save_json(results, file_path="generated/wiki/results.json")

# Create a prompt for the model
question = "Compare the families of Emma Stone and Ryan Gosling"
# Retrieve the context from the model
contexts: list[NodeWithScore] = retriever.retrieve(question)
context_list = [n.get_content() for n in contexts]
prompt = qa_template.format(
    context_str="\n\n".join(context_list), query_str=question)
# Generate the response
response = gpt3.complete(prompt)
results["initial_prompts"].append({
    "prompt": prompt,
    "response": response,
})
save_json(results, file_path="generated/wiki/results.json")


# Example 2
# # modify default values of chunk size and chunk overlap
# service_context_gpt3 = ServiceContext.from_defaults(
#     llm=gpt3, chunk_size=512, chunk_overlap=50, embed_model=embed_model)

# # build index
# index = VectorStoreIndex.from_documents(
#     documents, service_context=service_context_gpt3

# )
# # returns the engine for the index
# query_engine = index.as_query_engine(similarity_top_k=4)

settings = SettingsDict(
    llm_model="llama3.1",
    embedding_model="nomic-embed-text",
    chunk_size=512,
    chunk_overlap=50,
    base_url="http://localhost:11434",
)
settings_manager = SettingsManager.create(settings)
# Initialize the gpt3.5 model
# gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct",
#               api_key=OPENAI_API_KEY)
gpt3 = settings_manager.llm
# Initialize the embedding model
# embed_model = OpenAIEmbedding(
#     model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002, api_key=OPENAI_API_KEY)
embed_model = settings_manager.embed_model

# Create nodes
logger.debug("Creating nodes...")
nodes = IndexManager.create_nodes(
    documents=documents, parser=settings_manager.node_parser)

# Create index
logger.debug("Creating index...")
index = IndexManager.create_index(
    embed_model=settings_manager.embed_model,
    nodes=nodes,
)

# Create query engine
logger.debug("Creating query engine...")
query_engine = IndexManager.create_query_engine(
    index=index, similarity_top_k=3)

# generate the response
query = "What is the plot of the film that led Emma Stone to win her first Academy Award?"
response = query_engine.query()
results["initial_queries"].append({
    "query": query,
    "response": response,
})
save_json(results, file_path="generated/wiki/results.json")

# generate the response
query = "Compare the families of Emma Stone and Ryan Gosling"
response = query_engine.query(query)
results["initial_queries"].append({
    "query": query,
    "response": response,
})
save_json(results, file_path="generated/wiki/results.json")

# Example 3
# Retrieve the top three chunks for the second query
retriever = index.as_retriever(similarity_top_k=3)
query = "Compare the families of Emma Stone and Ryan Gosling"
nodes = retriever.retrieve(query)
# Print the chunks
for node in nodes:
    print('----------------------------------------------------')
    display_source_node(node, source_length=500)
results["retrieved_nodes"].append({
    "query": query,
    "response": nodes,
})
save_json(results, file_path="generated/wiki/results.json")


# Example 4
hf_token = os.getenv("HF_TOKEN")
logger.log("HF_TOKEN:", hf_token, colors=["GRAY", "INFO"])
# HF_TOKEN = userdata.get('HF_TOKEN')
# os.environ['HF_TOKEN'] = HF_TOKEN

# Re-Rank chunks based on the bge-reranker-base-model
reranker = FlagEmbeddingReranker(
    top_n=3,
    model="BAAI/bge-reranker-base",
)
# Return the updated chunks
query_bundle = QueryBundle(query_str=query)
ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle=query_bundle)
for ranked_node in ranked_nodes:
    print('----------------------------------------------------')
    display_source_node(ranked_node, source_length=500)
# Initialize the query engine with Re-Ranking
query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[reranker]
)

# Print the response from the model
response = query_engine.query(
    "Compare the families of Emma Stone and Ryan Gosling")
results["rerank"] = {
    "flag": {
        "query": query,
        "response": response,
    }
}
save_json(results, file_path="generated/wiki/results.json")


# Example 5
# Re-Rank the top 3 chunks based on the gpt-3.5-turbo-0125 model
reranker = RankGPTRerank(
    top_n=3,
    llm=settings_manager.llm,
)
# Display the top 3 chunks based on RankGPT
query_bundle = QueryBundle(query_str=query)
ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle=query_bundle)

for ranked_node in ranked_nodes:
    print('----------------------------------------------------')
    display_source_node(ranked_node, source_length=500)
# Print the response from the model
response = query_engine.query(query)
results["rerank"] = {
    "llm": {
        "query": query,
        "response": response,
    }
}
save_json(results, file_path="generated/wiki/results.json")


# Example 6
# build index and query engine for the index
query_engine = IndexManager.create_query_engine(
    index=index, similarity_top_k=4)

# HyDE setup
hyde = HyDEQueryTransform(include_original=True)
# Transform the query engine using HyDE
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
response = hyde_query_engine.query(
    "Compare the families of Emma Stone and Ryan Gosling")
results["hyde"] = {"query": query, **make_serializable(response)}
save_json(results, file_path="generated/wiki/results.json")

# Multi-step query setup
step_decompose_transform_gpt3 = StepDecomposeQueryTransform(gpt3, verbose=True)
index_summary = "Breaks down the initial query"
# Return query engine for the index
multi_step_query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary
)
response = multi_step_query_engine.query(
    "Compare the families of Emma Stone and Ryan Gosling")
results["multi_step"] = {
    "query": query,
    "response": response,
}
save_json(results, file_path="generated/wiki/results.json")
