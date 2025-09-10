from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.tavily_search_tool import TavilySearchResults
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.object import make_serializable
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


# Define structured output model for anime list
class AnimeList(BaseModel):
    titles: List[str] = Field(
        description="List of top 10 isekai anime titles for 2025")


# Define graph state
class GraphState(TypedDict):
    question: str
    generation: List[str]
    documents: List[str]
    search_query: str  # Added to store the search query
    search_results: List[dict]  # Added to store raw Tavily search results


# Initialize LLM and tools
llm = ChatOllama(model="llama3.2", temperature=0)
web_search_tool = TavilySearchResults(k=5)

# Define prompt for generating the top 10 isekai anime list
prompt = PromptTemplate(
    template="""You are an expert in anime recommendations. Based on the provided web search results, generate a list of the top 10 isekai anime for 2025. \n
    Here are the search results: \n\n {documents} \n\n
    Provide a list of exactly 10 anime titles. If insufficient data is available, infer plausible titles based on trends to ensure exactly 10 titles. \n
    Return a structured response with a single key 'titles' containing the list of titles.""",
    input_variables=["documents"],
)

# Chain for generating the anime list
anime_chain = prompt | llm.with_structured_output(AnimeList)


def web_search(state):
    """
    Perform web search to fetch recent isekai anime data for 2025.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with web search results, query, and raw results
    """
    logger.debug("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {
        "documents": [web_results],
        "question": question,
        "search_query": question,
        "search_results": make_serializable(docs)
    }


def generate(state):
    """
    Generate the top 10 isekai anime list based on web search results.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updated state with generated anime list
    """
    logger.debug("---GENERATE ANIME LIST---")
    question = state["question"]
    documents = state["documents"]
    generation = anime_chain.invoke(
        {"documents": [doc.page_content for doc in documents]})
    return {
        "documents": documents,
        "question": question,
        "search_query": state["search_query"],
        "search_results": state["search_results"],
        "generation": generation.titles  # Extract titles list from AnimeList
    }


# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_edge(START, "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile(checkpointer=MemorySaver())
render_mermaid_graph(app, f"{OUTPUT_DIR}/graph_output.png", xray=True)

# Execute the query
inputs = {"question": "top 10 isekai anime 2025"}
config = {"configurable": {"thread_id": "1"}}  # Add config for checkpointer
state = app.invoke(inputs, config)
save_file(state, f"{OUTPUT_DIR}/workflow_state.json")
save_file(
    {
        "search_query": state["search_query"],
        "search_results": state["search_results"]
    },
    f"{OUTPUT_DIR}/search_data.json"
)
logger.debug(f"Generated anime list: {state['generation']}")
logger.info("\n\n[DONE]", bright=True)
