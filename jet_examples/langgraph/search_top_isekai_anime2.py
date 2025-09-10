from jet.file.utils import save_file
from jet.logger import logger
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing import List, TypedDict
import os
import shutil

# Setup directories and logging
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


# Define structured output for anime list

class Anime(BaseModel):
    title: str = Field(description="The title of the isekai anime")
    description: str = Field(description="A brief description of the anime")


class AnimeList(BaseModel):
    animes: List[Anime] = Field(description="List of top 10 isekai anime")


# Define graph state

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    anime_list: AnimeList


# Initialize tools and LLM
llm = ChatOllama(model="llama3.2")
# Increased to 5 for broader results
web_search_tool = TavilySearchResults(k=5)

# Question router to always use web search for isekai anime
prompt = PromptTemplate(
    template="""Always route questions about isekai anime to web search. Return JSON with a single key 'datasource' set to 'web_search'.""",
    input_variables=["question"],
)
question_router = prompt | llm | JsonOutputParser()

# Web search node


def web_search(state: GraphState):
    logger.debug("---WEB SEARCH---")
    question = state["question"]
    try:
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        return {"documents": [web_results], "question": question}
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return {"documents": [], "question": question}


# Generate structured anime list
prompt = PromptTemplate(
    template="""You are an expert in anime recommendations. Based on the provided context, generate a list of the top 10 isekai anime for 2025. Each anime should have a title and a brief description. Return the result as a JSON object with a key 'animes' containing a list of objects with 'title' and 'description' fields.

Context:
{context}

Question: {question}

Output format:
{{
    "animes": [
        {{"title": "Anime Name", "description": "Brief description"}},
        ...
    ]
}}
""",
    input_variables=["context", "question"],
)
rag_chain = prompt | llm | JsonOutputParser()


def generate(state: GraphState):
    logger.debug("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)
    try:
        generation = rag_chain.invoke(
            {"context": context, "question": question})
        anime_list = AnimeList(**generation)
        if len(anime_list.animes) != 10:
            logger.debug("---GENERATION: INCOMPLETE ANIME LIST, RETRY---")
            return {"documents": documents, "question": question, "generation": "", "anime_list": None}
        return {"documents": documents, "question": question, "generation": generation, "anime_list": anime_list}
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return {"documents": documents, "question": question, "generation": "", "anime_list": None}


# Grade generation for completeness and relevance
prompt = PromptTemplate(
    template="""You are a grader assessing whether a generated list of top 10 isekai anime is complete and relevant to the question. The list should contain exactly 10 anime with titles and descriptions relevant to isekai anime for 2025.

Here is the question: {question}
Here is the generated list: {generation}

Return a JSON with a single key 'score' and value 'yes' if the list has exactly 10 relevant isekai anime, otherwise 'no'.""",
    input_variables=["question", "generation"],
)
answer_grader = prompt | llm | JsonOutputParser()


def grade_generation(state: GraphState):
    logger.debug("---GRADE GENERATION---")
    question = state["question"]
    generation = state["generation"]
    score = answer_grader.invoke(
        {"question": question, "generation": generation})
    grade = score["score"]
    if grade == "yes":
        logger.debug("---DECISION: GENERATION IS COMPLETE AND RELEVANT---")
        return "useful"
    logger.debug(
        "---DECISION: GENERATION IS INCOMPLETE OR IRRELEVANT, RETRY---")
    return "not useful"


# Transform query if generation fails
re_write_prompt = PromptTemplate(
    template="""You are a question re-writer optimizing for web search. Convert the input question to a more specific version for retrieving top isekai anime for 2025.

Here is the initial question: {question}
Improved question with no preamble: Top 10 isekai anime for 2025 with titles and descriptions""",
    input_variables=["question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


def transform_query(state: GraphState):
    logger.debug("---TRANSFORM QUERY---")
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}


# Route question to web search (always for isekai queries)
def route_question(state: GraphState):
    """
    Route question to web search or RAG.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call (always 'web_search' for this use case)
    """
    logger.debug("---ROUTE QUESTION---")
    question = state["question"]
    logger.debug(question)
    source = question_router.invoke({"question": question})
    logger.debug(source)
    if source["datasource"] == "web_search":
        logger.debug("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    else:
        logger.debug("---ROUTE QUESTION TO VECTORSTORE (FALLBACK)---")
        return "web_search"  # Always web_search for isekai, but fallback for flexibility


# Define workflow
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "useful": END,
        "not useful": "transform_query",
    },
)
workflow.add_edge("transform_query", "web_search")
app = workflow.compile(checkpointer=MemorySaver())
render_mermaid_graph(app, f"{OUTPUT_DIR}/graph_output.png", xray=True)

# Execute query
inputs = {"question": "What are the top 10 isekai anime for 2025?"}
for output in app.stream(inputs):
    for key, value in output.items():
        logger.debug(f"Node '{key}':")
        logger.debug(value)
if value.get("anime_list"):
    logger.info("Top 10 Isekai Anime for 2025:")
    for anime in value["anime_list"].animes:
        logger.info(f"- {anime.title}: {anime.description}")
save_file(app, f"{OUTPUT_DIR}/workflow_state.json")
logger.info("\n\n[DONE]", bright=True)
