from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, StateGraph, START
from pprint import pprint
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

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Self RAG

Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. 

[Paper](https://arxiv.org/abs/2310.11511)

![Screenshot 2024-04-01 at 12.41.50 PM.png](attachment:15cba0ab-a549-4909-8373-fb761e384eff.png)

# Environment
"""
logger.info("# Self RAG")

# %pip install -qU langchain-pinecone langchain-ollama langchainhub langgraph

"""
### Tracing

Use [LangSmith](https://docs.smith.langchain.com/) for tracing (shown at bottom)
"""
logger.info("### Tracing")


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"


os.environ["LANGCHAIN_PROJECT"] = "pinecone-devconnect"

"""
## Retriever
 
Let's use Pinecone's sample movies database
"""
logger.info("## Retriever")


# use pinecone movies database

# Add to vectorDB
vectorstore = PineconeVectorStore(
    embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    index_name="sample-movies",
    text_key="summary",
)
retriever = vectorstore.as_retriever()

docs = retriever.invoke("James Cameron")
for doc in docs:
    logger.debug("# " + doc.metadata["title"])
    logger.debug(doc.page_content)
    logger.debug()

"""
## Structured Output - Retrieval Grader
"""
logger.info("## Structured Output - Retrieval Grader")



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


grade_prompt = hub.pull("efriis/self-rag-retrieval-grader")

llm = ChatOllama(model="llama3.2")
structured_llm_grader = llm.with_structured_output(GradeDocuments)

retrieval_grader = grade_prompt | structured_llm_grader


"""
# Generation Step

Standard RAG
"""
logger.info("# Generation Step")



class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


llm = ChatOllama(model="llama3.2")
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

hallucination_prompt = hub.pull("efriis/self-rag-hallucination-grader")

hallucination_grader = hallucination_prompt | structured_llm_grader
logger.debug(generation)
hallucination_grader.invoke({"documents": docs, "generation": generation})

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOllama(model="llama3.2")
structured_llm_grader = llm.with_structured_output(GradeAnswer)

answer_prompt = hub.pull("efriis/self-rag-answer-grader")

answer_grader = answer_prompt | structured_llm_grader
logger.debug(question)
logger.debug(generation)
answer_grader.invoke({"question": question, "generation": generation})


"""
# Graph 

Capture the flow in as a graph.

## Graph state
"""
logger.info("# Graph")




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]




"""
## Build Graph

The just follows the flow we outlined in the figure above.
"""
logger.info("## Build Graph")


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()


# Run
inputs = {"question": "Movies that star Daniel Craig"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        plogger.debug(f"Node '{key}':")
    plogger.debug("\n---\n")

# Final generation
plogger.debug(value["generation"])

inputs = {"question": "Which movies are about aliens?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        plogger.debug(f"Node '{key}':")
    plogger.debug("\n---\n")

# Final generation
plogger.debug(value["generation"])

logger.info("\n\n[DONE]", bright=True)