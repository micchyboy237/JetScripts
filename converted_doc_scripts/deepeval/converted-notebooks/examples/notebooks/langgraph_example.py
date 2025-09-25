from deepeval.dataset import EvaluationDataset
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, List, TypedDict, Literal
import os
import random
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
## Evaluating a Health Assistant Agent Built with LangGraph

In this notebook you will learn: 

- Evalauate the agent using [TaskCompletion metric](https://deepeval.com/docs/metrics-task-completion)
- Change the hyperparameter to improve the agent's performance
- Evaluate the agent again
"""
logger.info("## Evaluating a Health Assistant Agent Built with LangGraph")

# !pip install -U langgraph langchain langchain-community langchain-ollama chromadb --quiet

"""
# Export you OPENAI_API_KEY as an environment variable
"""
# logger.info("Export you OPENAI_API_KEY as an environment variable")


# os.environ["OPENAI_API_KEY"] = "<your-api-key>"

"""
### Health assistant agent built with LangGraph

Given a user query, the agent will decide the best way to process the query. Here is the diagram of the agent:
 
<img src="static/output.png" alt="Agent Diagram" height="300" style="display: block; margin: 0 auto;">

We are keeping the model as `llama3.2` for the first iteration. Later in the same notebook we will evaluate the agent with `gpt-4` to see the performance difference.
"""
logger.info("### Health assistant agent built with LangGraph")


llm = ChatOllama(model="llama3.2")

"""
Pull the `manual.txt` which will form knowlege base of the agent
"""
logger.info("Pull the `manual.txt` which will form knowlege base of the agent")

# !curl -o manual.txt "https://confident-bucket.s3.us-east-1.amazonaws.com/manual.txt"



# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class AgentState(TypedDict):
    """State schema for the RAG agent"""

    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    selected_tools: List[str]
    retrieved_context: str
    tool_outputs: List[str]
    next_action: str


def setup_vector_store():
    """Set up vector database with documents from local text file"""
    text_file_path = f"{os.path.dirname(__file__)}/static/manual.txt"

    try:
        loader = TextLoader(text_file_path, encoding="utf-8")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma.from_documents(doc_splits, embeddings, persist_directory=PERSIST_DIR)
        return vector_store.as_retriever()

    except FileNotFoundError as e:
        logger.error(f"File '{text_file_path}' not found. Please ensure the file exists.")
        raise FileNotFoundError(f"File '{text_file_path}' not found.") from e
    except Exception as e:
        logger.error(f"Error setting up vector store: {str(e)}")
        raise RuntimeError(f"Failed to set up vector store: {str(e)}") from e


retriever = setup_vector_store()


@tool
def get_last_day_steps():
    """
    Get the last day's steps from the database
    """
    return random.randint(1000, 5000)


@tool
def get_last_day_average_heart_rate():
    """
    Get the last day's average heart rate from the database
    """
    return random.randint(60, 100)


@tool
def get_last_day_average_sleep_duration_in_hours():
    """
    Get the last day's average sleep duration from the database
    """
    return random.randint(3, 10)


tools = [
    get_last_day_steps,
    get_last_day_average_heart_rate,
    get_last_day_average_sleep_duration_in_hours,
]
tool_registry = {tool.name: tool for tool in tools}


class RouteQuery(BaseModel):
    """Schema for routing decisions"""

    reasoning: str = Field(description="Reasoning for the routing decision")
    route: Literal["retrieval", "tools", "direct"] = Field(
        description="Where to route the query"
    )
    tools_needed: List[str] = Field(
        description="List of tools needed if route is 'tools'"
    )
    retrieval_query: str = Field(
        description="Optimized query for retrieval if route is 'retrieval'"
    )


def router_node(state: AgentState) -> AgentState:
    """Route the query to appropriate processing path"""

    system_prompt = """You are an intelligent router that decides how to process user queries.

    Available options:
    - 'retrieval': Query needs information from the knowledge base
    - 'tools': Query needs external tools (web search, calculations, etc.)
    - 'direct': Query can be answered directly with general knowledge

    Available tools: {tools}

    Analyze the user query and decide the best routing approach. If tools are needed,
    specify which ones. If retrieval is needed, optimize the query for better results."""

    user_query = state["messages"][-1].content

    structured_llm = llm.with_structured_output(
        RouteQuery, method="function_calling"
    )

    response = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": system_prompt.format(tools=[t.name for t in tools]),
            },
            {"role": "user", "content": user_query},
        ]
    )

    return {
        "query": user_query,
        "next_action": response.route,
        "selected_tools": response.tools_needed,
        "retrieved_context": (
            response.retrieval_query if response.route == "retrieval" else ""
        ),
    }


def tool_execution_node(state: AgentState) -> AgentState:
    """Execute selected tools"""
    tool_outputs = []
    for tool_name in state["selected_tools"]:
        if tool_name in tool_registry:
            tool = tool_registry[tool_name]
            try:
                output = tool.invoke({"query": state["query"]})
                tool_outputs.append(f"{tool_name}: {output}")
            except Exception as e:
                tool_outputs.append(f"{tool_name}: Error - {str(e)}")

    return {"tool_outputs": tool_outputs}


def retrieval_node(state: AgentState) -> AgentState:
    """Execute retrieval from vector database"""
    query = state["retrieved_context"] or state["query"]

    if retriever is None:
        logger.error("Retriever is not initialized. Cannot perform retrieval.")
        return {"retrieved_context": "Error: Knowledge base retriever not initialized."}

    try:
        docs = retriever.invoke(query)
        if not docs:
            logger.warning(f"No documents retrieved for query: {query}")
            return {"retrieved_context": "No relevant information found in the knowledge base."}
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        return {"retrieved_context": f"Retrieval error: {str(e)}"}


def response_synthesis_node(state: AgentState) -> AgentState:
    """Synthesize final response from all available information"""

    context_parts = []

    if state.get("retrieved_context"):
        context_parts.append(
            f"Knowledge Base Context:\n{state['retrieved_context']}"
        )

    if state.get("tool_outputs"):
        tool_context = "\n".join(state["tool_outputs"])
        context_parts.append(f"Tool Outputs:\n{tool_context}")

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant that synthesizes information from multiple sources.

    Use the provided context to answer the user's question accurately and comprehensively.
    If using information from the context, be sure to reference it appropriately.
    If the context doesn't contain enough information, acknowledge this limitation."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {state['query']}\n\nContext:\n{context}",
        },
    ]

    response = llm.invoke(messages)

    return {"messages": [AIMessage(content=response.content)]}


def intial_route_decision(state: AgentState) -> str:
    """Determine next node based on routing decision"""
    next_action = state.get("next_action", "direct")

    if next_action == "tools":
        return "tools"

    if next_action == "retrieval":
        return "retrieval"

    return "retrieval"


def create_rag_graph():
    """Create and compile the RAG workflow graph"""

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_node("synthesis", response_synthesis_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        intial_route_decision,
        {
            "retrieval": "retrieval",
            "tools": "tools",
            "synthesis": "synthesis",
        },
    )

    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("tools", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


app = create_rag_graph()

"""
Now we have the graph, we can run the agent with the following code:
"""
logger.info("Now we have the graph, we can run the agent with the following code:")

initial_state = {
    "query": "",
    "selected_tools": [],
    "retrieved_context": "",
    "tool_outputs": [],
    "next_action": "",
}


def run_rag_query(query: str):
    """Run a query through the RAG system"""

    initial_state["messages"] = [HumanMessage(content=query)]
    result = app.invoke(initial_state)
    final_message = result["messages"][-1]
    return final_message.content


run_rag_query("What is the average heart rate of the user?")

"""
### Evaluate the agent

[DeepEval](https://deepeval.com/) provides a `CallbackHandler` for LangGraph and LangChain agents to evaluate (and trace) the agents. 


> (Pro Tip) View your Agent's trace and publish test runs on [Confident AI](https://www.confident-ai.com/). Apart from this you get an in-house dataset editor and more advaced tools to monitor and enventually improve your Agent's performance. Get your API key from [here](https://app.confident-ai.com/)

OPTIONAL: Set CONFIDENT_API_KEY as an environment variable to publish test results on Confident AI.
"""
logger.info("### Evaluate the agent")

# !export CONFIDENT_API_KEY=your-api-key

"""
Initialize the CallbackHandler and pass TaskCompletionMetric to it.
"""
logger.info("Initialize the CallbackHandler and pass TaskCompletionMetric to it.")



def run_rag_query(query: str):
    """Run a query through the RAG system"""

    initial_state["messages"] = [HumanMessage(content=query)]

    result = app.invoke(
        initial_state,
        config={
            "callbacks": [
                CallbackHandler(
                    metrics=[
                        TaskCompletionMetric(strict_mode=True, async_mode=False)
                    ]
                )
            ]  # pass the metrics to the callback handler
        },
    )

    final_message = result["messages"][-1]
    return final_message.content

"""
### Pull the dataset
For tutorial purposes, we will use the public dataset of health queries. You can use your own dataset as well. Refer to the [docs](https://deepeval.com/docs/evaluation-end-to-end-llm-evals#setup-your-test-environment) to learn more about how to create your own dataset.
"""
logger.info("### Pull the dataset")


dataset = EvaluationDataset()
dataset.pull(alias="health_rag_queries", public=True)

"""
Run evals using dataset iterator
"""
logger.info("Run evals using dataset iterator")

for golden in dataset.evals_iterator():
    run_rag_query(golden.input)

"""
### Change the model to gpt-4 and evaluate again

Now we will change the model to `gpt-4`, redefine the nodes and evaluate the agent again.
"""
logger.info("### Change the model to gpt-4 and evaluate again")

llm = ChatOllama(model="llama3.2")
app = create_rag_graph()


def run_rag_query(query: str):

    initial_state["messages"] = [HumanMessage(content=query)]
    result = app.invoke(
        initial_state,
        config={
            "callbacks": [
                CallbackHandler(
                    metrics=[
                        TaskCompletionMetric(strict_mode=True, async_mode=False)
                    ]
                )
            ]
        },
    )
    final_message = result["messages"][-1]
    return final_message.content


for golden in dataset.evals_iterator():
    run_rag_query(golden.input)

"""
Try changing other hyperparameters of the model and evaluate the agent again.
"""
logger.info("Try changing other hyperparameters of the model and evaluate the agent again.")

logger.info("\n\n[DONE]", bright=True)