from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import BaseMessage
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from typing import Annotated, Sequence, TypedDict
import json
import operator
import os
import pprint
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

# ! pip install langchain-chroma langchain_community tiktoken langchain-ollama langchainhub langchain langgraph

"""
# LangGraph Retrieval Agent

We can implement [Retrieval Agents](https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents) in [LangGraph](https://python.langchain.com/docs/langgraph).

## Retriever
"""
logger.info("# LangGraph Retrieval Agent")


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
retriever = vectorstore.as_retriever()


tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

tools = [tool]


tool_executor = ToolExecutor(tools)

"""
📘 **Note on `SystemMessage` usage with LangGraph-based agents**

When constructing the `messages` list for an agent, you *must* manually include any `SystemMessage`s.
Unlike some agent executors in LangChain that set a default, LangGraph requires explicit inclusion.

## Agent state
 
We will defined a graph.

A `state` object that it passes around to each node.

Our state will be a list of `messages`.

Each node in our graph will append to it.
"""
logger.info("## Agent state")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


"""
## Nodes and Edges

Each node will - 

1/ Either be a function or a runnable.

2/ Modify the `state`.

The edges choose which node to call next.

We can lay out an agentic RAG graph like this:

![Screenshot 2024-02-02 at 1.36.50 PM.png](attachment:f886806c-0aec-4c2a-8027-67339530cb60.png)
"""
logger.info("## Nodes and Edges")


def should_retrieve(state):
    """
    Decides whether the agent should retrieve more information or end the process.

    This function checks the last message in the state for a function call. If a function call is
    present, the process continues to retrieve information. Otherwise, it ends the process.

    Args:
        state (messages): The current state of the agent, including all messages.

    Returns:
        str: A decision to either "continue" the retrieval process or "end" it.
    """
    logger.debug("---DECIDE TO RETRIEVE---")
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        logger.debug("---DECISION: DO NOT RETRIEVE / DONE---")
        return "end"
    else:
        logger.debug("---DECISION: RETRIEVE---")
        return "continue"


def check_relevance(state):
    """
    Determines whether the Agent should continue based on the relevance of retrieved documents.

    This function checks if the last message in the conversation is of type FunctionMessage, indicating
    that document retrieval has been performed. It then evaluates the relevance of these documents to the user's
    initial question using a predefined model and output parser. If the documents are relevant, the conversation
    is considered complete. Otherwise, the retrieval process is continued.

    Args:
        state messages: The current state of the conversation, including all messages.

    Returns:
        str: A directive to either "end" the conversation if relevant documents are found, or "continue" the retrieval process.
    """

    logger.debug("---CHECK RELEVANCE---")

    class FunctionOutput(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    parser = PydanticOutputParser(pydantic_object=FunctionOutput)

    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved docs to a user question. \n
        Here are the retrieved docs:
        \n ------- \n
        {context}
        \n ------- \n
        Here is the user question: {question}
        If the docs contain keyword(s) in the user question, then score them as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question. \n
        Output format instructions: \n {format_instructions}""",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )

    model = ChatOllama(model="llama3.2")

    chain = prompt | model | parser

    messages = state["messages"]
    last_message = messages[-1]
    score = chain.invoke(
        {"question": messages[0].content, "context": last_message.content}
    )

    if score.binary_score == "yes":
        logger.debug("---DECISION: DOCS RELEVANT---")
        return "yes"

    else:
        logger.debug("---DECISION: DOCS NOT RELEVANT---")
        logger.debug(score.binary_score)
        return "no"


def call_model(state):
    """
    Invokes the agent model to generate a response based on the current state.

    This function calls the agent model to generate a response to the current conversation state.
    The response is added to the state's messages.

    Args:
        state (messages): The current state of the agent, including all messages.

    Returns:
        dict: The updated state with the new message added to the list of messages.
    """
    logger.debug("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOllama(model="llama3.2")
    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)
    response = model.invoke(messages)
    return {"messages": [response]}


def call_tool(state):
    """
    Executes a tool based on the last message's function call.

    This function is responsible for executing a tool invocation based on the function call
    specified in the last message. The result from the tool execution is added to the conversation
    state as a new message.

    Args:
        state (messages): The current state of the agent, including all messages.

    Returns:
        dict: The updated state with the new function message added to the list of messages.
    """
    logger.debug("---EXECUTE RETRIEVAL---")
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)

    return {"messages": [function_message]}


"""
## Graph

* Start with an agent, `call_model`
* Agent make a decision to call a function
* If so, then `action` to call tool (retriever)
* Then call agent with the tool output added to messages (`state`)
"""
logger.info("## Graph")


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)  # agent
workflow.add_node("action", call_tool)  # retrieval

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_retrieve,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "action",
    check_relevance,
    {
        "yes": "agent",
        "no": END,  # placeholder
    },
)

app = workflow.compile()


inputs = {
    "messages": [
        HumanMessage(
            content="What are the types of agent memory based on Lilian Weng's blog post?"
        )
    ]
}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint.plogger.debug(f"Output from node '{key}':")
        pprint.plogger.debug("---")
        pprint.plogger.debug(value, indent=2, width=80, depth=None)
    pprint.plogger.debug("\n---\n")

"""
Trace:

https://smith.langchain.com/public/6f45c61b-69a0-4b35-bab9-679a8840a2d6/r
"""
logger.info("Trace:")


logger.info("\n\n[DONE]", bright=True)
