from jet.logger import logger
from langchain.agents import AgentExecutor
from langchain.retrievers import CohereRagRetriever
from langchain_cohere import ChatCohere
from langchain_cohere import ChatCohere, create_cohere_react_agent
from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import (
HumanMessage,
ToolMessage,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
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
# Cohere

>[Cohere](https://cohere.ai/about) is a Canadian startup that provides natural language processing models
> that help companies improve human-machine interactions.

## Installation and Setup
- Install the Python SDK :
"""
logger.info("# Cohere")

pip install langchain-cohere

"""
Get a [Cohere api key](https://dashboard.cohere.ai/) and set it as an environment variable (`COHERE_API_KEY`)

## Cohere langchain integrations

|API|description|Endpoint docs|Import|Example usage|
|---|---|---|---|---|
|Chat|Build chat bots|[chat](https://docs.cohere.com/reference/chat)|`from langchain_cohere import ChatCohere`|[cohere.ipynb](/docs/integrations/chat/cohere)|
|LLM|Generate text|[generate](https://docs.cohere.com/reference/generate)|`from langchain_cohere.llms import Cohere`|[cohere.ipynb](/docs/integrations/llms/cohere)|
|RAG Retriever|Connect to external data sources|[chat + rag](https://docs.cohere.com/reference/chat)|`from langchain.retrievers import CohereRagRetriever`|[cohere.ipynb](/docs/integrations/retrievers/cohere)|
|Text Embedding|Embed strings to vectors|[embed](https://docs.cohere.com/reference/embed)|`from langchain_cohere import CohereEmbeddings`|[cohere.ipynb](/docs/integrations/text_embedding/cohere)|
|Rerank Retriever|Rank strings based on relevance|[rerank](https://docs.cohere.com/reference/rerank)|`from langchain.retrievers.document_compressors import CrossEncoderRerank`|[cohere.ipynb](/docs/integrations/retrievers/cohere-reranker)|

## Quick copy examples

### Chat
"""
logger.info("## Cohere langchain integrations")

chat = ChatCohere()
messages = [HumanMessage(content="knock knock")]
logger.debug(chat.invoke(messages))

"""
Usage of the Cohere [chat model](/docs/integrations/chat/cohere)

### LLM
"""
logger.info("### LLM")


llm = Cohere()
logger.debug(llm.invoke("Come up with a pet name"))

"""
Usage of the Cohere (legacy) [LLM model](/docs/integrations/llms/cohere)

### Tool calling
"""
logger.info("### Tool calling")


@tool
def magic_function(number: int) -> int:
    """Applies a magic operation to an integer

    Args:
        number: Number to have magic operation performed on
    """
    return number + 10

def invoke_tools(tool_calls, messages):
    for tool_call in tool_calls:
        selected_tool = {"magic_function":magic_function}[
            tool_call["name"].lower()
        ]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    return messages

tools = [magic_function]

llm = ChatCohere()
llm_with_tools = llm.bind_tools(tools=tools)
messages = [
    HumanMessage(
        content="What is the value of magic_function(2)?"
    )
]

res = llm_with_tools.invoke(messages)
while res.tool_calls:
    messages.append(res)
    messages = invoke_tools(res.tool_calls, messages)
    res = llm_with_tools.invoke(messages)

logger.debug(res.content)

"""
Tool calling with Cohere LLM can be done by binding the necessary tools to the llm as seen above.
An alternative, is to support multi hop tool calling with the ReAct agent as seen below.

### ReAct Agent

The agent is based on the paper
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
"""
logger.info("### ReAct Agent")


llm = ChatCohere()

internet_search = TavilySearchResults(max_results=4)
internet_search.name = "internet_search"
internet_search.description = "Route a user query to the internet"

prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(
    llm,
    [internet_search],
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search], verbose=True)

agent_executor.invoke({
    "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",
})

"""
The ReAct agent can be used to call multiple tools in sequence.

### RAG Retriever
"""
logger.info("### RAG Retriever")


rag = CohereRagRetriever(llm=ChatCohere())
logger.debug(rag.invoke("What is cohere ai?"))

"""
Usage of the Cohere [RAG Retriever](/docs/integrations/retrievers/cohere)

### Text Embedding
"""
logger.info("### Text Embedding")


embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
logger.debug(embeddings.embed_documents(["This is a test document."]))

"""
Usage of the Cohere [Text Embeddings model](/docs/integrations/text_embedding/cohere)

### Reranker

Usage of the Cohere [Reranker](/docs/integrations/retrievers/cohere-reranker)
"""
logger.info("### Reranker")

logger.info("\n\n[DONE]", bright=True)