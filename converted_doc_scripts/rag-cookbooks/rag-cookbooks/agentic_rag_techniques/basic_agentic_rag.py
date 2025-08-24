# from athina_client.datasets import Dataset
# from athina_client.keys import AthinaApiKey
# from google.colab import userdata
from jet.llm.mlx.adapters.mlx_langchain_llm_adapter import ChatMLX
from jet.logger import CustomLogger
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.tools.render import render_text_description_and_args
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# **Basic Agentic RAG**

Agentic Retrieval-Augmented Generation (RAG) is a system that combines generative language models with AI agents to retrieve up-to-date information and generate accurate, context-aware responses. Unlike traditional RAG, which relies only on static databases, agentic RAG uses tools like APIs, search engines, and external datasets. These AI agents can plan tasks, make decisions, and adapt in real time to solve complex problems efficiently.

In this example, we will build a basic RAG system. This system will use two key tools:

- **VectorStore:** To retrieve relevant information from a database of pre-indexed documents.
- **WebSearch:** To fetch up-to-date information from the web when the required data is not available in the VectorStore.

The AI agent will dynamically decide which tool to use based on the query, ensuring accurate and contextually relevant responses. This showcases the flexibility and efficiency of an agentic RAG system.

An interesting read on Agentic RAG: https://arxiv.org/pdf/2501.09136

## **Initial Setup**
"""
logger.info("# **Basic Agentic RAG**")

# !pip install --upgrade --quiet athina-client langchain langchain_community langchain-google-genai pypdf faiss-gpu langchain-huggingface

# os.environ['ATHINA_API_KEY'] = userdata.get('ATHINA_API_KEY')
# os.environ['TAVILY_API_KEY'] = userdata.get('TAVILY_API_KEY')
# os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

"""
## **Indexing**
"""
logger.info("## **Indexing**")

data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/rag-cookbooks/rag-cookbooks/data"
loader = PyPDFLoader(f"{data_dir}/tesla_q3.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})

"""
## **Vector Store**
"""
logger.info("## **Vector Store**")

vectorstore = FAISS.from_documents(documents, embeddings)


retriever = vectorstore.as_retriever()

"""
## **Web Search**
"""
logger.info("## **Web Search**")

web_search_tool = TavilySearchResults(k=10)

"""
## **Agentic RAG**
"""
logger.info("## **Agentic RAG**")

llm = ChatMLX(model="llama-3.2-3b-instruct-4bit")


def vector_search(query: str):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)


def web_search(query: str):
    return web_search_tool.run(query)


@tool
def vector_search_tool(query: str) -> str:
    """Tool for searching the vector store."""
    return vector_search(query)


@tool
def web_search_tool_func(query: str) -> str:
    """Tool for performing web search."""
    return web_search(query)


tools = [
    Tool(
        name="VectorStoreSearch",
        func=vector_search_tool,
        description="Use this to search the vector store for information."
    ),
    Tool(
        name="WebSearch",
        func=web_search_tool_func,
        description="Use this to perform a web search for information."
    ),
]

system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}
Always try the \"VectorStoreSearch\" tool first. Only use \"WebSearch\" if the vector store does not contain the required information.
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:"
```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```
Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""

human_prompt = """{input}
{agent_scratchpad}
(reminder to always respond in a JSON blob)"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
)

prompt = prompt.partial(
    tools=render_text_description_and_args(list(tools)),
    tool_names=", ".join([t.name for t in tools]),
)

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
    )
    | prompt
    | llm
    | JSONAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True
)

agent_executor.invoke({"input": "Total automotive revenues Q3-2024"})

agent_executor.invoke({"input": "Tesla stock market summary for 2024?"})

"""
## **Multiple Query Data**
"""
logger.info("## **Multiple Query Data**")

agent_output = AgentExecutor(
    agent=chain,
    tools=tools,
    handle_parsing_errors=True,
    verbose=False
)

question = [
    "What milestones did the Shanghai factory achieve in Q3 2024?",
    "Tesla stock market summary for 2024?"
]
response = []
contexts = []

for query in question:
    vector_contexts = retriever.get_relevant_documents(query)
    if vector_contexts:
        context_texts = [doc.page_content for doc in vector_contexts]
        contexts.append(context_texts)
    else:
        logger.debug(
            f"[DEBUG] No relevant information in vector store for query: {query}. Falling back to web search.")
        web_results = web_search_tool.run(query)
        contexts.append([web_results])

    result = agent_output.invoke({"input": query})
    response.append(result['output'])

data = {
    "query": question,
    "response": response,
    "context": contexts,
}

"""
# **Connecting to Athina IDE**

Here we are connecting data to [Athina IDE](https://app.athina.ai/develop) to evaluate the performance of our Agentic RAG pipeline.
"""
logger.info("# **Connecting to Athina IDE**")

rows = []
for i in range(len(data["query"])):
    row = {
        'query': data["query"][i],
        'context': data["context"][i],
        'response': data["response"][i],
    }
    rows.append(row)

"""
In the Datasets section on [Athina IDE](https://app.athina.ai/develop), you will find the **Create Dataset** option in the top right corner. Click on it and select **Login via API or SDK** to get the `dataset_id` and `Athina API key`.
"""
logger.info("In the Datasets section on [Athina IDE](https://app.athina.ai/develop), you will find the **Create Dataset** option in the top right corner. Click on it and select **Login via API or SDK** to get the `dataset_id` and `Athina API key`.")


# AthinaApiKey.set_key(os.environ['ATHINA_API_KEY'])

# try:
#     Dataset.add_rows(
#         dataset_id='10a24e8f-3136-4ed0-89cc-a35908897a46',
#         rows=rows
#     )
# except Exception as e:
#     logger.debug(f"Failed to add rows: {e}")

"""
After connecting the data using the Athina SDK, you can access your data at https://app.athina.ai/develop/ {{your_data_id}}
"""

logger.info("\n\n[DONE]", bright=True)
