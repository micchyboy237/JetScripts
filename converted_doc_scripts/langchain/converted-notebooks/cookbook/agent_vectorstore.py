from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from pathlib import Path
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
# Combine agents and vector stores

This notebook covers how to combine agents and vector stores. The use case for this is that you've ingested your data into a vector store and want to interact with it in an agentic manner.

The recommended method for doing so is to create a `RetrievalQA` and then use that as a tool in the overall agent. Let's take a look at doing this below. You can do this with multiple different vector DBs, and use the agent as a way to route between them. There are two different ways of doing this - you can either let the agent use the vector stores as normal tools, or you can set `return_direct=True` to really just use the agent as a router.

## Create the vector store
"""
logger.info("# Combine agents and vector stores")


# os.environ["OPENAI_API_KEY"] = ""


llm = ChatOllama(temperature=0)


relevant_parts = []
for p in Path(".").absolute().parts:
    relevant_parts.append(p)
    if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
        break
doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")


loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
docsearch = Chroma.from_documents(
    texts, embeddings, collection_name="state-of-union")

state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)


loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")

docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)

"""
## Create the Agent
"""
logger.info("## Create the Agent")


tools = [
    Tool(
        name="state_of_union_qa_system",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
    ),
    Tool(
        name="ruff_qa_system",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What did biden say about ketanji brown jackson in the state of the union address?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

input_message = {
    "role": "user",
    "content": "Why use ruff over flake8?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Use the Agent solely as a router

You can also set `return_direct=True` if you intend to use the agent as a router and just want to directly return the result of the RetrievalQAChain.

Notice that in the above examples the agent did some extra work after querying the RetrievalQAChain. You can avoid that and just return the result directly.
"""
logger.info("## Use the Agent solely as a router")

tools = [
    Tool(
        name="state_of_union_qa_system",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
        return_direct=True,
    ),
    Tool(
        name="ruff_qa_system",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
        return_direct=True,
    ),
]


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What did biden say about ketanji brown jackson in the state of the union address?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

input_message = {
    "role": "user",
    "content": "Why use ruff over flake8?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Multi-Hop vector store reasoning

Because vector stores are easily usable as tools in agents, it is easy to use answer multi-hop questions that depend on vector stores using the existing agent framework.
"""
logger.info("## Multi-Hop vector store reasoning")

tools = [
    Tool(
        name="state_of_union_qa_system",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
    Tool(
        name="ruff_qa_system",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
]


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What tool does ruff use to run over Jupyter Notebooks? Did the president mention that tool in the state of the union?",
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)
