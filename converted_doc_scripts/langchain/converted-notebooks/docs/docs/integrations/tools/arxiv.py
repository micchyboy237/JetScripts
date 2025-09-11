from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_community.utilities import ArxivAPIWrapper
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
# ArXiv

This notebook goes over how to use the `arxiv` tool with an agent. 

First, you need to install the `arxiv` python package.
"""
logger.info("# ArXiv")

# %pip install --upgrade --quiet  langchain-community arxiv


llm = ChatOllama(model="llama3.2")
tools = load_tools(
    ["arxiv"],
)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What's the paper 1605.08386 about?",
    }
)

"""
## The ArXiv API Wrapper

The tool uses the `API Wrapper`. Below, we explore some of the features it provides.
"""
logger.info("## The ArXiv API Wrapper")


"""
You can use the ArxivAPIWrapper to get information about a scientific article or articles. The query text is limited to 300 characters.

The ArxivAPIWrapper returns these article fields:
- Publishing date
- Title
- Authors
- Summary

The following query returns information about one article with the arxiv ID "1605.08386".
"""
logger.info("You can use the ArxivAPIWrapper to get information about a scientific article or articles. The query text is limited to 300 characters.")

arxiv = ArxivAPIWrapper()
docs = arxiv.run("1605.08386")
docs

"""
Now, we want to get information about one author, `Caprice Stanley`.

This query returns information about three articles. By default, the query returns information only about three top articles.
"""
logger.info("Now, we want to get information about one author, `Caprice Stanley`.")

docs = arxiv.run("Caprice Stanley")
docs

"""
Now, we are trying to find information about non-existing article. In this case, the response is "No good Arxiv Result was found"
"""
logger.info("Now, we are trying to find information about non-existing article. In this case, the response is "No good Arxiv Result was found"")

docs = arxiv.run("1605.08386WWW")
docs

logger.info("\n\n[DONE]", bright=True)