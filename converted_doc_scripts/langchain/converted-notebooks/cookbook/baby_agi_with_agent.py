from jet.adapters.langchain.chat_ollama import Ollama
from jet.adapters.langchain.chat_ollama import Ollama, OllamaEmbeddings
from jet.logger import logger
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import BabyAGI
from typing import Optional
import faiss
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
# BabyAGI with Tools

This notebook builds on top of [baby agi](baby_agi.html), but shows how you can swap out the execution chain. The previous execution chain was just an LLM which made stuff up. By swapping it out with an agent that has access to tools, we can hopefully get real reliable information

## Install and Import Required Modules
"""
logger.info("# BabyAGI with Tools")



"""
## Connect to the Vector Store

Depending on what vectorstore you use, this step may look different.
"""
logger.info("## Connect to the Vector Store")

# %pip install faiss-cpu > /dev/null
# %pip install google-search-results > /dev/null

embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

"""
## Define the Chains

BabyAGI relies on three LLM chains:
- Task creation chain to select new tasks to add to the list
- Task prioritization chain to re-prioritize tasks
- Execution Chain to execute the tasks


NOTE: in this notebook, the Execution chain will now be an agent.
"""
logger.info("## Define the Chains")


todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=Ollama(temperature=0), prompt=todo_prompt)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]


prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = Ollama(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

"""
### Run the BabyAGI

Now it's time to create the BabyAGI controller and watch it try to accomplish your objective.
"""
logger.info("### Run the BabyAGI")

OBJECTIVE = "Write a weather report for SF today"

verbose = False
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=verbose,
    max_iterations=max_iterations,
)

baby_agi({"objective": OBJECTIVE})

logger.info("\n\n[DONE]", bright=True)