from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, create_react_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
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
# Shared memory across agents and tools

This notebook goes over adding memory to **both** an Agent and its tools. Before going through this notebook, please walk through the following notebooks, as this will build on top of both of them:

- [Adding memory to an LLM Chain](/docs/modules/memory/integrations/adding_memory)
- [Custom Agents](/docs/modules/agents/how_to/custom_agent)

We are going to create a custom Agent. The agent has access to a conversation memory, search tool, and a summarization tool. The summarization tool also needs access to the conversation memory.
"""
logger.info("# Shared memory across agents and tools")


template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
readonlymemory = ReadOnlySharedMemory(memory=memory)
summary_chain = LLMChain(
    llm=Ollama(),
    prompt=prompt,
    verbose=True,
    # use the read-only memory to prevent the tool from modifying the memory
    memory=readonlymemory,
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    ),
]

prompt = hub.pull("hwchase17/react")

"""
We can now construct the `LLMChain`, with the Memory object, and then create the agent.
"""
logger.info(
    "We can now construct the `LLMChain`, with the Memory object, and then create the agent.")

model = Ollama()
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

agent_executor.invoke({"input": "What is ChatGPT?"})

"""
To test the memory of this agent, we can ask a followup question that relies on information in the previous exchange to be answered correctly.
"""
logger.info("To test the memory of this agent, we can ask a followup question that relies on information in the previous exchange to be answered correctly.")

agent_executor.invoke({"input": "Who developed it?"})

agent_executor.invoke(
    {"input": "Thanks. Summarize the conversation, for my daughter 5 years old."}
)

"""
Confirm that the memory was correctly updated.
"""
logger.info("Confirm that the memory was correctly updated.")

logger.debug(agent_executor.memory.buffer)

"""


For comparison, below is a bad example that uses the same memory for both the Agent and the tool.
"""
logger.info(
    "For comparison, below is a bad example that uses the same memory for both the Agent and the tool.")

template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

prompt = PromptTemplate(
    input_variables=["input", "chat_history"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")
summary_chain = LLMChain(
    llm=Ollama(),
    prompt=prompt,
    verbose=True,
    memory=memory,  # <--- this is the only change
)

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Summary",
        func=summary_chain.run,
        description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
    ),
]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

agent_executor.invoke({"input": "What is ChatGPT?"})

agent_executor.invoke({"input": "Who developed it?"})

agent_executor.invoke(
    {"input": "Thanks. Summarize the conversation, for my daughter 5 years old."}
)

"""
The final answer is not wrong, but we see the 3rd Human input is actually from the agent in the memory because the memory was modified by the summary tool.
"""
logger.info("The final answer is not wrong, but we see the 3rd Human input is actually from the agent in the memory because the memory was modified by the summary tool.")

logger.debug(agent_executor.memory.buffer)

logger.info("\n\n[DONE]", bright=True)
