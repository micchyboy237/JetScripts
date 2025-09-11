from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllamaEmbeddings
from jet.logger import logger
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from typing import Callable
from typing import Union
import os
import re
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
# Custom agent with tool retrieval

The novel idea introduced in this notebook is the idea of using retrieval to select the set of tools to use to answer an agent query. This is useful when you have many many tools to select from. You cannot put the description of all the tools in the prompt (because of context length issues) so instead you dynamically select the N tools you do want to consider using at run time.

In this notebook we will create a somewhat contrived example. We will have one legitimate tool (search) and then 99 fake tools which are just nonsense. We will then add a step in the prompt template that takes the user input and retrieves tool relevant to the query.

## Set up environment

Do necessary imports, etc.
"""
logger.info("# Custom agent with tool retrieval")


"""
## Set up tools

We will create one legitimate tool (search) and then 99 fake tools.
"""
logger.info("## Set up tools")

search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="useful for when you need to answer questions about current events",
)


def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=f"a silly function that you can use to get more information about the number {i}",
    )
    for i in range(99)
]
ALL_TOOLS = [search_tool] + fake_tools

"""
## Tool Retriever

We will use a vector store to create embeddings for each tool description. Then, for an incoming query we can create embeddings for that query and do a similarity search for relevant tools.
"""
logger.info("## Tool Retriever")


docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]

vector_store = FAISS.from_documents(
    docs, OllamaEmbeddings(model="nomic-embed-text"))

retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.invoke(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]


"""
We can now test this retriever to see if it seems to work.
"""
logger.info("We can now test this retriever to see if it seems to work.")

get_tools("whats the weather?")

get_tools("whats the number 13?")

"""
## Prompt template

The prompt template is pretty standard, because we're not actually changing that much logic in the actual prompt template, but rather we are just changing how retrieval is done.
"""
logger.info("## Prompt template")

template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

"""
The custom prompt template now has the concept of a `tools_getter`, which we call on the input to select the tools to use.
"""
logger.info("The custom prompt template now has the concept of a `tools_getter`, which we call on the input to select the tools to use.")


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        tools = self.tools_getter(kwargs["input"])
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    input_variables=["input", "intermediate_steps"],
)

"""
## Output parser

The output parser is unchanged from the previous notebook, since we are not changing anything about the output format.
"""
logger.info("## Output parser")


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

"""
## Set up LLM, stop sequence, and the agent

Also the same as the previous notebook.
"""
logger.info("## Set up LLM, stop sequence, and the agent")

llm = ChatOllama(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = get_tools("whats the weather?")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

"""
## Use the Agent

Now we can use it!
"""
logger.info("## Use the Agent")

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

agent_executor.run("What's the weather in SF?")

logger.info("\n\n[DONE]", bright=True)
