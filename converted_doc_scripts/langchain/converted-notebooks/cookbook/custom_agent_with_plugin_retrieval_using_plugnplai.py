from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllamaEmbeddings
from jet.logger import logger
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_community.agent_toolkits import NLAToolkit
from langchain_community.tools.plugin import AIPlugin
from langchain_community.vectorstores import FAISS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from typing import Callable
from typing import Union
import os
import plugnplai
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
# Plug-and-Plai

This notebook builds upon the idea of [plugin retrieval](./custom_agent_with_plugin_retrieval.html), but pulls all tools from `plugnplai` - a directory of AI Plugins.

## Set up environment

Do necessary imports, etc.

Install plugnplai lib to get a list of active plugins from https://plugplai.com directory
"""
logger.info("# Plug-and-Plai")

pip install plugnplai - q


"""
## Setup LLM
"""
logger.info("## Setup LLM")

llm = ChatOllama(temperature=0)

"""
## Set up plugins

Load and index plugins
"""
logger.info("## Set up plugins")

urls = plugnplai.get_plugins()

urls = plugnplai.get_plugins(filter="ChatGPT")

urls = plugnplai.get_plugins(filter="working")


AI_PLUGINS = [AIPlugin.from_url(
    url + "/.well-known/ai-plugin.json") for url in urls]

"""
## Tool Retriever

We will use a vectorstore to create embeddings for each tool description. Then, for an incoming query we can create embeddings for that query and do a similarity search for relevant tools.
"""
logger.info("## Tool Retriever")


embeddings = OllamaEmbeddings(model="nomic-embed-text")
docs = [
    Document(
        page_content=plugin.description_for_model,
        metadata={"plugin_name": plugin.name_for_model},
    )
    for plugin in AI_PLUGINS
]
vector_store = FAISS.from_documents(docs, embeddings)
toolkits_dict = {
    plugin.name_for_model: NLAToolkit.from_llm_and_ai_plugin(llm, plugin)
    for plugin in AI_PLUGINS
}

retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.invoke(query)
    tool_kits = [toolkits_dict[d.metadata["plugin_name"]] for d in docs]
    tools = []
    for tk in tool_kits:
        tools.extend(tk.nla_tools)
    return tools


"""
We can now test this retriever to see if it seems to work.
"""
logger.info("We can now test this retriever to see if it seems to work.")

tools = get_tools("What could I do today with my kiddo")
[t.name for t in tools]

tools = get_tools("what shirts can i buy?")
[t.name for t in tools]

"""
## Prompt Template

The prompt template is pretty standard, because we're not actually changing that much logic in the actual prompt template, but rather we are just changing how retrieval is done.
"""
logger.info("## Prompt Template")

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
The custom prompt template now has the concept of a tools_getter, which we call on the input to select the tools to use
"""
logger.info("The custom prompt template now has the concept of a tools_getter, which we call on the input to select the tools to use")


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
## Output Parser

The output parser is unchanged from the previous notebook, since we are not changing anything about the output format.
"""
logger.info("## Output Parser")


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

Also the same as the previous notebook
"""
logger.info("## Set up LLM, stop sequence, and the agent")

llm = ChatOllama(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

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

agent_executor.run("what shirts can i buy?")

logger.info("\n\n[DONE]", bright=True)
