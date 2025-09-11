from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import (
AgentExecutor,
AgentOutputParser,
LLMSingleActionAgent,
Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Dict, List
from typing import List, Union
from typing import Optional
import configparser
import json
import os
import re
import requests
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
# Wikibase Agent

This notebook demonstrates a very simple wikibase agent that uses sparql generation. Although this code is intended to work against any
wikibase instance, we use http://wikidata.org for testing.

If you are interested in wikibases and sparql, please consider helping to improve this agent. Look [here](https://github.com/donaldziff/langchain-wikibase) for more details and open questions.

## Preliminaries

### API keys and other secrets

We use an `.ini` file, like this: 
```
[OPENAI]
# OPENAI_API_KEY=xyzzy
[WIKIDATA]
WIKIDATA_USER_AGENT_HEADER=argle-bargle
```
"""
logger.info("# Wikibase Agent")


config = configparser.ConfigParser()
config.read("./secrets.ini")

"""
### Ollama API Key

An Ollama API key is required unless you modify the code below to use another LLM provider.
"""
logger.info("### Ollama API Key")

# ollama_api_key = config["OPENAI"]["OPENAI_API_KEY"]

# os.environ.update({"OPENAI_API_KEY": ollama_api_key})

"""
### Wikidata user-agent header

Wikidata policy requires a user-agent header. See https://meta.wikimedia.org/wiki/User-Agent_policy. However, at present this policy is not strictly enforced.
"""
logger.info("### Wikidata user-agent header")

wikidata_user_agent_header = (
    None
    if not config.has_section("WIKIDATA")
    else config["WIKIDATA"]["WIKIDATA_USER_AGENT_HEADER"]
)

"""
### Enable tracing if desired
"""
logger.info("### Enable tracing if desired")



"""
# Tools

Three tools are provided for this simple agent:
* `ItemLookup`: for finding the q-number of an item
* `PropertyLookup`: for finding the p-number of a property
* `SparqlQueryRunner`: for running a sparql query

## Item and Property lookup

Item and Property lookup are implemented in a single method, using an elastic search endpoint. Not all wikibase instances have it, but wikidata does, and that's where we'll start.
"""
logger.info("# Tools")

def get_nested_value(o: dict, path: list) -> any:
    current = o
    for key in path:
        try:
            current = current[key]
        except KeyError:
            return None
    return current





def vocab_lookup(
    search: str,
    entity_type: str = "item",
    url: str = "https://www.wikidata.org/w/api.php",
    user_agent_header: str = wikidata_user_agent_header,
    srqiprofile: str = None,
) -> Optional[str]:
    headers = {"Accept": "application/json"}
    if wikidata_user_agent_header is not None:
        headers["User-Agent"] = wikidata_user_agent_header

    if entity_type == "item":
        srnamespace = 0
        srqiprofile = "classic_noboostlinks" if srqiprofile is None else srqiprofile
    elif entity_type == "property":
        srnamespace = 120
        srqiprofile = "classic" if srqiprofile is None else srqiprofile
    else:
        raise ValueError("entity_type must be either 'property' or 'item'")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": srnamespace,
        "srlimit": 1,
        "srqiprofile": srqiprofile,
        "srwhat": "text",
        "format": "json",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        title = get_nested_value(response.json(), ["query", "search", 0, "title"])
        if title is None:
            return f"I couldn't find any {entity_type} for '{search}'. Please rephrase your request and try again"
        return title.split(":")[-1]
    else:
        return "Sorry, I got an error. Please try again."

logger.debug(vocab_lookup("Malin 1"))

logger.debug(vocab_lookup("instance of", entity_type="property"))

logger.debug(vocab_lookup("Ceci n'est pas un q-item"))

"""
## Sparql runner

This tool runs sparql - by default, wikidata is used.
"""
logger.info("## Sparql runner")




def run_sparql(
    query: str,
    url="https://query.wikidata.org/sparql",
    user_agent_header: str = wikidata_user_agent_header,
) -> List[Dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if wikidata_user_agent_header is not None:
        headers["User-Agent"] = wikidata_user_agent_header

    response = requests.get(
        url, headers=headers, params={"query": query, "format": "json"}
    )

    if response.status_code != 200:
        return "That query failed. Perhaps you could try a different one?"
    results = get_nested_value(response.json(), ["results", "bindings"])
    return json.dumps(results)

run_sparql("SELECT (COUNT(?children) as ?count) WHERE { wd:Q1339 wdt:P40 ?children . }")

"""
# Agent

## Wrap the tools
"""
logger.info("# Agent")



tools = [
    Tool(
        name="ItemLookup",
        func=(lambda x: vocab_lookup(x, entity_type="item")),
        description="useful for when you need to know the q-number for an item",
    ),
    Tool(
        name="PropertyLookup",
        func=(lambda x: vocab_lookup(x, entity_type="property")),
        description="useful for when you need to know the p-number for a property",
    ),
    Tool(
        name="SparqlQueryRunner",
        func=run_sparql,
        description="useful for getting results from a wikibase",
    ),
]

"""
## Prompts
"""
logger.info("## Prompts")

template = """
Answer the following questions by running a sparql query against a wikibase where the p and q items are
completely unknown to you. You will need to discover the p and q items before you can generate the sparql.
Do not assume you know the p and q items for any concepts. Always use tools to find all p and q items.
After you generate the sparql, you should run it. The results will be returned in json.
Summarize the json results in natural language.

You may assume the following prefixes:
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>

When generating sparql:
* Try to avoid "count" and "filter" queries if possible
* Never enclose the sparql in back-quotes

You have access to the following tools:

{tools}

Use the following format:

Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)

"""
## Output parser 
This is unchanged from langchain docs
"""
logger.info("## Output parser")

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
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
## Specify the LLM model
"""
logger.info("## Specify the LLM model")


llm = ChatOllama(model="llama3.2")

"""
## Agent and agent executor
"""
logger.info("## Agent and agent executor")

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

"""
## Run it!
"""
logger.info("## Run it!")



agent_executor.run("How many children did J.S. Bach have?")

agent_executor.run(
    "What is the Basketball-Reference.com NBA player ID of Hakeem Olajuwon?"
)

logger.info("\n\n[DONE]", bright=True)