from jet.logger import logger
from langchain import hub
from langchain.agents import (
AgentExecutor,
create_react_agent,
)
from langchain.chains import LLMChain
from langchain.globals import set_debug
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import NIBittensorLLM
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pprint import pprint
import json
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
# Bittensor

>[Bittensor](https://bittensor.com/) is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage miners to contribute compute + knowledge.
>
>`NIBittensorLLM` is developed by [Neural Internet](https://neuralinternet.ai/), powered by `Bittensor`.

>This LLM showcases true potential of decentralized AI by giving you the best response(s) from the `Bittensor protocol`, which consist of various AI models such as `Ollama`, `LLaMA2` etc.

Users can view their logs, requests, and API keys on the [Validator Endpoint Frontend](https://api.neuralinternet.ai/). However, changes to the configuration are currently prohibited; otherwise, the user's queries will be blocked.

If you encounter any difficulties or have any questions, please feel free to reach out to our developer on [GitHub](https://github.com/Kunj-2206), [Discord](https://discordapp.com/users/683542109248159777) or join our discord server for latest update and queries [Neural Internet](https://discord.gg/neuralinternet).

## Different Parameter and response handling for NIBittensorLLM
"""
logger.info("# Bittensor")



set_debug(True)

llm_sys = NIBittensorLLM(
    system_prompt="Your task is to determine response based on user prompt.Explain me like I am technical lead of a project"
)
sys_resp = llm_sys(
    "What is bittensor and What are the potential benefits of decentralized AI?"
)
logger.debug(f"Response provided by LLM with system prompt set is : {sys_resp}")


""" {
    "choices":  [
                    {"index": Bittensor's Metagraph index number,
                    "uid": Unique Identifier of a miner,
                    "responder_hotkey": Hotkey of a miner,
                    "message":{"role":"assistant","content": Contains actual response},
                    "response_ms": Time in millisecond required to fetch response from a miner}
                ]
    } """

multi_response_llm = NIBittensorLLM(top_responses=10)
multi_resp = multi_response_llm.invoke("What is Neural Network Feeding Mechanism?")
json_multi_resp = json.loads(multi_resp)
plogger.debug(json_multi_resp)

"""
##  Using NIBittensorLLM with LLMChain and PromptTemplate
"""
logger.info("##  Using NIBittensorLLM with LLMChain and PromptTemplate")


set_debug(True)

template = """Question: {question}

Answer: Let's think step by step."""


prompt = PromptTemplate.from_template(template)

llm = NIBittensorLLM(
    system_prompt="Your task is to determine response based on user prompt."
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is bittensor?"

llm_chain.run(question)

"""
##  Using NIBittensorLLM with Conversational Agent and Google Search Tool
"""
logger.info("##  Using NIBittensorLLM with Conversational Agent and Google Search Tool")


search = GoogleSearchAPIWrapper()

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)


tools = [tool]

prompt = hub.pull("hwchase17/react")


llm = NIBittensorLLM(
    system_prompt="Your task is to determine a response based on user prompt"
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

response = agent_executor.invoke({"input": prompt})

logger.info("\n\n[DONE]", bright=True)