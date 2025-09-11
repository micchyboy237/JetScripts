from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage
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
# MLX

This notebook shows how to get started using `MLX` LLM's as chat models.

In particular, we will:
1. Utilize the [MLXPipeline](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/mlx_pipeline.py), 
2. Utilize the `ChatMLX` class to enable any of these LLMs to interface with LangChain's [Chat Messages](https://python.langchain.com/docs/modules/model_io/chat/#messages) abstraction.
3. Demonstrate how to use an open-source LLM to power an `ChatAgent` pipeline
"""
logger.info("# MLX")

# %pip install --upgrade --quiet  mlx-lm transformers huggingface_hub

"""
## 1. Instantiate an LLM

There are three LLM options to choose from.
"""
logger.info("## 1. Instantiate an LLM")


llm = MLXPipeline.from_model_id(
    "mlx-community/quantized-gemma-2b-it",
    pipeline_kwargs={"max_tokens": 10, "temp": 0.1},
)

"""
## 2. Instantiate the `ChatMLX` to apply chat templates

Instantiate the chat model and some messages to pass.
"""
logger.info("## 2. Instantiate the `ChatMLX` to apply chat templates")


messages = [
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

chat_model = ChatMLX(llm=llm)

"""
Inspect how the chat messages are formatted for the LLM call.
"""
logger.info("Inspect how the chat messages are formatted for the LLM call.")

chat_model._to_chat_prompt(messages)

"""
Call the model.
"""
logger.info("Call the model.")

res = chat_model.invoke(messages)
logger.debug(res.content)

"""
## 3. Take it for a spin as an agent!

Here we'll test out `gemma-2b-it` as a zero-shot `ReAct` Agent. The example below is taken from [here](https://python.langchain.com/docs/modules/agents/agent_types/react#using-chat-models).

> Note: To run this section, you'll need to have a [SerpAPI Token](https://serpapi.com/) saved as an environment variable: `SERPAPI_API_KEY`
"""
logger.info("## 3. Take it for a spin as an agent!")


"""
Configure the agent with a `react-json` style prompt and access to a search engine and calculator.
"""
logger.info("Configure the agent with a `react-json` style prompt and access to a search engine and calculator.")

tools = load_tools(["serpapi", "llm-math"], llm=llm)

human_prompt = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.

{input}

{agent_scratchpad}

"""

prompt = human_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)

logger.info("\n\n[DONE]", bright=True)