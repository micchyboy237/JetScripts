from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core.agent.workflow import FunctionAgent
import asyncio
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
---
sidebar:
  order: 1
---

# Building an agent

In LlamaIndex, an agent is a semi-autonomous piece of software powered by an LLM that is given a task and executes a series of steps towards solving that task. It is given a set of tools, which can be anything from arbitrary functions up to full LlamaIndex query engines, and it selects the best available tool to complete each step. When each step is completed, the agent judges whether the task is now complete, in which case it returns a result to the user, or whether it needs to take another step, in which case it loops back to the start.

In LlamaIndex, you can either [build your own agentic workflows from scratch](/python/framework/understanding/workflows), covered in the "Building Workflows" section, or you can use our pre-built agentic workflows like `FunctionAgent` (a simple function/tool calling agent) or `AgentWorkflow` (an agent capable of managing multiple agents). This tutorial covers building a function calling agent using `FunctionAgent`.

To learn about the various ways to build multi-agent systems, go to ["Multi-agent systems"](/python/framework/understanding/agent/multi_agent).

![agent flow](/python/framework/understanding/agent/agent_flow.png)

## Getting started

You can find all of this code in [the agents tutorial repo](https://github.com/run-llama/python-agents-tutorial).

To avoid conflicts and keep things clean, we'll start a new Python virtual environment. You can use any virtual environment manager, but we'll use `poetry` here:
"""
logger.info("# Building an agent")

poetry init
poetry shell

"""
And then we'll install the LlamaIndex library and some other dependencies that will come in handy:
"""
logger.info("And then we'll install the LlamaIndex library and some other dependencies that will come in handy:")

pip install llama-index-core llama-index-llms-ollama python-dotenv

"""
If any of this gives you trouble, check out our more detailed [installation guide](/python/framework/getting_started/installation).

## Ollama Key

Our agent will be powered by Ollama's `llama3.2` LLM, so you'll need an [API key](https://platform.ollama.com/). Once you have your key, you can put it in a `.env` file in the root of your project:
"""
logger.info("## Ollama Key")

# OPENAI_API_KEY=sk-proj-xxxx

"""
If you don't want to use Ollama, you can use [any other LLM](/python/framework/understanding/using_llms) including local models. Agents require capable models, so smaller models may be less reliable.

## Bring in dependencies

We'll start by importing the components of LlamaIndex we need, as well as loading the environment variables from our `.env` file:
"""
logger.info("## Bring in dependencies")


load_dotenv()


"""
## Create basic tools

For this simple example we'll be creating two tools: one that knows how to multiply numbers together, and one that knows how to add them.
"""
logger.info("## Create basic tools")

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

"""
As you can see, these are regular Python functions. When deciding what tool to use, your agent will use the tool's name, parameters, and docstring to determine what the tool does and whether it's appropriate for the task at hand. So it's important to make sure the docstrings are descriptive and helpful. It will also use the type hints to determine the expected parameters and return type.

## Initialize the LLM

`llama3.2` is going to be doing the work today:
"""
logger.info("## Initialize the LLM")

llm = Ollama(model="llama3.2")

"""
You could also pick another popular model accessible via API, such as those from [Mistral](/python/examples/llm/mistralai), [Claude from Ollama](/python/examples/llm/anthropic) or [Gemini from Google](/python/examples/llm/google_genai).

## Initialize the agent

Now we create our agent. It needs an array of tools, an LLM, and a system prompt to tell it what kind of agent to be. Your system prompt would usually be more detailed than this!
"""
logger.info("## Initialize the agent")

workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)

"""
GPT-4o-mini is actually smart enough to not need tools to do such simple math, which is why we specified that it should use tools in the prompt.

Beyond `FunctionAgent`, there are other agents available in LlamaIndex, such as [`ReActAgent`](/python/examples/agent/react_agent) and [`CodeActAgent`](/python/examples/agent/code_act_agent), which use different prompting strategies to execute tools.

## Ask a question

Now we can ask the agent to do some math:
"""
logger.info("## Ask a question")

response = await workflow.run(user_msg="What is 20+(2*4)?")
logger.success(format_json(response))
logger.debug(response)

"""
Note that this is asynchronous code. It will work in a notebook environment, but if you want to run it in regular Python you'll need to wrap it an asynchronous function, like this:
"""
logger.info("Note that this is asynchronous code. It will work in a notebook environment, but if you want to run it in regular Python you'll need to wrap it an asynchronous function, like this:")

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    logger.success(format_json(response))
    logger.debug(response)


if __name__ == "__main__":

    asyncio.run(main())

"""
This should give you output similar to the following:

The result of (20 + (2 times 4)) is 28.

<Aside type="tip">
Some models might not support streaming LLM output. While streaming is enabled by default, if you encounter an error, you can always set `FunctionAgent(..., streaming=False)` to disable streaming.
</Aside>

Check the [repo](https://github.com/run-llama/python-agents-tutorial/blob/main/1_basic_agent.py) to see what the final code should look like.

Congratulations! You've built the most basic kind of agent. Next let's learn how to use [pre-built tools](/python/framework/understanding/agent/tools).
"""
logger.info("This should give you output similar to the following:")

logger.info("\n\n[DONE]", bright=True)