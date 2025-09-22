from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
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
# Agents

In LlamaIndex, we define an "agent" as a specific system that uses an LLM, memory, and tools, to handle inputs from outside users. Contrast this with the term "agentic", which generally refers to a superclass of agents, which is any system with LLM decision making in the process.

To create an agent in LlamaIndex, it takes only a few lines of code:
"""
logger.info("# Agents")



def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(model="llama3.2"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    response = await agent.run("What is 1234 * 4567?")
    logger.success(format_json(response))
    logger.debug(str(response))


if __name__ == "__main__":
    asyncio.run(main())

"""
Calling this agent kicks off a specific loop of actions:

- Agent gets the latest message + chat history
- The tool schemas and chat history get sent over the API
- The Agent responds either with a direct response, or a list of tool calls
    - Every tool call is executed
    - The tool call results are added to the chat history
    - The Agent is invoked again with updated history, and either responds directly or selects more calls

The `FunctionAgent` is a type of agent that uses an LLM provider's function/tool calling capabilities to execute tools. Other types of agents, such as [`ReActAgent`](/python/examples/agent/react_agent) and [`CodeActAgent`](/python/examples/agent/code_act_agent), use different prompting strategies to execute tools.

You can visit [the agents guide](/python/framework/understanding/agent) to learn more about agents and their capabilities.

<Aside type="tip">
Some models might not support streaming LLM output. While streaming is enabled by default, if you encounter an error, you can always set `FunctionAgent(..., streaming=False)` to disable streaming.
</Aside>

## Tools

Tools can be defined simply as python functions, or further customized using classes like `FunctionTool` and `QueryEngineTool`. LlamaIndex also provides sets of pre-defined tools for common APIs using something called `Tool Specs`.

You can read more about configuring tools in the [tools guide](/python/framework/module_guides/deploying/agents/tools)

## Memory

Memory is a core-component when building agents. By default, all LlamaIndex agents are using a ChatMemoryBuffer for memory.

To customize it, you can declare it outside the agent and pass it in:
"""
logger.info("## Tools")


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

response = await agent.run(..., memory=memory)
logger.success(format_json(response))

"""
You can read more about configuring memory in the [memory guide](/python/framework/module_guides/deploying/agents/memory)

## Multi-Modal Agents

Some LLMs will support multiple modalities, such as images and text. Using chat messages with content blocks, we can pass in images to an agent for reasoning.

For example, imagine you had a screenshot of the [slide from this presentation](https://docs.google.com/presentation/d/1wy3nuO9ezGS4R99mzP3Q3yvrjAkZ26OGI2NjfqtwAaE/edit?usp=sharing).

You can pass this image to an agent for reasoning, and see that it reads the image and acts accordingly.
"""
logger.info("## Multi-Modal Agents")


llm = Ollama(model="llama3.2")


def add(a: int, b: int) -> int:
    """Useful for adding two numbers together."""
    return a + b


workflow = FunctionAgent(
    tools=[add],
    llm=llm,
)

msg = ChatMessage(
    role="user",
    blocks=[
        TextBlock(text="Follow what the image says."),
        ImageBlock(path="./screenshot.png"),
    ],
)

response = await workflow.run(msg)
logger.success(format_json(response))
logger.debug(str(response))

"""
## Multi-Agent Systems

You can combine agents into a multi-agent system, where each agent is able to hand off control to another agent to coordinate while completing tasks.
"""
logger.info("## Multi-Agent Systems")


multi_agent = AgentWorkflow(agents=[FunctionAgent(...), FunctionAgent(...)])

resp = await agent.run("query")
logger.success(format_json(resp))

"""
This is only one way to build multi-agent systems. Read on to learn more about [multi-agent systems](/python/framework/understanding/agent/multi_agent).

## Manual Agents

While the agent classes like `FunctionAgent`, `ReActAgent`, `CodeActAgent`, and `AgentWorkflow` abstract away a lot of details, sometimes its desirable to build your own lower-level agents.

Using the `LLM` objects directly, you can quickly implement a basic agent loop, while having full control over how the tool calling and error handling works.
"""
logger.info("## Manual Agents")



def select_song(song_name: str) -> str:
    """Useful for selecting a song."""
    return f"Song selected: {song_name}"


tools = [FunctionTool.from_defaults(select_song)]
tools_by_name = {t.metadata.name: t for t in [tool]}

chat_history = [ChatMessage(role="user", content="Pick a random song for me")]
resp = llm.chat_with_tools([tool], chat_history=chat_history)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

while tool_calls:
    chat_history.append(resp.message)

    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        logger.debug(f"Calling {tool_name} with {tool_kwargs}")
        tool_output = tool(**tool_kwargs)
        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )
        )

        resp = llm.chat_with_tools([tool], chat_history=chat_history)
        tool_calls = llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )

logger.debug(resp.message.content)

"""
## Examples / Module Guides

You can find a more complete list of examples and module guides in the [module guides page](/python/framework/module_guides/deploying/agents/modules).
"""
logger.info("## Examples / Module Guides")

logger.info("\n\n[DONE]", bright=True)