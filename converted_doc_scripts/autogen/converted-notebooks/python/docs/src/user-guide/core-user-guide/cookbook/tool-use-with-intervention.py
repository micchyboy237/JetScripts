import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
AgentId,
AgentType,
DefaultInterventionHandler,
DropMessage,
FunctionCall,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
message_handler,
)
from autogen_core.models import (
ChatCompletionClient,
LLMMessage,
SystemMessage,
UserMessage,
)
from autogen_core.tool_agent import ToolAgent, ToolException, tool_agent_caller_loop
from autogen_core.tools import ToolSchema
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
from typing import Any, List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# User Approval for Tool Execution using Intervention Handler

This cookbook shows how to intercept the tool execution using
an intervention hanlder, and prompt the user for permission to execute the tool.
"""
logger.info("# User Approval for Tool Execution using Intervention Handler")



"""
Let's define a simple message type that carries a string content.
"""
logger.info("Let's define a simple message type that carries a string content.")

@dataclass
class Message:
    content: str

"""
Let's create a simple tool use agent that is capable of using tools through a
{py:class}`~autogen_core.tool_agent.ToolAgent`.
"""
logger.info("Let's create a simple tool use agent that is capable of using tools through a")

class ToolUseAgent(RoutedAgent):
    """An agent that uses tools to perform tasks. It executes the tools
    by itself by sending the tool execution task to a ToolAgent."""

    def __init__(
        self,
        description: str,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        tool_schema: List[ToolSchema],
        tool_agent_type: AgentType,
    ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._system_messages = system_messages
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(type=tool_agent_type, key=self.id.key)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        """Handle a user message, execute the model and tools, and returns the response."""
        session: List[LLMMessage] = [UserMessage(content=message.content, source="User")]
        output_messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        final_response = output_messages[-1].content
        assert isinstance(final_response, str)
        return Message(content=final_response)

"""
The tool use agent sends tool call requests to the tool agent to execute tools,
so we can intercept the messages sent by the tool use agent to the tool agent
to prompt the user for permission to execute the tool.

Let's create an intervention handler that intercepts the messages and prompts
user for before allowing the tool execution.
"""
logger.info("The tool use agent sends tool call requests to the tool agent to execute tools,")

class ToolInterventionHandler(DefaultInterventionHandler):
    async def on_send(
        self, message: Any, *, message_context: MessageContext, recipient: AgentId
    ) -> Any | type[DropMessage]:
        if isinstance(message, FunctionCall):
            user_input = input(
                f"Function call: {message.name}\nArguments: {message.arguments}\nDo you want to execute the tool? (y/n): "
            )
            if user_input.strip().lower() != "y":
                raise ToolException(content="User denied tool execution.", call_id=message.id, name=message.name)
        return message

"""
Now, we can create a runtime with the intervention handler registered.
"""
logger.info("Now, we can create a runtime with the intervention handler registered.")

runtime = SingleThreadedAgentRuntime(intervention_handlers=[ToolInterventionHandler()])

"""
In this example, we will use a tool for Python code execution.
First, we create a Docker-based command-line code executor
using {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`,
and then use it to instantiate a built-in Python code execution tool
{py:class}`~autogen_core.tools.PythonCodeExecutionTool`
that runs code in a Docker container.
"""
logger.info("In this example, we will use a tool for Python code execution.")

docker_executor = DockerCommandLineCodeExecutor()

python_tool = PythonCodeExecutionTool(executor=docker_executor)

"""
Register the agents with tools and tool schema.
"""
logger.info("Register the agents with tools and tool schema.")

async def async_func_0():
    tool_agent_type = await ToolAgent.register(
        runtime,
        "tool_executor_agent",
        lambda: ToolAgent(
            description="Tool Executor Agent",
            tools=[python_tool],
        ),
    )
    return tool_agent_type
tool_agent_type = asyncio.run(async_func_0())
logger.success(format_json(tool_agent_type))
model_client = MLXAutogenChatLLMAdapter(model="qwen3-1.7b-4bit-mini")
async def async_func_9():
    await ToolUseAgent.register(
        runtime,
        "tool_enabled_agent",
        lambda: ToolUseAgent(
            description="Tool Use Agent",
            system_messages=[SystemMessage(content="You are a helpful AI Assistant. Use your tools to solve problems.")],
            model_client=model_client,
            tool_schema=[python_tool.schema],
            tool_agent_type=tool_agent_type,
        ),
    )
asyncio.run(async_func_9())

"""
Run the agents by starting the runtime and sending a message to the tool use agent.
The intervention handler will prompt you for permission to execute the tool.
"""
logger.info("Run the agents by starting the runtime and sending a message to the tool use agent.")

async def run_async_code_21307994():
    await docker_executor.start()
asyncio.run(run_async_code_21307994())
async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())

async def async_func_3():
    response = await runtime.send_message(
        Message("Run the following Python code: logger.debug('Hello, World!')"), AgentId("tool_enabled_agent", "default")
    )
    return response
response = asyncio.run(async_func_3())
logger.success(format_json(response))
logger.debug(response.content)

async def run_async_code_4aaa8dea():
    await runtime.stop()
asyncio.run(run_async_code_4aaa8dea())
async def run_async_code_3c182a23():
    await docker_executor.stop()
asyncio.run(run_async_code_3c182a23())

async def run_async_code_0349fda4():
    await model_client.close()
asyncio.run(run_async_code_0349fda4())

logger.info("\n\n[DONE]", bright=True)