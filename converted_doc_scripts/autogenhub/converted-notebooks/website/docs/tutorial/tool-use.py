from autogen import ConversableAgent
from autogen import register_function
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Tool Use

In the previous chapter, we explored code executors which give
agents the super power of programming.
Agents writing arbitrary code is useful, however,
controlling what code an agent writes can be challenging. 
This is where tools come in.

Tools are pre-defined functions that agents can use. Instead of writing arbitrary
code, agents can call tools to perform actions, such as searching the web,
performing calculations, reading files, or calling remote APIs.
Because you can control what tools are available to an agent, you can control what
actions an agent can perform.

````mdx-code-block
:::note
Tool use is currently only available for LLMs
that support Ollama-compatible tool call API.
:::
````

## Creating Tools

Tools can be created as regular Python functions.
For example, let's create a calculator tool which
can only perform a single operation at a time.
"""
logger.info("# Tool Use")


Operator = Literal["+", "-", "*", "/"]


def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")

"""
The above function takes three arguments:
`a` and `b` are the integer numbers to be operated on;
`operator` is the operation to be performed.
We used type hints to define the types of the arguments and the return value.

````mdx-code-block
:::tip
Always use type hints to define the types of the arguments and the return value
as they provide helpful hints to the agent about the tool's usage.
:::
````

## Registering Tools

Once you have created a tool, you can register it with the agents that
are involved in conversation.
"""
logger.info("## Registering Tools")



assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple calculations. "
    "Return 'TERMINATE' when the task is done.",
#     llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]},
)

user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)

user_proxy.register_for_execution(name="calculator")(calculator)

"""
In the above code, we registered the `calculator` function as a tool with
the assistant and user proxy agents. We also provide a name and a description
for the tool for the assistant agent to understand its usage.

````mdx-code-block
:::tip
Always provide a clear and concise description for the tool as it helps the
agent's underlying LLM to understand the tool's usage.
:::
````

Similar to code executors, a tool must be registered with at least two agents
for it to be useful in conversation. 
The agent registered with the tool's signature
through 
[`register_for_llm`](/docs/reference/agentchat/conversable_agent#register_for_llm)
can call the tool;
the agent registered with the tool's function object through 
[`register_for_execution`](/docs/reference/agentchat/conversable_agent#register_for_execution)
can execute the tool's function.

Alternatively, you can use 
[`autogen.register_function`](/docs/reference/agentchat/conversable_agent#register_function-1)
function to register a tool with both agents at once.
"""
logger.info("In the above code, we registered the `calculator` function as a tool with")


register_function(
    calculator,
    caller=assistant,  # The assistant agent can suggest calls to the calculator.
    executor=user_proxy,  # The user proxy agent can execute the calculator calls.
    name="calculator",  # By default, the function name is used as the tool name.
    description="A simple calculator",  # A description of the tool.
)

"""
## Using Tool

Once the tool is registered, we can use it in conversation.
In the code below, we ask the assistant to perform some arithmetic
calculation using the `calculator` tool.
"""
logger.info("## Using Tool")

chat_result = user_proxy.initiate_chat(assistant, message="What is (44232 + 13312 / (232 - 32)) * 5?")

"""
Let's verify the answer:
"""
logger.info("Let's verify the answer:")

(44232 + int(13312 / (232 - 32))) * 5

"""
The answer is correct.
You can see that the assistant is able to understand the tool's usage
and perform calculation correctly.

## Tool Schema

If you are familiar with [Ollama's tool use API](https://platform.openai.com/docs/guides/function-calling), 
you might be wondering
why we didn't create a tool schema.
In fact, the tool schema is automatically generated from the function signature
and the type hints.
You can see the tool schema by inspecting the `llm_config` attribute of the
agent.
"""
logger.info("## Tool Schema")

assistant.llm_config["tools"]

"""
You can see the tool schema has been automatically generated from the function
signature and the type hints, as well as the description.
This is why it is important to use type hints and provide a clear description
for the tool as the LLM uses them to understand the tool's usage.

You can also use Pydantic model for the type hints to provide more complex
type schema. In the example below, we use a Pydantic model to define the
calculator input.
"""
logger.info("You can see the tool schema has been automatically generated from the function")



class CalculatorInput(BaseModel):
    a: Annotated[int, Field(description="The first number.")]
    b: Annotated[int, Field(description="The second number.")]
    operator: Annotated[Operator, Field(description="The operator.")]


def calculator(input: Annotated[CalculatorInput, "Input to the calculator."]) -> int:
    if input.operator == "+":
        return input.a + input.b
    elif input.operator == "-":
        return input.a - input.b
    elif input.operator == "*":
        return input.a * input.b
    elif input.operator == "/":
        return int(input.a / input.b)
    else:
        raise ValueError("Invalid operator")

"""
Same as before, we register the tool with the agents using the name `"calculator"`.

````mdx-code-block
:::tip
Registering tool to the same name will override the previous tool.
:::
````
"""
logger.info("Same as before, we register the tool with the agents using the name `"calculator"`.")

assistant.register_for_llm(name="calculator", description="A calculator tool that accepts nested expression as input")(
    calculator
)
user_proxy.register_for_execution(name="calculator")(calculator)

"""
You can see the tool schema has been updated to reflect the new type schema.
"""
logger.info("You can see the tool schema has been updated to reflect the new type schema.")

assistant.llm_config["tools"]

"""
Let's use the tool in conversation.
"""
logger.info("Let's use the tool in conversation.")

chat_result = user_proxy.initiate_chat(assistant, message="What is (1423 - 123) / 3 + (32 + 23) * 5?")

"""
Let's verify the answer:
"""
logger.info("Let's verify the answer:")

int((1423 - 123) / 3) + (32 + 23) * 5

"""
Again, the answer is correct. You can see that the assistant is able to understand
the new tool schema and perform calculation correctly.

## How to hide tool usage and code execution within a single agent?

Sometimes it is preferable to hide the tool usage inside a single agent, 
i.e., the tool call and tool response messages are kept invisible from outside
of the agent, and the agent responds to outside messages with tool usages
as "internal monologues". 
For example, you might want build an agent that is similar to
the [Ollama's Assistant](https://platform.openai.com/docs/assistants/how-it-works)
which executes built-in tools internally.

To achieve this, you can use [nested chats](/docs/tutorial/conversation-patterns#nested-chats).
Nested chats allow you to create "internal monologues" within an agent
to call and execute tools. This works for code execution as well.
See [nested chats for tool use](/docs/notebooks/agentchat_nested_chats_chess) for an example.

## Summary

In this chapter, we showed you how to create, register and use tools.
Tools allows agents to perform actions without writing arbitrary code.
In the next chapter, we will introduce conversation patterns, and show
how to use the result of a conversation.
"""
logger.info("## How to hide tool usage and code execution within a single agent?")

logger.info("\n\n[DONE]", bright=True)