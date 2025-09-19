from dataclasses import dataclass, field
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import create_tool_from_function
from haystack_integrations.components.generators.anthropic import OllamaFunctionCallingAdapterChatGenerator
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from jet.logger import CustomLogger
from typing import Annotated, Callable, Tuple
import os
import random, re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🐝🐝🐝 Create a Swarm of Agents

> As of Haystack 2.9.0, experimental dataclasses (refactored ChatMessage and ChatRole, ToolCall and Tool) and components (refactored OllamaFunctionCallingAdapterChatGenerator, ToolInvoker) are removed from the `haystack-experimental` and merged into Haystack core.

OllamaFunctionCalling recently released Swarm: an educational framework that proposes lightweight techniques for creating and orchestrating multi-agent systems.


In this notebook, we'll explore the core concepts of Swarm ([Routines and Handoffs](https://cookbook.openai.com/examples/orchestrating_agents)) and implement them using Haystack and its tool support.

This exploration is not only educational: we will unlock features missing in the original implementation, like the ability of using models from various providers. In fact, our final example will include 3 agents: one powered by llama3.2 (OllamaFunctionCalling), one using Claude 3.5 Sonnet (OllamaFunctionCalling) and a third running Llama-3.2-3B locally via Ollama.

## Setup

We install the required dependencies. In addition to Haystack, we also need integrations with OllamaFunctionCalling and Ollama.
"""
logger.info("# 🐝🐝🐝 Create a Swarm of Agents")

# ! pip install haystack-ai jsonschema anthropic-haystack ollama-haystack

"""
Next, we configure our API keys for OllamaFunctionCalling and OllamaFunctionCalling.
"""
logger.info("Next, we configure our API keys for OllamaFunctionCalling and OllamaFunctionCalling.")

# from getpass import getpass


# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass("Enter your OllamaFunctionCalling API key:")
# if not os.environ.get("ANTHROPIC_API_KEY"):
#     os.environ["ANTHROPIC_API_KEY"] = getpass("Enter your OllamaFunctionCalling API key:")




"""
## Starting simple: building an Assistant

The first step toward building an Agent is creating an Assistant: think of it as a Chat Language Model + a system prompt.

We can implement this as a lightweight dataclass with three parameters:
- name
- LLM (Haystack Chat Generator)
- instructions (they will constitute the system message)
"""
logger.info("## Starting simple: building an Assistant")

@dataclass
class Assistant:
    name: str = "Assistant"
    llm: object = OllamaFunctionCallingAdapterChatGenerator()
    instructions: str = "You are a helpful Agent"

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)

    def run(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        new_message = self.llm.run(messages=[self._system_message] + messages)["replies"][0]

        if new_message.text:
            logger.debug(f"\n{self.name}: {new_message.text}")

        return [new_message]

"""
Let's create a Joker assistant, tasked with telling jokes.
"""
logger.info("Let's create a Joker assistant, tasked with telling jokes.")

joker = Assistant(name="Joker", instructions="you are a funny assistant making jokes")

messages = []
logger.debug("Type 'quit' to exit")

while True:
    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    new_messages = joker.run(messages)
    messages.extend(new_messages)

"""
Let's say it tried to do its best 😀

## Tools and Routines

The term Agent has a broad definition. 

However, to qualify as an Agent, a software application built on a Language Model should go beyond simply generating text; it must also be capable of performing actions, such as executing functions.

A popular way of doing this is **Tool calling**:
1. We provide a set of tools (functions, APIs with a given spec) to the model.
2. The model prepares function calls based on user request and available tools.
3. Actual invocation is executed outside the model (at the Agent level).
4. The model can further elaborate on the result of the invocation.

For information on tool support in Haystack, check out the [documentation](https://docs.haystack.deepset.ai/docs/tool).

Swarm introduces **routines**, which are natural-language instructions paired with the tools needed to execute them. Below, we’ll build an agent capable of calling tools and executing routines.


### Implementation

- `instructions` could already be passed to the Assistant, to guide its behavior.

- The Agent introduces a new init parameter called `functions`. These functions are automatically converted into Tools. Key difference: to be passed to a Language Model, a Tool must have a name, a description and JSON schema specifying its parameters.

- During initialization, we also create a `ToolInvoker`. This Haystack component takes in Chat Messages containing prepared `tool_calls`, performs the tool invocation and wraps the results in Chat Message with `tool` role.

- What happens during `run`? The agent first generates a response. If the response includes tool calls, these are executed, and the results are integrated into the conversation.

- The `while` loop manages user interactions:
  - If the last message role is `assistant`, it waits for user input.
  - If the last message role is `tool`, it continues running to handle tool execution and its responses.

*Note: This implementation differs from the original approach by making the Agent responsible for invoking tools directly, instead of delegating control to the `while` loop.*
"""
logger.info("## Tools and Routines")

@dataclass
class ToolCallingAgent:
    name: str = "ToolCallingAgent"
    llm: object = OllamaFunctionCallingAdapterChatGenerator()
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:

        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            logger.debug(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return new_messages

        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        return new_messages

"""
Here's an example of a Refund Agent using this setup.
"""
logger.info("Here's an example of a Refund Agent using this setup.")

def execute_refund(item_name: Annotated[str, "The name of the item to refund"]):
    return f"report: refund succeeded for {item_name} - refund id: {random.randint(0,10000)}"


refund_agent = ToolCallingAgent(
    name="Refund Agent",
    instructions=(
        "You are a refund agent. "
        "Help the user with refunds. "
        "1. Before executing a refund, collect all specific information needed about the item and the reason for the refund. "
        "2. Then collect personal information of the user and bank account details. "
        "3. After executing it, provide a report to the user. "
    ),
    functions=[execute_refund],
)

messages = []
logger.debug("Type 'quit' to exit")

while True:

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    new_messages = refund_agent.run(messages)
    messages.extend(new_messages)

"""
Promising!

## Handoff: switching control between Agents

The most interesting idea of Swarm is probably handoffs: enabling one Agent to transfer control to another with Tool calling. 

**How it works**
1. Add specific handoff functions to the Agent's available tools, allowing it to transfer control when needed.
2. Modify the Agent to return the name of the next agent along with its messages.
3. Handle the switch in  `while` loop.

*The implementation is similar to the previous one, but, compared to `ToolCallingAgent`, a `SwarmAgent` also returns the name of the next agent to be called, enabling handoffs.*
"""
logger.info("## Handoff: switching control between Agents")

HANDOFF_TEMPLATE = "Transferred to: {agent_name}. Adopt persona immediately."
HANDOFF_PATTERN = r"Transferred to: (.*?)(?:\.|$)"


@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    llm: object = OllamaFunctionCallingAdapterChatGenerator()
    instructions: str = "You are a helpful Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = [create_tool_from_function(fun) for fun in self.functions] if self.functions else None
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        if agent_message.text:
            logger.debug(f"\n{self.name}: {agent_message.text}")

        if not agent_message.tool_calls:
            return self.name, new_messages

        for tc in agent_message.tool_calls:
            if tc.id is None:
                tc.id = str(random.randint(0, 1000000))
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        last_result = tool_results[-1].tool_call_result.result
        match = re.search(HANDOFF_PATTERN, last_result)
        new_agent_name = match.group(1) if match else self.name

        return new_agent_name, new_messages

"""
Let's see this in action with a Joker Agent and a Refund Agent!
"""
logger.info("Let's see this in action with a Joker Agent and a Refund Agent!")

def transfer_to_refund():
    """Pass to this Agent for anything related to refunds"""
    return HANDOFF_TEMPLATE.format(agent_name="Refund Agent")


def transfer_to_joker():
    """Pass to this Agent for anything NOT related to refunds."""
    return HANDOFF_TEMPLATE.format(agent_name="Joker Agent")

refund_agent = SwarmAgent(
    name="Refund Agent",
    instructions=(
        "You are a refund agent. "
        "Help the user with refunds. "
        "Ask for basic information but be brief. "
        "For anything unrelated to refunds, transfer to other agent."
    ),
    functions=[execute_refund, transfer_to_joker],
)

joker_agent = SwarmAgent(
    name="Joker Agent",
    instructions=(
        "you are a funny assistant making jokes. "
        "If the user asks questions related to refunds, send him to other agent."
    ),
    functions=[transfer_to_refund],
)

agents = {agent.name: agent for agent in [joker_agent, refund_agent]}

logger.debug("Type 'quit' to exit")

messages = []
current_agent_name = "Joker Agent"

while True:
    agent = agents[current_agent_name]

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)

"""
Nice ✨

# A more complex multi-agent system

Now, we move on to a more intricate multi-agent system that simulates a customer service setup for ACME Corporation, a fictional entity from the Road Runner/Wile E. Coyote cartoons, which sells quirky products meant to catch roadrunners.
(We are reimplementing the example from the original article by OllamaFunctionCalling.)


This system involves several different agents (each with specific tools):
- Triage Agent: handles general questions and directs to other agents. Tools: `transfer_to_sales_agent`, `transfer_to_issues_and_repairs` and `escalate_to_human`.
- Sales Agent: proposes and sells products to the user, it can execute the order or redirect the user back to the Triage Agent. Tools: `execute_order` and `transfer_back_to_triage`.
- Issues and Repairs Agent: supports customers with their problems, it can look up item IDs, execute refund or redirect the user back to triage. Tools: `look_up_item`,  `execute_refund`, and `transfer_back_to_triage`.

A nice bonus feature of our implementation is that **we can use different model providers** supported by Haystack. In this case, the Triage Agent is powered by (OllamaFunctionCalling) llama3.2, while we use (OllamaFunctionCalling) Claude 3.5 Sonnet for the other two agents.
"""
logger.info("# A more complex multi-agent system")

def escalate_to_human(summary: Annotated[str, "A summary"]):
    """Only call this if explicitly asked to."""
    logger.debug("Escalating to human agent...")
    logger.debug("\n=== Escalation Report ===")
    logger.debug(f"Summary: {summary}")
    logger.debug("=========================\n")
    exit()


def transfer_to_sales_agent():
    """Use for anything sales or buying related."""
    return HANDOFF_TEMPLATE.format(agent_name="Sales Agent")


def transfer_to_issues_and_repairs():
    """Use for issues, repairs, or refunds."""
    return HANDOFF_TEMPLATE.format(agent_name="Issues and Repairs Agent")


def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return HANDOFF_TEMPLATE.format(agent_name="Triage Agent")


triage_agent = SwarmAgent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "If the user asks general questions, try to answer them yourself without transferring to another agent. "
        "Only if the user has problems with already bought products, transfer to Issues and Repairs Agent."
        "If the user looks for new products, transfer to Sales Agent."
        "Make tool calls only if necessary and make sure to provide the right arguments."
    ),
    functions=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)


def execute_order(
    product: Annotated[str, "The name of the product"], price: Annotated[int, "The price of the product in USD"]
):
    logger.debug("\n\n=== Order Summary ===")
    logger.debug(f"Product: {product}")
    logger.debug(f"Price: ${price}")
    logger.debug("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        logger.debug("Order execution successful!")
        return "Success"
    else:
        logger.debug("Order cancelled!")
        return "User cancelled order."


sales_agent = SwarmAgent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
    llm=OllamaFunctionCallingAdapterChatGenerator(),
    functions=[execute_order, transfer_back_to_triage],
)


def look_up_item(search_query: Annotated[str, "Search query to find item ID; can be a description or keywords"]):
    """Use to find item ID."""
    item_id = "item_132612938"
    logger.debug("Found item:", item_id)
    return item_id


def execute_refund(
    item_id: Annotated[str, "The ID of the item to refund"], reason: Annotated[str, "The reason for refund"]
):
    logger.debug("\n\n=== Refund Summary ===")
    logger.debug(f"Item ID: {item_id}")
    logger.debug(f"Reason: {reason}")
    logger.debug("=================\n")
    logger.debug("Refund execution successful!")
    return "success"


issues_and_repairs_agent = SwarmAgent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. If the user is interested in buying or general questions, transfer back to Triage Agent.\n"
        "2. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "3. Propose a fix (make one up).\n"
        "4. ONLY if not satesfied, offer a refund.\n"
        "5. If accepted, search for the ID and then execute refund."
        ""
    ),
    functions=[look_up_item, execute_refund, transfer_back_to_triage],
    llm=OllamaFunctionCallingAdapterChatGenerator(),
)

agents = {agent.name: agent for agent in [triage_agent, sales_agent, issues_and_repairs_agent]}

logger.debug("Type 'quit' to exit")

messages = []
current_agent_name = "Triage Agent"

while True:
    agent = agents[current_agent_name]

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)

"""
# 🦙 Put Llama 3.2 in the mix

As demonstrated, our implementation is model-provider agnostic, meaning it can work with both proprietary models and open models running locally.

In practice, you can have Agents that handle complex tasks using powerful proprietary models, and other Agents that perform simpler tasks using smaller open models.

In our example, we will use Llama-3.2-3B-Instruct, a small model with impressive instruction following capabilities (high IFEval score). We'll use **Ollama** to host and serve this model.

### Install and run Ollama

In general, the installation of Ollama is very simple. In this case, we will do some tricks to make it run on Colab.

If you have/enable GPU support, the model will run faster. It can also run well on CPU.
"""
logger.info("# 🦙 Put Llama 3.2 in the mix")

# ! apt install pciutils

# ! curl https://ollama.ai/install.sh | sh

# ! nohup ollama serve > ollama.log 2>&1 &

# ! ollama pull llama3.2:3b

# ! ollama list

"""
### Action!

At this point, we can easily swap the Triage Agent's `llm` with the Llama 3.2 model running on Ollama.

We set a `temperature` < 1 to ensure that generated text is more controlled and not too creative.

⚠️ *Keep in mind that the model is small and that Ollama support for tools is not fully refined yet. As a result, the model may be biased towards generating tool calls (even when not needed) and sometimes may hallucinate tools.*
"""
logger.info("### Action!")

triage_agent.llm = OllamaChatGenerator(model="llama3.2:3b", generation_kwargs={"temperature": 0.8})

agents = {agent.name: agent for agent in [triage_agent, sales_agent, issues_and_repairs_agent]}

logger.debug("Type 'quit' to exit")

messages = []
current_agent_name = "Triage Agent"

while True:
    agent = agents[current_agent_name]

    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)

"""
In conclusion, we have built a multi-agent system using Swarm concepts and Haystack tools, demonstrating how to integrate models from different providers, including a local model running on Ollama.

Swarm ideas are pretty simple and useful for several use cases and the abstractions provided by Haystack make it easy to implement them.
However, this architecture may not be the best fit for all use cases: memory is handled as a list of messages; this system only runs one Agent at a time.

Looking ahead, we plan to develop and showcase more advanced Agents with Haystack. Stay tuned! 📻

## Notebooks on Tool support
- [🛠️ Define & Run Tools](https://haystack.deepset.ai/cookbook/tools_support)
- [📰 Newsletter Sending Agent](https://haystack.deepset.ai/cookbook/newsletter-agent)

(Notebook by [Stefano Fiorucci](https://github.com/anakin87))
"""
logger.info("## Notebooks on Tool support")

logger.info("\n\n[DONE]", bright=True)