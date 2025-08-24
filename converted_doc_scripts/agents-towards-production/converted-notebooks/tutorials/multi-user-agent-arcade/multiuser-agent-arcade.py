from arcadepy import Arcade
from jet.logger import CustomLogger
from langchain_arcade import ToolManager
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.types import interrupt, Command
from typing import Callable, Any
import os
import pprint
import shutil
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--multi-user-agent-arcade--multiuser-agent-arcade)

# Multi-User tool calling with Arcade.dev

This tutorial will help you close one of the biggest gaps between demo and production agents: __Multi-user auth__

When your agents work well in your computer, they are excellent personal assistants, but scaling that up to many users is not easy, as the security assumptions from a local deployment do not apply to agents at scale. Personal Access Tokens simply won't cut it for multiple users. Even if you encapsulate all of the functionality in a remote MCP server, tool-level auth will require you to implement the auth flow for all the providers that your agent relies on.

Arcade solves this by providing a unified platform for agentic tool execution. It will handle the auth flow for you offering a secure multi-user solution for your agents.

In this tutorial you'll learn how to use Arcade and LangGraph to
- Build agents
- Give tools that can interact with
    - GMail
    - Slack
    - Notion
- Implement safety guardrails when calling specific tools (Human-in-the-Loop)

# Setup the environment

Before getting into the code, let's setup our development environment with the right dependencies

---

## System Flow
Here’s how a simple chatbot grows into a robust multi-user system with full safety controls:

```mermaid
graph TD
    A[User Request] --> B[Basic Chat Agent]
    B --> C{Requires External Tools?}
    
    C -->|No| D[Direct LLM Response]
    C -->|Yes| E[Tool Authentication Check]
    
    E --> F{User Authorized?}
    F -->|No| G[Arcade OAuth Flow]
    G --> H[User Grants Permissions]
    H --> I[Store Authorization]
    
    F -->|Yes| J[Tool Execution Request]
    I --> J
    
    J --> K{Sensitive Operation?}
    K -->|No| L[Execute Tool Directly]
    K -->|Yes| M[Human-in-the-Loop Check]
    
    M --> N{User Approves?}
    N -->|Yes| L
    N -->|No| O[Block Execution]
    
    L --> P[Return Results]
    O --> Q[Notify User of Blocked Action]
    D --> R[Final Response]
    P --> R
    Q --> R
    
    style G fill:#e1f5fe
    style M fill:#fff3e0
    style O fill:#ffebee
```

## Development Environment Setup

Before implementing our multi-user agent system, we need to establish a proper development environment with the necessary dependencies. The following installation includes LangGraph for agent orchestration, LangChain-Arcade for tool integration, and the core LangChain library with MLX support.
"""
logger.info("# Multi-User tool calling with Arcade.dev")

# !pip install langgraph langchain-arcade "langchain[openai]"

"""
## API Key Configuration

Our tutorial requires two essential API keys for operation. You will need an [MLX API](https://platform.openai.com/signup) key, as well as an [Arcade API](https://api.arcade.dev/signup?utm_source=github&utm_medium=notebook&utm_campaign=nir_diamant&utm_content=tutorial) key for this tutorial. Both services offer straightforward registration processes, with Arcade specifically designed to simplify the integration of external tools into AI applications.
"""
logger.info("## API Key Configuration")

# import getpass

def _set_env(key: str, default: str | None):
    if key not in os.environ:
        if default:
            os.environ[key] = default
        else:
#             os.environ[key] = getpass.getpass(f"{key}:")


# _set_env("OPENAI_API_KEY")
_set_env("ARCADE_API_KEY")

"""
## User Identity Configuration

The Arcade platform requires user identification to properly manage tool authorizations and maintain security boundaries between different users. This identifier must correspond to the email address used during Arcade account creation, ensuring that tool permissions and OAuth tokens are correctly associated with the appropriate user account.
"""
logger.info("## User Identity Configuration")

_set_env("ARCADE_USER_ID")

"""
# Simple Conversational Agent

We begin our journey by implementing a basic conversational agent that demonstrates core LangGraph functionality without external tool dependencies. This foundational agent provides conversational capabilities with short-term memory, allowing it to maintain context throughout a conversation while establishing the architectural patterns we'll extend throughout this tutorial.

## Core Agent Implementation

The following implementation creates a React-style agent using [LangGraph and Arcade](https://docs.arcade.dev/home/langchain/use-arcade-tools#create-a-react-style-agent?utm_source=github&utm_medium=notebook&utm_campaign=nir_diamant&utm_content=tutorial). We configure it with conversation memory through a MemorySaver checkpointer, enabling the agent to remember previous interactions within the same conversation thread. The agent receives a clear prompt defining its helpful and concise personality, along with instructions for handling unclear requests.
"""
logger.info("# Simple Conversational Agent")


checkpointer = MemorySaver()

agent_a = create_react_agent(
    model="ollama:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times.",
    tools=[], # no tools for now!
    checkpointer=checkpointer
)

"""
## Agent Interaction Utility

To facilitate consistent interaction with our agents throughout this tutorial, we implement a utility function that streams agent responses and displays them in a readable format. This function processes the graph's streaming output and presents the latest message from each interaction cycle, providing immediate feedback during agent conversations.
"""
logger.info("## Agent Interaction Utility")

def run_graph(graph: CompiledStateGraph, config, input):
    for event in graph.stream(input, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_logger.debug()

"""
## Interactive Chat Interface

The following implementation provides a complete interactive chat interface for testing our basic agent. The system generates a unique conversation thread identifier for each session, enabling memory persistence across multiple exchanges within the same conversation. Users can engage naturally with the agent and terminate the session by typing "exit".
"""
logger.info("## Interactive Chat Interface")

config = {
    "configurable": {
        "thread_id": uuid.uuid4()
    }
}
while True:
    user_input = input("👤: ")
    if user_input.lower() == "exit":
        break

    user_message = {"messages": [HumanMessage(content=user_input)]}
    run_graph(agent_a, config, user_message)

"""
## Testing Agent Limitations

To understand the boundaries of our basic agent, we'll test it with requests that require external data access. The following test demonstrates the agent's inability to provide current date information, as most language models lack real-time data access and may provide outdated or inaccurate temporal information.
"""
logger.info("## Testing Agent Limitations")

config = {
    "configurable": {
        "thread_id": uuid.uuid4()
    }
}
logger.debug(f'thread_id = {config["configurable"]["thread_id"]}')

prompt = "what's today's date?"
user_message = {"messages": [HumanMessage(content=prompt)]}
run_graph(agent_a, config, user_message)

"""
## Demonstrating Authentication Requirements

The following test illustrates the agent's complete inability to access private, authenticated data sources. When asked to summarize personal emails, the agent cannot proceed without proper authentication mechanisms and authorized access to external services. This limitation highlights the critical need for secure tool integration in production agent systems.
"""
logger.info("## Demonstrating Authentication Requirements")

config = {
    "configurable": {
        "thread_id": uuid.uuid4()
    }
}
logger.debug(f'thread_id = {config["configurable"]["thread_id"]}')

prompt = "summarize my latest 3 emails please"
user_message = {"messages": [HumanMessage(content=prompt)]}
run_graph(agent_a, config, user_message)

"""
# Tool Integration with Secure Authentication

Having established our basic agent architecture, we now address the core challenge of enabling secure access to external services. This section demonstrates how Arcade.dev solves the complex problem of tool-level authentication, providing a streamlined approach to OAuth integration that scales across multiple users and services.

## Arcade Client Initialization

We begin by establishing connections to the Arcade platform through both the core client and the LangChain integration layer. The ToolManager serves as our primary interface for configuring and authorizing tools, while the Arcade client handles the underlying authentication infrastructure.
"""
logger.info("# Tool Integration with Secure Authentication")


arcade_client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
manager = ToolManager(client=arcade_client)

"""
## Gmail Tool Configuration

Our first tool integration focuses on Gmail access, specifically the email listing capability that our basic agent could not provide. The Gmail_ListEmails tool enables our agent to retrieve and analyze email data, but requires proper user authorization before it can access private email accounts.
"""
logger.info("## Gmail Tool Configuration")

gmail_tool = manager.init_tools(tools=["Gmail_ListEmails"])[0]

"""
## Authorization Utility Function

To streamline the authorization process throughout this tutorial, we implement a reusable function that handles OAuth flow initiation and completion. For reading our email, however, we need to give our app permissions to read it in a secure way. Arcade lets us do this easily by [handling the OAuth2 for us](https://docs.arcade.dev/home/auth/how-arcade-helps?utm_source=github&utm_medium=notebook&utm_campaign=nir_diamant&utm_content=tutorial). This function checks the current authorization status for a specific tool and user combination, initiating the OAuth process when necessary and waiting for user completion of the authorization flow.
"""
logger.info("## Authorization Utility Function")

def authorize_tool(tool_name, user_id, manager):
    auth_response = manager.authorize(
        tool_name=tool_name,
        user_id=user_id
    )
    if auth_response.status != "completed":
        logger.debug(f"The app wants to use the {tool_name} tool.\n"
              f"Please click this url to authorize it {auth_response.url}")
        manager.wait_for_auth(auth_response.id)

"""
## Gmail Authorization Process

The following cell initiates the authorization process for Gmail access. If the user has not previously granted permissions, Arcade will provide an OAuth URL for completing the authorization. Once authorized, the permission persists for future sessions, eliminating the need for repeated authorization flows.
"""
logger.info("## Gmail Authorization Process")

authorize_tool(gmail_tool.name, os.getenv("ARCADE_USER_ID"), manager)

"""
## Enhanced Agent with Gmail Capabilities

With Gmail authorization complete, we can now create an enhanced agent that incorporates email access capabilities. This agent retains all the conversational abilities of our basic implementation while adding the power to interact with authenticated email services. Notice the updated prompt that explicitly mentions Gmail capabilities and the inclusion of the user_id in the configuration for tool execution.
"""
logger.info("## Enhanced Agent with Gmail Capabilities")

agent_b = create_react_agent(
    model="ollama:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times."
           " Use the Gmail tools that you have to address requests about emails.",
    tools=[gmail_tool], # we pass the tool we previously authorized.
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID") # When using Arcade tools, we must provide the user_id on the LangGraph config, so Arcade can execute the tool invoked by the agent.
    }
}
logger.debug(f'thread_id = {config["configurable"]["thread_id"]}')

prompt = "summarize my latest 3 emails please"
user_message = {"messages": [HumanMessage(content=prompt)]}
run_graph(agent_b, config, user_message)

"""
# Multi-Service Tool Integration

Building upon our successful Gmail integration, we now expand our agent's capabilities to include multiple external services. This section demonstrates how to efficiently manage authentication across multiple providers while maintaining security and user experience standards.

## Batch Authorization Utility

Managing multiple tool authorizations individually becomes cumbersome as our agent's capabilities expand. This requires [initializing multiple tools](https://docs.arcade.dev/home/faq#can-i-authenticate-multiple-tools-at-once?utm_source=github&utm_medium=notebook&utm_campaign=nir_diamant&utm_content=tutorial) for the agent, and authenticating the scope of each tool. The following function streamlines this process by grouping authorization scopes by provider, minimizing the number of OAuth flows users must complete while ensuring comprehensive tool access.
"""
logger.info("# Multi-Service Tool Integration")

def authorize_tools(tools, user_id, client):

    provider_to_scopes = {}
    for tool in tools:
        provider = tool.requirements.authorization.provider_id
        if provider not in provider_to_scopes:
            provider_to_scopes[provider] = set()

        if tool.requirements.authorization.oauth2.scopes:
            provider_to_scopes[provider] |= set(tool.requirements.authorization.oauth2.scopes)

    for provider, scopes in provider_to_scopes.items():
        auth_response = client.auth.start(
            user_id=user_id,
            scopes=list(scopes),
            provider=provider
        )

        if auth_response.status != "completed":
            logger.debug(f"🔗 Please click here to authorize: {auth_response.url}")
            logger.debug(f"⏳ Waiting for authorization completion...")

            client.auth.wait_for_completion(auth_response),

"""
## Comprehensive Tool Suite Configuration

We now expand our agent's capabilities by incorporating tools for email sending, Slack communication, and Notion content management. This configuration provides our agent with the ability to not only read information from various services but also to create and send content, enabling more sophisticated workflow automation.
"""
logger.info("## Comprehensive Tool Suite Configuration")

manager.add_tool("Gmail.SendEmail")
manager.add_toolkit("Slack")
manager.add_toolkit("NotionToolkit")

"""
## Multi-Service Authorization

The following cell executes the authorization process for all configured tools simultaneously. This efficient approach minimizes user interaction while establishing the necessary permissions for Gmail, Slack, and Notion access. The batch authorization system automatically groups scopes by provider to present the minimum number of authorization flows.
"""
logger.info("## Multi-Service Authorization")

authorize_tools(
    tools=manager.definitions,
    user_id=os.getenv("ARCADE_USER_ID"),
    client=arcade_client
)

"""
## Multi-Service Agent Implementation

With comprehensive tool authorization complete, we create our most capable agent yet. This implementation leverages the ToolManager's LangChain conversion functionality to provide seamless integration between Arcade's tool definitions and LangGraph's execution framework. The enhanced prompt guides the agent in selecting appropriate tools for different types of requests.
"""
logger.info("## Multi-Service Agent Implementation")

agent_c = create_react_agent(
    model="ollama:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times."
           " Use the Gmail tools to address requests about reading or sending emails."
           " Use the Slack tools to address requests about interactions with users and channels in Slack."
           " Use the Notion tools to address requests about managing content in Notion Pages."
           " In general, when possible, use the most relevant tool for the job.",
    tools=manager.to_langchain(),
    checkpointer=checkpointer
)

"""
## Complex Multi-Service Task Execution

This demonstration showcases our agent's ability to orchestrate complex workflows across multiple services. The request requires the agent to analyze email data, retrieve Slack communications, and explore Notion workspace structure, demonstrating sophisticated tool selection and execution coordination.
"""
logger.info("## Complex Multi-Service Task Execution")

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID") # When using Arcade tools, we must provide the user_id on the LangGraph config, so Arcade can execute the tool invoked by the agent.
    }
}
logger.debug(f'thread_id = {config["configurable"]["thread_id"]}')

prompt = "summarize my latest 3 emails, then show me the latest 3 messages in the #general Slack channel, and tell me about the structure of my Notion Workspace"
user_message = {"messages": [HumanMessage(content=prompt)]}
run_graph(agent_c, config, user_message)

"""
# Human-in-the-Loop Safety Implementation

While our multi-service agent demonstrates impressive capabilities, production systems require robust safety mechanisms to prevent unintended actions. This section implements human-in-the-loop controls for sensitive operations, ensuring that potentially harmful or irreversible actions require explicit user approval before execution.

## Identifying Sensitive Operations

Before implementing safety controls, we must identify which tools require human oversight. The following examination of available tools helps us categorize operations based on their potential impact and irreversibility.
"""
logger.info("# Human-in-the-Loop Safety Implementation")

for tool_name, _ in manager:
    logger.debug(tool_name)

"""
## Sensitive Tool Classification

Based on potential impact analysis, we identify tools that could cause unintended consequences if executed with incorrect parameters. These tools typically involve creating, sending, or modifying data rather than simply retrieving information. The classification focuses on operations that have external effects or could compromise user privacy or system integrity.
"""
logger.info("## Sensitive Tool Classification")

tools_to_protect = [
    "Gmail_SendEmail",
    "Slack_SendDmToUser",
    "Slack_SendMessage",
    "Slack_SendMessageToChannel",
    "NotionToolkit_AppendContentToEndOfPage",
    "NotionToolkit_CreatePage",
]

"""
## Human-in-the-Loop Tool Wrapper

The following implementation creates a wrapper function that transforms regular tools into human-supervised versions. This wrapper intercepts tool execution requests, presents the planned action to the user for approval, and only proceeds with execution upon receiving explicit consent. The implementation leverages LangGraph's interrupt mechanism to pause execution pending user input.
"""
logger.info("## Human-in-the-Loop Tool Wrapper")



def add_human_in_the_loop(
    target_tool: Callable | BaseTool,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(target_tool, BaseTool):
        target_tool = tool(target_tool)

    @tool(
        target_tool.name,
        description=target_tool.description,
        args_schema=target_tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):

        arguments = pprint.pformat(tool_input, indent=4)
        response = interrupt(
            f"Do you allow the call to {target_tool.name} with arguments:\n"
            f"{arguments}"
        )

        if response == "yes":
            tool_response = target_tool.invoke(tool_input, config)
        elif response == "no":
            tool_response = "The User did not allow the tool to run"
        else:
            raise ValueError(
                f"Unsupported interrupt response type: {response}"
            )

        return tool_response

    return call_tool_with_interrupt

"""
## Selective Tool Protection Application

This implementation applies human-in-the-loop protection selectively, wrapping only the tools identified as sensitive while leaving read-only operations unchanged. This approach maintains agent efficiency for safe operations while ensuring appropriate oversight for potentially risky actions.
"""
logger.info("## Selective Tool Protection Application")

protected_tools = [
    add_human_in_the_loop(t)
    if t.name in tools_to_protect else t
    for t in manager.to_langchain()
]

"""
## Interrupt Handling Utilities

LangGraph interrupts require specialized handling to resume execution after user input. The following utilities provide a user-friendly interface for approval decisions and automate the process of resuming agent execution with the user's response. The yes/no loop ensures clear decision-making while the interrupt handler manages the technical aspects of execution resumption.
"""
logger.info("## Interrupt Handling Utilities")

def yes_no_loop(prompt: str) -> str:
    """
    Force the user to say yes or no
    """
    logger.debug(prompt)
    user_input = input("Your response [y/n]: ")
    while user_input.lower() not in ["y", "n"]:
        user_input = input("Your response (must be 'y' or 'n'): ")
    return "yes" if user_input.lower() == "y" else "no"


def handle_interrupts(graph: CompiledStateGraph, config):
    for interr in graph.get_state(config).interrupts:
        approved = yes_no_loop(interr.value)
        run_graph(graph, config, Command(resume=approved))

"""
## Protected Agent Implementation

Our final agent implementation incorporates comprehensive safety controls while maintaining all the multi-service capabilities developed throughout this tutorial. This agent represents a production-ready system that balances functionality with security, ensuring that users maintain control over sensitive operations while benefiting from automated assistance for routine tasks.
"""
logger.info("## Protected Agent Implementation")

agent_hitl = create_react_agent(
    model="ollama:gpt-5",
    prompt="You are a helpful assistant that can help with everyday tasks."
           " If the user's request is confusing you must ask them to clarify"
           " their intent, and fulfill the instruction to the best of your"
           " ability. Be concise and friendly at all times."
           " Use the Gmail tools to address requests about reading or sending emails."
           " Use the Slack tools to address requests about interactions with users and channels in Slack."
           " Use the Notion tools to address requests about managing content in Notion Pages."
           " In general, when possible, use the most relevant tool for the job.",
    tools=protected_tools,
    checkpointer=checkpointer
)

"""
## Safety Mechanism Demonstration

The following test demonstrates our safety system in action by attempting to send a potentially sensitive email. This scenario illustrates how the human-in-the-loop mechanism intercepts the action, presents the details for user review, and awaits explicit approval before proceeding with execution.
"""
logger.info("## Safety Mechanism Demonstration")

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
        "user_id": os.getenv("ARCADE_USER_ID") # When using Arcade tools, we must provide the user_id on the LangGraph config, so Arcade can execute the tool invoked by the agent.
    }
}
logger.debug(f'thread_id = {config["configurable"]["thread_id"]}')

prompt = 'send an email with subject "confidential data" and body "this is top secret information" to random-dude@example.com'
user_message = {"messages": [HumanMessage(content=prompt)]}
run_graph(agent_hitl, config, user_message)

"""
## Interrupt State Inspection

When our safety system activates, the agent execution pauses and enters an interrupt state. The following examination reveals the pending approval request, demonstrating how the system captures the intended action details and awaits user decision before proceeding.
"""
logger.info("## Interrupt State Inspection")

agent_hitl.get_state(config).interrupts

"""
## User Decision Processing

The following cell processes the pending interrupt, presenting the action details to the user and collecting their approval decision. This demonstration shows how users can review potentially sensitive actions and make informed decisions about whether to proceed with agent-proposed operations.
"""
logger.info("## User Decision Processing")

handle_interrupts(agent_hitl, config)

"""
## Complete Interactive System

This final implementation provides a complete interactive system that combines all the capabilities developed throughout this tutorial. Users can engage in natural conversations with an agent that has access to multiple external services while maintaining safety through human-in-the-loop controls for sensitive operations. The system automatically handles authorization, tool execution, and safety approvals in a seamless user experience.
"""
logger.info("## Complete Interactive System")

config = {
    "configurable": {
        "thread_id": uuid.uuid4()
    }
}
while True:
    user_input = input("👤: ")
    if user_input.lower() == "exit":
        break

    user_message = {"messages": [HumanMessage(content=user_input)]}

    run_graph(agent_hitl, config, user_message)

    handle_interrupts(agent_hitl, config)

logger.info("\n\n[DONE]", bright=True)