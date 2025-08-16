from autogen import ConversableAgent, UserProxyAgent, config_list_from_json
from autogen import ConversableAgent, config_list_from_json, register_function
from jet.logger import CustomLogger
from typing import Annotated, Literal
import agentops
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent Tracking with AgentOps

<img src="https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/logo/banner-badge.png?raw=true"/>

[AgentOps](https://agentops.ai/?=autogen) provides session replays, metrics, and monitoring for AI agents.

At a high level, AgentOps gives you the ability to monitor LLM calls, costs, latency, agent failures, multi-agent interactions, tool usage, session-wide statistics, and more. For more info, check out the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

### Overview Dashboard
<img src="https://raw.githubusercontent.com/AgentOps-AI/agentops/main/docs/images/external/app_screenshots/overview.gif"/>

### Session Replays
<img src="https://raw.githubusercontent.com/AgentOps-AI/agentops/main/docs/images/external/app_screenshots/drilldown.gif"/>

## Adding AgentOps to an existing Autogen service.
To get started, you'll need to install the AgentOps package and set an API key.

AgentOps automatically configures itself when it's initialized meaning your agent run data will be tracked and logged to your AgentOps account right away.

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install pyautogen agentops
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

### Set an API key

By default, the AgentOps `init()` function will look for an environment variable named `AGENTOPS_API_KEY`. Alternatively, you can pass one in as an optional parameter.

Create an account and obtain an API key at [AgentOps.ai](https://agentops.ai/settings/projects)
"""
logger.info("# Agent Tracking with AgentOps")



agentops.init(api_key="...")

"""
Autogen will now start automatically tracking
- LLM prompts and completions
- Token usage and costs
- Agent names and actions
- Correspondence between agents
- Tool usage
- Errors

# Simple Chat Example
"""
logger.info("# Simple Chat Example")


agentops.init(tags=["simple-autogen-example"])

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
assistant = ConversableAgent("agent", llm_config={"config_list": config_list})

user_proxy = UserProxyAgent("user", code_execution_config=False)

assistant.initiate_chat(user_proxy, message="How can I help you today?")

agentops.end_session("Success")

"""
You can view data on this run at [app.agentops.ai](https://app.agentops.ai). 

The dashboard will display LLM events for each message sent by each agent, including those made by the human user.

![session replay](https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/app_screenshots/session-overview.png?raw=true)

# Tool Example
AgentOps also tracks when Autogen agents use tools. You can find more information on this example in [tool-use.ipynb](https://github.com/microsoft/autogen/blob/main/website/docs/tutorial/tool-use.ipynb)
"""
logger.info("# Tool Example")



agentops.start_session(tags=["autogen-tool-example"])

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


config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with simple calculations. "
    "Return 'TERMINATE' when the task is done.",
    llm_config={"config_list": config_list},
)

user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)
user_proxy.register_for_execution(name="calculator")(calculator)

register_function(
    calculator,
    caller=assistant,  # The assistant agent can suggest calls to the calculator.
    executor=user_proxy,  # The user proxy agent can execute the calculator calls.
    name="calculator",  # By default, the function name is used as the tool name.
    description="A simple calculator",  # A description of the tool.
)

user_proxy.initiate_chat(assistant, message="What is (1423 - 123) / 3 + (32 + 23) * 5?")

agentops.end_session("Success")

"""
You can see your run in action at [app.agentops.ai](https://app.agentops.ai). In this example, the AgentOps dashboard will show:
- Agents talking to each other
- Each use of the `calculator` tool
- Each call to Ollama for LLM use

![Session Drilldown](https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/app_screenshots/session-replay.png?raw=true)
"""
logger.info("You can see your run in action at [app.agentops.ai](https://app.agentops.ai). In this example, the AgentOps dashboard will show:")

logger.info("\n\n[DONE]", bright=True)