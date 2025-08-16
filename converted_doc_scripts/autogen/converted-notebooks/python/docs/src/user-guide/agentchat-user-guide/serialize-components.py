import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.conditions import MaxMessageTermination, StopMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Serializing Components 

AutoGen provides a  {py:class}`~autogen_core.Component` configuration class that defines behaviours  to serialize/deserialize component into declarative specifications. We can accomplish this by calling `.dump_component()` and `.load_component()` respectively. This is useful for debugging, visualizing, and even for sharing your work with others. In this notebook, we will demonstrate how to serialize multiple components to a declarative specification like a JSON file.  


```{warning}

ONLY LOAD COMPONENTS FROM TRUSTED SOURCES.

With serilized components, each component implements the logic for how it is serialized and deserialized - i.e., how the declarative specification is generated and how it is converted back to an object. 

In some cases, creating an object may include executing code (e.g., a serialized function). ONLY LOAD COMPONENTS FROM TRUSTED SOURCES. 
 
```

```{note}
`selector_func` is not serializable and will be ignored during serialization and deserialization process.
```

 
### Termination Condition Example 

In the example below, we will define termination conditions (a part of an agent team) in python, export this to a dictionary/json and also demonstrate how the termination condition object can be loaded from the dictionary/json.
"""
logger.info("# Serializing Components")


max_termination = MaxMessageTermination(5)
stop_termination = StopMessageTermination()

or_termination = max_termination | stop_termination

or_term_config = or_termination.dump_component()
logger.debug("Config: ", or_term_config.model_dump_json())

new_or_termination = or_termination.load_component(or_term_config)

"""
## Agent Example 

In the example below, we will define an agent in python, export this to a dictionary/json and also demonstrate how the agent object can be loaded from the dictionary/json.
"""
logger.info("## Agent Example")


model_client = MLXChatCompletionClient(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="Use tools to solve tasks.",
)
user_proxy = UserProxyAgent(name="user")

user_proxy_config = user_proxy.dump_component()  # dump component
logger.debug(user_proxy_config.model_dump_json())
up_new = user_proxy.load_component(user_proxy_config)  # load component

agent_config = agent.dump_component()  # dump component
logger.debug(agent_config.model_dump_json())
agent_new = agent.load_component(agent_config)  # load component

"""
A similar approach can be used to serialize the `MultiModalWebSurfer` agent.

```python

agent = MultimodalWebSurfer(
    name="web_surfer",
    model_client=model_client,
    headless=False,
)

web_surfer_config = agent.dump_component()  # dump component
logger.debug(web_surfer_config.model_dump_json())

```

## Team Example

In the example below, we will define a team in python, export this to a dictionary/json and also demonstrate how the team object can be loaded from the dictionary/json.
"""
logger.info("## Team Example")


model_client = MLXChatCompletionClient(
    model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
)
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    handoffs=["flights_refunder", "user"],
    system_message="Use tools to solve tasks.",
)

team = RoundRobinGroupChat(participants=[agent], termination_condition=MaxMessageTermination(2))

team_config = team.dump_component()  # dump component
logger.debug(team_config.model_dump_json())

async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)