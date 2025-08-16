from PIL import Image
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.messages import TextMessage
from autogen_core import Image as AGImage
from io import BytesIO
import requests

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Messages

In AutoGen AgentChat, _messages_ facilitate communication and information exchange with other agents, orchestrators, and applications. AgentChat supports various message types, each designed for specific purposes.

## Types of Messages

At a high level, messages in AgentChat can be categorized into two types: agent-agent messages and an agent's internal events and messages.

### Agent-Agent Messages
AgentChat supports many message types for agent-to-agent communication. They belong to subclasses of the base class {py:class}`~autogen_agentchat.messages.BaseChatMessage`. Concrete subclasses covers basic text and multimodal communication, such as {py:class}`~autogen_agentchat.messages.TextMessage` and {py:class}`~autogen_agentchat.messages.MultiModalMessage`.

For example, the following code snippet demonstrates how to create a text message, which accepts a string content and a string source:
"""
logger.info("# Messages")


text_message = TextMessage(content="Hello, world!", source="User")

"""
Similarly, the following code snippet demonstrates how to create a multimodal message, which accepts
a list of strings or {py:class}`~autogen_core.Image` objects:
"""
logger.info("Similarly, the following code snippet demonstrates how to create a multimodal message, which accepts")



pil_image = Image.open(BytesIO(requests.get("https://picsum.photos/300/200").content))
img = AGImage(pil_image)
multi_modal_message = MultiModalMessage(content=["Can you describe the content of this image?", img], source="User")
img

"""
The {py:class}`~autogen_agentchat.messages.TextMessage` and  {py:class}`~autogen_agentchat.messages.MultiModalMessage` we have created can be passed to agents directly via the {py:class}`~autogen_agentchat.base.ChatAgent.on_messages` method, or as tasks given to a team {py:meth}`~autogen_agentchat.teams.BaseGroupChat.run` method. Messages are also used in the responses of an agent. We will explain these in more detail in [Agents](./agents.ipynb) and [Teams](./teams.ipynb).

### Internal Events

AgentChat also supports the concept of `events` - messages that are internal to an agent. These messages are used to communicate events and information on actions _within_ the agent itself, and belong to subclasses of the base class {py:class}`~autogen_agentchat.messages.BaseAgentEvent`.

Examples of these include {py:class}`~autogen_agentchat.messages.ToolCallRequestEvent`, which indicates that a request was made to call a tool, and {py:class}`~autogen_agentchat.messages.ToolCallExecutionEvent`, which contains the results of tool calls.

Typically, events are created by the agent itself and are contained in the {py:attr}`~autogen_agentchat.base.Response.inner_messages` field of the {py:class}`~autogen_agentchat.base.Response` returned from {py:class}`~autogen_agentchat.base.ChatAgent.on_messages`. If you are building a custom agent and have events that you want to communicate to other entities (e.g., a UI), you can include these in the {py:attr}`~autogen_agentchat.base.Response.inner_messages` field of the {py:class}`~autogen_agentchat.base.Response`. We will show examples of this in [Custom Agents](../custom-agents.ipynb).


You can read about the full set of messages supported in AgentChat in the {py:mod}`~autogen_agentchat.messages` module.

## Custom Message Types

You can create custom message types by subclassing the base class {py:class}`~autogen_agentchat.messages.BaseChatMessage` or {py:class}`~autogen_agentchat.messages.BaseAgentEvent`. This allows you to define your own message formats and behaviors, tailored to your application. Custom message types are useful when you write custom agents.
"""
logger.info("### Internal Events")

logger.info("\n\n[DONE]", bright=True)