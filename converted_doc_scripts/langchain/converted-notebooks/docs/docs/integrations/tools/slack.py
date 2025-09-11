from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits import SlackToolkit
from langgraph.prebuilt import create_react_agent
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
# Slack Toolkit

This will help you get started with the Slack [toolkit](/docs/concepts/tools/#toolkits). For detailed documentation of all SlackToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.slack.toolkit.SlackToolkit.html).

## Setup

To use this toolkit, you will need to get a token as explained in the [Slack API docs](https://api.slack.com/tutorials/tracks/getting-a-token). Once you've received a SLACK_USER_TOKEN, you can input it as an environment variable below.
"""
logger.info("# Slack Toolkit")

# import getpass

if not os.getenv("SLACK_USER_TOKEN"):
#     os.environ["SLACK_USER_TOKEN"] = getpass.getpass("Enter your Slack user token: ")

"""
T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
i
n
d
i
v
i
d
u
a
l
 
t
o
o
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("T")



"""
### Installation

This toolkit lives in the `langchain-community` package. We will also need the Slack SDK:
"""
logger.info("### Installation")

# %pip install -qU langchain-community slack_sdk

"""
Optionally, we can install beautifulsoup4 to assist in parsing HTML messages:
"""
logger.info("Optionally, we can install beautifulsoup4 to assist in parsing HTML messages:")

# %pip install -qU beautifulsoup4 # This is optional but is useful for parsing HTML messages

"""
## Instantiation

Now we can instantiate our toolkit:
"""
logger.info("## Instantiation")


toolkit = SlackToolkit()

"""
## Tools

View available tools:
"""
logger.info("## Tools")

tools = toolkit.get_tools()

tools

"""
This toolkit loads:

- [SlackGetChannel](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.get_channel.SlackGetChannel.html)
- [SlackGetMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.get_message.SlackGetMessage.html)
- [SlackScheduleMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.schedule_message.SlackScheduleMessage.html)
- [SlackSendMessage](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.slack.send_message.SlackSendMessage.html)

## Use within an agent

Let's equip an agent with the Slack toolkit and query for information about a channel.
"""
logger.info("## Use within an agent")


llm = ChatOllama(model="llama3.2")

agent_executor = create_react_agent(llm, tools)

example_query = "When was the #general channel created?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    message = event["messages"][-1]
    if message.type != "tool":  # mask sensitive information
        event["messages"][-1].pretty_logger.debug()

example_query = "Send a friendly greeting to channel C072Q1LP4QM."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    message = event["messages"][-1]
    if message.type != "tool":  # mask sensitive information
        event["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all `SlackToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.slack.toolkit.SlackToolkit.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)