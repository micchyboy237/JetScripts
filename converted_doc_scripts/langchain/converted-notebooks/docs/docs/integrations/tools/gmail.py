from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
build_resource_service,
get_gmail_credentials,
)
from langgraph.prebuilt import create_react_agent
import ChatModelTabs from "@theme/ChatModelTabs";
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
# Gmail Toolkit

This will help you get started with the GMail [toolkit](/docs/concepts/tools/#toolkits). This toolkit interacts with the GMail API to read messages, draft and send messages, and more. For detailed documentation of all GmailToolkit features and configurations head to the [API reference](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.toolkit.GmailToolkit.html).

## Setup

To use this toolkit, you will need to set up your credentials explained in the [Gmail API docs](https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application). Once you've downloaded the `credentials.json` file, you can start using the Gmail API.

### Installation

This toolkit lives in the `langchain-google-community` package. We'll need the `gmail` extra:
"""
logger.info("# Gmail Toolkit")

# %pip install -qU langchain-google-community\[gmail\]

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
## Instantiation

By default the toolkit reads the local `credentials.json` file. You can also manually provide a `Credentials` object.
"""
logger.info("## Instantiation")


toolkit = GmailToolkit()

"""
### Customizing Authentication

Behind the scenes, a `googleapi` resource is created using the following methods. 
you can manually build a `googleapi` resource for more auth control.
"""
logger.info("### Customizing Authentication")


credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

tools = toolkit.get_tools()
tools

"""
- [GmailCreateDraft](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.create_draft.GmailCreateDraft.html)
- [GmailSendMessage](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.send_message.GmailSendMessage.html)
- [GmailSearch](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.search.GmailSearch.html)
- [GmailGetMessage](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.get_message.GmailGetMessage.html)
- [GmailGetThread](https://python.langchain.com/api_reference/google_community/gmail/langchain_google_community.gmail.get_thread.GmailGetThread.html)

## Use within an agent

Below we show how to incorporate the toolkit into an [agent](/docs/tutorials/agents).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within an agent")


llm = ChatOllama(model="llama3.2")


agent_executor = create_react_agent(llm, tools)

example_query = "Draft an email to fake@fake.com thanking them for coffee."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all `GmailToolkit` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.gmail.toolkit.GmailToolkit.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)