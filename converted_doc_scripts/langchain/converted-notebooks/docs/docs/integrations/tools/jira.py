from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
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
# Jira Toolkit

This notebook goes over how to use the `Jira` toolkit.

The `Jira` toolkit allows agents to interact with a given Jira instance, performing actions such as searching for issues and creating issues, the tool wraps the atlassian-python-api library, for more see: https://atlassian-python-api.readthedocs.io/jira.html

## Installation and setup

To use this tool, you must first set as environment variables:
    JIRA_INSTANCE_URL,
    JIRA_CLOUD

You have the choice between two authentication methods:
- API token authentication: set the JIRA_API_TOKEN (and JIRA_USERNAME if needed) environment variables
- OAuth2.0 authentication: set the JIRA_OAUTH2 environment variable as a dict having as fields "client_id" and "token" which is a dict containing at least "access_token" and "token_type"
"""
logger.info("# Jira Toolkit")

# %pip install --upgrade --quiet  atlassian-python-api

# %pip install -qU langchain-community langchain-ollama


"""
F
o
r
 
a
u
t
h
e
n
t
i
c
a
t
i
o
n
 
w
i
t
h
 
A
P
I
 
t
o
k
e
n
"""
logger.info("F")

os.environ["JIRA_API_TOKEN"] = "abc"
os.environ["JIRA_USERNAME"] = "123"
os.environ["JIRA_INSTANCE_URL"] = "https://jira.atlassian.com"
# os.environ["OPENAI_API_KEY"] = "xyz"
os.environ["JIRA_CLOUD"] = "True"

"""
F
o
r
 
a
u
t
h
e
n
t
i
c
a
t
i
o
n
 
w
i
t
h
 
a
 
O
a
u
t
h
2
.
0
"""
logger.info("F")

os.environ["JIRA_OAUTH2"] = (
    '{"client_id": "123", "token": {"access_token": "abc", "token_type": "bearer"}}'
)
os.environ["JIRA_INSTANCE_URL"] = "https://jira.atlassian.com"
# os.environ["OPENAI_API_KEY"] = "xyz"
os.environ["JIRA_CLOUD"] = "True"

llm = ChatOllama(temperature=0)
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)

"""
## Tool usage

Let's see what individual tools are in the Jira toolkit:
"""
logger.info("## Tool usage")

[(tool.name, tool.description) for tool in toolkit.get_tools()]

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("make a new issue in project PW to remind me to make more fried rice")

logger.info("\n\n[DONE]", bright=True)
