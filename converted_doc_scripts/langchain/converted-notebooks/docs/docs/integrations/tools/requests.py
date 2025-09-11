from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langgraph.prebuilt import create_react_agent
from typing import Any, Dict, Union
import os
import requests
import shutil
import yaml


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
# Requests Toolkit

We can use the Requests [toolkit](/docs/concepts/tools/#toolkits) to construct agents that generate HTTP requests.

For detailed documentation of all API toolkit features and configurations head to the API reference for [RequestsToolkit](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.openapi.toolkit.RequestsToolkit.html).

## ⚠️ Security note ⚠️
There are inherent risks in giving models discretion to execute real-world actions. Take precautions to mitigate these risks:

- Make sure that permissions associated with the tools are narrowly-scoped (e.g., for database operations or API requests);
- When desired, make use of human-in-the-loop workflows.

## Setup

### Installation

This toolkit lives in the `langchain-community` package:
"""
logger.info("# Requests Toolkit")

# %pip install -qU langchain-community

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

First we will demonstrate a minimal example.

**NOTE**: There are inherent risks in giving models discretion to execute real-world actions. We must "opt-in" to these risks by setting `allow_dangerous_request=True` to use these tools.
**This can be dangerous for calling unwanted requests**. Please make sure your custom OpenAPI spec (yaml) is safe and that permissions associated with the tools are narrowly-scoped.
"""
logger.info("## Instantiation")

ALLOW_DANGEROUS_REQUEST = True

"""
We can use the [JSONPlaceholder](https://jsonplaceholder.typicode.com) API as a testing ground.

Let's create (a subset of) its API spec:
"""
logger.info("We can use the [JSONPlaceholder](https://jsonplaceholder.typicode.com) API as a testing ground.")




def _get_schema(response_json: Union[dict, list]) -> dict:
    if isinstance(response_json, list):
        response_json = response_json[0] if response_json else {}
    return {key: type(value).__name__ for key, value in response_json.items()}


def _get_api_spec() -> str:
    base_url = "https://jsonplaceholder.typicode.com"
    endpoints = [
        "/posts",
        "/comments",
    ]
    common_query_parameters = [
        {
            "name": "_limit",
            "in": "query",
            "required": False,
            "schema": {"type": "integer", "example": 2},
            "description": "Limit the number of results",
        }
    ]
    openapi_spec: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "JSONPlaceholder API", "version": "1.0.0"},
        "servers": [{"url": base_url}],
        "paths": {},
    }
    for endpoint in endpoints:
        response = requests.get(base_url + endpoint)
        if response.status_code == 200:
            schema = _get_schema(response.json())
            openapi_spec["paths"][endpoint] = {
                "get": {
                    "summary": f"Get {endpoint[1:]}",
                    "parameters": common_query_parameters,
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": schema}
                                }
                            },
                        }
                    },
                }
            }
    return yaml.dump(openapi_spec, sort_keys=False)


api_spec = _get_api_spec()

"""
Next we can instantiate the toolkit. We require no authorization or other headers for this API:
"""
logger.info("Next we can instantiate the toolkit. We require no authorization or other headers for this API:")


toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

tools = toolkit.get_tools()

tools

"""
- [RequestsGetTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.requests.tool.RequestsGetTool.html)
- [RequestsPostTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.requests.tool.RequestsPostTool.html)
- [RequestsPatchTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.requests.tool.RequestsPatchTool.html)
- [RequestsPutTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.requests.tool.RequestsPutTool.html)
- [RequestsDeleteTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.requests.tool.RequestsDeleteTool.html)

## Use within an agent
"""
logger.info("## Use within an agent")


llm = ChatOllama(model="llama3.2")

system_message = """
You have access to an API to help answer user queries.
Here is documentation on the API:
{api_spec}
""".format(api_spec=api_spec)

agent_executor = create_react_agent(llm, tools, prompt=system_message)

example_query = "Fetch the top two posts. What are their titles?"

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all API toolkit features and configurations head to the API reference for [RequestsToolkit](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.openapi.toolkit.RequestsToolkit.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)