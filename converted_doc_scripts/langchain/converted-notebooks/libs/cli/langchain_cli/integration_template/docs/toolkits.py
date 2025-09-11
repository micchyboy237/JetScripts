from __module_name__ import __ModuleName__Toolkit
from jet.logger import logger
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
---
sidebar_label: __ModuleName__
---

# __ModuleName__Toolkit

- TODO: Make sure API reference link is correct.

This will help you get started with the __ModuleName__ [toolkit](/docs/concepts/tools/#toolkits). For detailed documentation of all __ModuleName__Toolkit features and configurations head to the [API reference](https://api.python.langchain.com/en/latest/agent_toolkits/__module_name__.agent_toolkits.__ModuleName__.toolkit.__ModuleName__Toolkit.html).

## Setup

- TODO: Update with relevant info.

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
logger.info("# __ModuleName__Toolkit")



"""
### Installation

This toolkit lives in the `__package_name__` package:
"""
logger.info("### Installation")

# %pip install -qU __package_name__

"""
## Instantiation

Now we can instantiate our toolkit:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


toolkit = __ModuleName__Toolkit(
)

"""
## Tools

View available tools:
"""
logger.info("## Tools")

toolkit.get_tools()

"""
TODO: list API reference pages for individual tools.

## Use within an agent
"""
logger.info("## Use within an agent")


agent_executor = create_react_agent(llm, tools)

example_query = "..."

events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## TODO: Any functionality or considerations specific to this toolkit

Fill in or delete if not relevant.

## API reference

For detailed documentation of all __ModuleName__Toolkit features and configurations head to the [API reference](https://api.python.langchain.com/en/latest/agent_toolkits/__module_name__.agent_toolkits.__ModuleName__.toolkit.__ModuleName__Toolkit.html).
"""
logger.info("## TODO: Any functionality or considerations specific to this toolkit")

logger.info("\n\n[DONE]", bright=True)