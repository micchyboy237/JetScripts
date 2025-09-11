from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_oxylabs import OxylabsSearchAPIWrapper, OxylabsSearchRun
from langchain_oxylabs import OxylabsSearchResults
from langgraph.prebuilt import create_react_agent
from pprint import pprint
import json
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
#
 
O
x
y
l
a
b
s

>[Oxylabs](https://oxylabs.io/) is a market-leading web intelligence collection platform, driven by the highest business, ethics, and compliance standards, enabling companies worldwide to unlock data-driven insights.

## Overview

This package contains the LangChain integration with Oxylabs, providing tools to scrape Google search results with Oxylabs Web Scraper API using LangChain's framework.

The following classes are provided by this package:
- `OxylabsSearchRun` - A tool that returns scraped Google search results in a formatted text
- `OxylabsSearchResults` - A tool that returns scraped Google search results in a JSON format
- `OxylabsSearchAPIWrapper` - An API wrapper for initializing Oxylabs API

|             Pricing             |
|:-------------------------------:|
| ✅ Free 5,000 results for 1 week |

#
#
 
S
e
t
u
p

I
n
s
t
a
l
l
 
t
h
e
 
r
e
q
u
i
r
e
d
 
d
e
p
e
n
d
e
n
c
i
e
s
.
"""
logger.info("#")

# %pip install -qU langchain-oxylabs

"""
#
#
#
 
C
r
e
d
e
n
t
i
a
l
s

S
e
t
 
u
p
 
t
h
e
 
p
r
o
p
e
r
 
A
P
I
 
k
e
y
s
 
a
n
d
 
e
n
v
i
r
o
n
m
e
n
t
 
v
a
r
i
a
b
l
e
s
.
 
C
r
e
a
t
e
 
y
o
u
r
 
A
P
I
 
u
s
e
r
 
c
r
e
d
e
n
t
i
a
l
s
:
 
S
i
g
n
 
u
p
 
f
o
r
 
a
 
f
r
e
e
 
t
r
i
a
l
 
o
r
 
p
u
r
c
h
a
s
e
 
t
h
e
 
p
r
o
d
u
c
t
 
i
n
 
t
h
e
 
[
O
x
y
l
a
b
s
 
d
a
s
h
b
o
a
r
d
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
a
s
h
b
o
a
r
d
.
o
x
y
l
a
b
s
.
i
o
/
e
n
/
r
e
g
i
s
t
r
a
t
i
o
n
)
 
t
o
 
c
r
e
a
t
e
 
y
o
u
r
 
A
P
I
 
u
s
e
r
 
c
r
e
d
e
n
t
i
a
l
s
 
(
O
X
Y
L
A
B
S
_
U
S
E
R
N
A
M
E
 
a
n
d
 
O
X
Y
L
A
B
S
_
P
A
S
S
W
O
R
D
)
.
"""
logger.info("#")

# import getpass

# os.environ["OXYLABS_USERNAME"] = getpass.getpass("Enter your Oxylabs username: ")
# os.environ["OXYLABS_PASSWORD"] = getpass.getpass("Enter your Oxylabs password: ")

"""
#
#
 
I
n
s
t
a
n
t
i
a
t
i
o
n
"""
logger.info("#")


oxylabs_wrapper = OxylabsSearchAPIWrapper()
tool_ = OxylabsSearchRun(wrapper=oxylabs_wrapper)

"""
#
#
 
I
n
v
o
c
a
t
i
o
n

#
#
#
 
I
n
v
o
k
e
 
d
i
r
e
c
t
l
y
 
w
i
t
h
 
a
r
g
s

T
h
e
 
`
O
x
y
l
a
b
s
S
e
a
r
c
h
R
u
n
`
 
t
o
o
l
 
t
a
k
e
s
 
a
 
s
i
n
g
l
e
 
"
q
u
e
r
y
"
 
a
r
g
u
m
e
n
t
,
 
w
h
i
c
h
 
s
h
o
u
l
d
 
b
e
 
a
 
n
a
t
u
r
a
l
 
l
a
n
g
u
a
g
e
 
q
u
e
r
y
 
a
n
d
 
r
e
t
u
r
n
s
 
c
o
m
b
i
n
e
d
 
s
t
r
i
n
g
 
f
o
r
m
a
t
 
r
e
s
u
l
t
:
"""
logger.info("#")

tool_.invoke({"query": "Restaurants in Paris."})

"""
#
#
#
 
I
n
v
o
k
e
 
w
i
t
h
 
T
o
o
l
C
a
l
l
"""
logger.info("#")

tool_ = OxylabsSearchRun(
    wrapper=oxylabs_wrapper,
    kwargs={
        "result_categories": [
            "local_information",
            "combined_search_result",
        ]
    },
)


model_generated_tool_call = {
    "args": {
        "query": "Visit restaurants in Vilnius.",
        "geo_location": "Vilnius,Lithuania",
    },
    "id": "1",
    "name": "oxylabs_search",
    "type": "tool_call",
}
tool_call_result = tool_.invoke(model_generated_tool_call)

plogger.debug(tool_call_result.content)

"""
## Use within an agent
Install the required dependencies.
"""
logger.info("## Use within an agent")

# %
p
i
p

i
n
s
t
a
l
l

-
q
U

"
l
a
n
g
c
h
a
i
n
[
o
p
e
n
a
i
]
"

l
a
n
g
g
r
a
p
h

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for Ollama: ")
llm = init_chat_model("llama3.2", model_provider="ollama")


tool_ = OxylabsSearchRun(wrapper=oxylabs_wrapper)

agent = create_react_agent(llm, [tool_])

user_input = "What happened in the latest Burning Man floods?"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## JSON results
`OxylabsSearchResults` tool can be used as an alternative to `OxylabsSearchRun` to retrieve results in a JSON format:
"""
logger.info("## JSON results")



tool_ = OxylabsSearchResults(wrapper=oxylabs_wrapper)

response_results = tool_.invoke({"query": "What are the most famous artists?"})
response_results = json.loads(response_results)

for result in response_results:
    for key, value in result.items():
        logger.debug(f"{key}: {value}")

"""
## API reference
More information about this integration package can be found here: https://github.com/oxylabs/langchain-oxylabs

Oxylabs Web Scraper API documentation: https://developers.oxylabs.io/scraper-apis/web-scraper-api
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)