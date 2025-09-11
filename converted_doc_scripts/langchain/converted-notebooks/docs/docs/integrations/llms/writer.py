from jet.logger import logger
from langchain_community.llms import Writer as WriterLLM
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
# Writer LLM

[Writer](https://writer.com/) is a platform to generate different language content.

This example goes over how to use LangChain to interact with `Writer` [models](https://dev.writer.com/docs/models).

## Setup

To access Writer models you'll need to create a Writer account, get an API key, and install the `writer-sdk` and `langchain-community` packages.

### Credentials

Head to [Writer AI Studio](https://app.writer.com/aistudio/signup?utm_campaign=devrel) to sign up to Ollama and generate an API key. Once you've done this set the WRITER_API_KEY environment variable:
"""
logger.info("# Writer LLM")

# import getpass

if not os.environ.get("WRITER_API_KEY"):
#     os.environ["WRITER_API_KEY"] = getpass.getpass("Enter your Writer API key:")

"""
## Installation

The LangChain Writer integration lives in the `langchain-community` package:
"""
logger.info("## Installation")

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

l
a
n
g
c
h
a
i
n
-
c
o
m
m
u
n
i
t
y

w
r
i
t
e
r
-
s
d
k

"""
N
o
w
 
w
e
 
c
a
n
 
i
n
i
t
i
a
l
i
z
e
 
o
u
r
 
m
o
d
e
l
 
o
b
j
e
c
t
 
t
o
 
i
n
t
e
r
a
c
t
 
w
i
t
h
 
w
r
i
t
e
r
 
L
L
M
s
"""
logger.info("N")


llm = WriterLLM(
    temperature=0.7,
    max_tokens=1000,
)

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
"""
logger.info("#")

r
e
s
p
o
n
s
e
_
t
e
x
t

=

l
l
m
.
i
n
v
o
k
e
(
i
n
p
u
t
=
"
W
r
i
t
e

a

p
o
e
m
"
)

p
r
i
n
t
(
r
e
s
p
o
n
s
e
_
t
e
x
t
)

"""
#
#
 
S
t
r
e
a
m
i
n
g
"""
logger.info("#")

s
t
r
e
a
m
_
r
e
s
p
o
n
s
e

=

l
l
m
.
s
t
r
e
a
m
(
i
n
p
u
t
=
"
T
e
l
l

m
e

a

f
a
i
r
y
t
a
l
e
"
)

for chunk in stream_response:
    logger.debug(chunk, end="")

"""
## Async

Writer support asynchronous calls via **ainvoke()** and **astream()** methods

## API reference

For detailed documentation of all Writer features, head to our [API reference](https://dev.writer.com/api-guides/api-reference/completion-api/text-generation#text-generation).
"""
logger.info("## Async")

logger.info("\n\n[DONE]", bright=True)