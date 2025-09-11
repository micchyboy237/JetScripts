from jet.logger import logger
from langchain_exa import ExaSearchRetriever
from langchain_exa.tools import ExaFindSimilarResults
from langchain_exa.tools import ExaSearchResults
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
# Exa

>[Exa](https://exa.ai/) is a knowledge API for AI and developers.
>

## Installation and Setup

`Exa` integration exists in its own [partner package](https://pypi.org/project/langchain-exa/). You can install it with:
"""
logger.info("# Exa")

# %pip install -qU langchain-exa

"""
In order to use the package, you will also need to set the `EXA_API_KEY` environment variable to your Exa API key.

## Retriever

You can use the [`ExaSearchRetriever`](/docs/integrations/tools/exa_search#using-exasearchretriever) in a standard retrieval pipeline. You can import it as follows.

See a [usage example](/docs/integrations/tools/exa_search).
"""
logger.info("## Retriever")


"""
#
#
 
T
o
o
l
s


Y
o
u
 
c
a
n
 
u
s
e
 
E
x
a
 
a
s
 
a
n
 
a
g
e
n
t
 
t
o
o
l
 
a
s
 
d
e
s
c
r
i
b
e
d
 
i
n
 
t
h
e
 
[
E
x
a
 
t
o
o
l
 
c
a
l
l
i
n
g
 
d
o
c
s
]
(
/
d
o
c
s
/
i
n
t
e
g
r
a
t
i
o
n
s
/
t
o
o
l
s
/
e
x
a
_
s
e
a
r
c
h
#
u
s
e
-
w
i
t
h
i
n
-
a
n
-
a
g
e
n
t
)
.


S
e
e
 
a
 
[
u
s
a
g
e
 
e
x
a
m
p
l
e
]
(
/
d
o
c
s
/
i
n
t
e
g
r
a
t
i
o
n
s
/
t
o
o
l
s
/
e
x
a
_
s
e
a
r
c
h
)
.


#
#
#
 
E
x
a
F
i
n
d
S
i
m
i
l
a
r
R
e
s
u
l
t
s


A
 
t
o
o
l
 
t
h
a
t
 
q
u
e
r
i
e
s
 
t
h
e
 
M
e
t
a
p
h
o
r
 
S
e
a
r
c
h
 
A
P
I
 
a
n
d
 
g
e
t
s
 
b
a
c
k
 
J
S
O
N
.
"""
logger.info("#")


"""
### ExaSearchResults

Exa Search tool.
"""
logger.info("### ExaSearchResults")


logger.info("\n\n[DONE]", bright=True)