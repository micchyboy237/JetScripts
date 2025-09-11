from jet.logger import logger
from langchain_nimble import NimbeSearchRetriever
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
sidebar_label: Nimble
---

# Nimble

  [Nimble](https://www.linkedin.com/company/nimbledata) is the first business external data platform, making data decision-making easier than ever, with our award-winning AI-powered data structuring technology Nimble connects business users with the public web knowledge.
We empower businesses with mission-critical real-time external data to unlock advanced business intelligence, price comparison, and other public data for sales and marketing. We translate data into immediate business value.

If you'd like to learn more about Nimble, visit us at [nimbleway.com](https://www.nimbleway.com/).


## Retrievers:

#
#
#
 
N
i
m
b
l
e
S
e
a
r
c
h
R
e
t
r
i
e
v
e
r

Enables developers to build RAG applications and AI Agents that can search, access, and retrieve online information from anywhere on the web.

We need to install the `langchain-nimble` python package.
"""
logger.info("# Nimble")

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
n
i
m
b
l
e

"""
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
r
e
t
r
i
e
v
e
r
s
/
n
i
m
b
l
e
/
)
.

```python
```

N
o
t
e
 
t
h
a
t
 
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
 
i
s
 
r
e
q
u
i
r
e
d
,
 
p
l
e
a
s
e
 
r
e
f
e
r
 
t
o
 
t
h
e
 
[
S
e
t
u
p
 
s
e
c
t
i
o
n
 
i
n
 
t
h
e
 
d
o
c
u
m
e
n
t
a
t
i
o
n
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
r
e
t
r
i
e
v
e
r
s
/
n
i
m
b
l
e
/
#
s
e
t
u
p
)
.
"""
logger.info("#")

logger.info("\n\n[DONE]", bright=True)