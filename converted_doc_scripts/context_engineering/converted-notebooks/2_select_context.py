from jet.logger import logger
from langchain.embeddings import init_embeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
convert_positional_only_function_to_tool
)
from typing_extensions import Literal
from utils import format_messages
import math
import os
import shutil
import types
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


"""
# Selecting Context in LangGraph

*Selecting context means pulling it into the context window to help an agent perform a task.*

![Screenshot 2025-07-09 at 2.28.01 PM.png](attachment:da8d31d0-8a43-45bc-9784-570e68eca4e7.png)

## Scratchpad

The mechanism for selecting context from a scratchpad depends upon how the scratchpad is implemented. If it’s a [tool](https://www.anthropic.com/engineering/claude-think-tool), then an agent can simply read it by making a tool call. If it’s part of the agent’s runtime state, then the developer can choose what parts of state to expose to an agent each step. This provides a fine-grained level of control for exposing context to an agent.

### Scratchpad selecting in LangGraph

In `1_write_context.ipynb`, we saw how to write to the LangGraph state object. Now, we'll see how to select context from state and present it to an LLM call in a downstream node. This ability to select from state gives us control over what context we present to LLM calls.
"""
logger.info("# Selecting Context in LangGraph")

f
r
o
m

t
y
p
i
n
g

i
m
p
o
r
t

T
y
p
e
d
D
i
c
t


f
r
o
m

r
i
c
h
.
c
o
n
s
o
l
e

i
m
p
o
r
t

C
o
n
s
o
l
e

f
r
o
m

r
i
c
h
.
p
r
e
t
t
y

i
m
p
o
r
t

p
p
r
i
n
t



I
n
i
t
i
a
l
i
z
e

c
o
n
s
o
l
e

f
o
r

r
i
c
h

f
o
r
m
a
t
t
i
n
g

c
o
n
s
o
l
e

=

C
o
n
s
o
l
e
(
)



c
l
a
s
s

S
t
a
t
e
(
T
y
p
e
d
D
i
c
t
)
:





"
"
"
S
t
a
t
e

s
c
h
e
m
a

f
o
r

t
h
e

c
o
n
t
e
x
t

s
e
l
e
c
t
i
o
n

w
o
r
k
f
l
o
w
.










A
t
t
r
i
b
u
t
e
s
:









t
o
p
i
c
:

T
h
e

t
o
p
i
c

f
o
r

j
o
k
e

g
e
n
e
r
a
t
i
o
n









j
o
k
e
:

T
h
e

g
e
n
e
r
a
t
e
d

j
o
k
e

c
o
n
t
e
n
t





"
"
"





t
o
p
i
c
:

s
t
r





j
o
k
e
:

s
t
r

i
m
p
o
r
t

g
e
t
p
a
s
s

i
m
p
o
r
t

o
s


f
r
o
m

I
P
y
t
h
o
n
.
d
i
s
p
l
a
y

i
m
p
o
r
t

I
m
a
g
e
,

d
i
s
p
l
a
y

f
r
o
m

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
h
a
t
_
m
o
d
e
l
s

i
m
p
o
r
t

i
n
i
t
_
c
h
a
t
_
m
o
d
e
l

f
r
o
m

l
a
n
g
g
r
a
p
h
.
g
r
a
p
h

i
m
p
o
r
t

E
N
D
,

S
T
A
R
T
,

S
t
a
t
e
G
r
a
p
h



d
e
f

_
s
e
t
_
e
n
v
(
v
a
r
:

s
t
r
)

-
>

N
o
n
e
:





"
"
"
S
e
t

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

i
f

n
o
t

a
l
r
e
a
d
y

s
e
t
.
"
"
"





i
f

n
o
t

o
s
.
e
n
v
i
r
o
n
.
g
e
t
(
v
a
r
)
:









o
s
.
e
n
v
i
r
o
n
[
v
a
r
]

=

g
e
t
p
a
s
s
.
g
e
t
p
a
s
s
(
f
"
{
v
a
r
}
:

"
)




S
e
t

u
p

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

a
n
d

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

m
o
d
e
l

_
s
e
t
_
e
n
v
(
"
A
N
T
H
R
O
P
I
C
_
A
P
I
_
K
E
Y
"
)

l
l
m

=

i
n
i
t
_
c
h
a
t
_
m
o
d
e
l
(
"
a
n
t
h
r
o
p
i
c
:
c
l
a
u
d
e
-
s
o
n
n
e
t
-
4
-
2
0
2
5
0
5
1
4
"
,

t
e
m
p
e
r
a
t
u
r
e
=
0
)



d
e
f

g
e
n
e
r
a
t
e
_
j
o
k
e
(
s
t
a
t
e
:

S
t
a
t
e
)

-
>

d
i
c
t
[
s
t
r
,

s
t
r
]
:





"
"
"
G
e
n
e
r
a
t
e

a
n

i
n
i
t
i
a
l

j
o
k
e

a
b
o
u
t

t
h
e

t
o
p
i
c
.










A
r
g
s
:









s
t
a
t
e
:

C
u
r
r
e
n
t

s
t
a
t
e

c
o
n
t
a
i
n
i
n
g

t
h
e

t
o
p
i
c














R
e
t
u
r
n
s
:









D
i
c
t
i
o
n
a
r
y

w
i
t
h

t
h
e

g
e
n
e
r
a
t
e
d

j
o
k
e





"
"
"





m
s
g

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
f
"
W
r
i
t
e

a

s
h
o
r
t

j
o
k
e

a
b
o
u
t

{
s
t
a
t
e
[
'
t
o
p
i
c
'
]
}
"
)





r
e
t
u
r
n

{
"
j
o
k
e
"
:

m
s
g
.
c
o
n
t
e
n
t
}



d
e
f

i
m
p
r
o
v
e
_
j
o
k
e
(
s
t
a
t
e
:

S
t
a
t
e
)

-
>

d
i
c
t
[
s
t
r
,

s
t
r
]
:





"
"
"
I
m
p
r
o
v
e

a
n

e
x
i
s
t
i
n
g

j
o
k
e

b
y

a
d
d
i
n
g

w
o
r
d
p
l
a
y
.










T
h
i
s

d
e
m
o
n
s
t
r
a
t
e
s

s
e
l
e
c
t
i
n
g

c
o
n
t
e
x
t

f
r
o
m

s
t
a
t
e

-

w
e

r
e
a
d

t
h
e

e
x
i
s
t
i
n
g





j
o
k
e

f
r
o
m

s
t
a
t
e

a
n
d

u
s
e

i
t

t
o

g
e
n
e
r
a
t
e

a
n

i
m
p
r
o
v
e
d

v
e
r
s
i
o
n
.










A
r
g
s
:









s
t
a
t
e
:

C
u
r
r
e
n
t

s
t
a
t
e

c
o
n
t
a
i
n
i
n
g

t
h
e

o
r
i
g
i
n
a
l

j
o
k
e














R
e
t
u
r
n
s
:









D
i
c
t
i
o
n
a
r
y

w
i
t
h

t
h
e

i
m
p
r
o
v
e
d

j
o
k
e





"
"
"





p
r
i
n
t
(
f
"
I
n
i
t
i
a
l

j
o
k
e
:

{
s
t
a
t
e
[
'
j
o
k
e
'
]
}
"
)











S
e
l
e
c
t

t
h
e

j
o
k
e

f
r
o
m

s
t
a
t
e

t
o

p
r
e
s
e
n
t

i
t

t
o

t
h
e

L
L
M





m
s
g

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
f
"
M
a
k
e

t
h
i
s

j
o
k
e

f
u
n
n
i
e
r

b
y

a
d
d
i
n
g

w
o
r
d
p
l
a
y
:

{
s
t
a
t
e
[
'
j
o
k
e
'
]
}
"
)





r
e
t
u
r
n

{
"
i
m
p
r
o
v
e
d
_
j
o
k
e
"
:

m
s
g
.
c
o
n
t
e
n
t
}




B
u
i
l
d

t
h
e

w
o
r
k
f
l
o
w

w
i
t
h

t
w
o

s
e
q
u
e
n
t
i
a
l

n
o
d
e
s

w
o
r
k
f
l
o
w

=

S
t
a
t
e
G
r
a
p
h
(
S
t
a
t
e
)



A
d
d

b
o
t
h

j
o
k
e

g
e
n
e
r
a
t
i
o
n

n
o
d
e
s

w
o
r
k
f
l
o
w
.
a
d
d
_
n
o
d
e
(
"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
,

g
e
n
e
r
a
t
e
_
j
o
k
e
)

w
o
r
k
f
l
o
w
.
a
d
d
_
n
o
d
e
(
"
i
m
p
r
o
v
e
_
j
o
k
e
"
,

i
m
p
r
o
v
e
_
j
o
k
e
)



C
o
n
n
e
c
t

n
o
d
e
s

i
n

s
e
q
u
e
n
c
e

w
o
r
k
f
l
o
w
.
a
d
d
_
e
d
g
e
(
S
T
A
R
T
,

"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
)

w
o
r
k
f
l
o
w
.
a
d
d
_
e
d
g
e
(
"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
,

"
i
m
p
r
o
v
e
_
j
o
k
e
"
)

w
o
r
k
f
l
o
w
.
a
d
d
_
e
d
g
e
(
"
i
m
p
r
o
v
e
_
j
o
k
e
"
,

E
N
D
)



C
o
m
p
i
l
e

t
h
e

w
o
r
k
f
l
o
w

c
h
a
i
n

=

w
o
r
k
f
l
o
w
.
c
o
m
p
i
l
e
(
)



D
i
s
p
l
a
y

t
h
e

w
o
r
k
f
l
o
w

v
i
s
u
a
l
i
z
a
t
i
o
n

d
i
s
p
l
a
y
(
I
m
a
g
e
(
c
h
a
i
n
.
g
e
t
_
g
r
a
p
h
(
)
.
d
r
a
w
_
m
e
r
m
a
i
d
_
p
n
g
(
)
)
)

E
x
e
c
u
t
e

t
h
e

w
o
r
k
f
l
o
w

t
o

s
e
e

c
o
n
t
e
x
t

s
e
l
e
c
t
i
o
n

i
n

a
c
t
i
o
n

j
o
k
e
_
g
e
n
e
r
a
t
o
r
_
s
t
a
t
e

=

c
h
a
i
n
.
i
n
v
o
k
e
(
{
"
t
o
p
i
c
"
:

"
c
a
t
s
"
}
)



D
i
s
p
l
a
y

t
h
e

f
i
n
a
l

s
t
a
t
e

w
i
t
h

r
i
c
h

f
o
r
m
a
t
t
i
n
g

c
o
n
s
o
l
e
.
p
r
i
n
t
(
"
\
n
[
b
o
l
d

b
l
u
e
]
F
i
n
a
l

W
o
r
k
f
l
o
w

S
t
a
t
e
:
[
/
b
o
l
d

b
l
u
e
]
"
)

p
p
r
i
n
t
(
j
o
k
e
_
g
e
n
e
r
a
t
o
r
_
s
t
a
t
e
)

"""
## Memory

If agents have the ability to save memories, they also need the ability to select memories relevant to the task they are performing. This can be useful for a few reasons. Agents might select few-shot examples ([episodic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) for examples of desired behavior, instructions ([procedural](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) to steer behavior, or facts ([semantic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) [memories](https://arxiv.org/pdf/2309.02427)) give the agent task-relevant context.

![image (1).webp](attachment:2fc5dc77-8eba-4a80-8e38-ad00688adc3c.webp)

One challenge is ensure that relevant memories are selected. Some popular agents simply use a narrow set of files to store memories. For example, many code agent use “rules” files to save instructions (”procedural” memories) or, in some cases, examples (”episodic” memories). Claude Code uses [`CLAUDE.md`](http://CLAUDE.md). [Cursor](https://docs.cursor.com/context/rules) and [Windsurf](https://windsurf.com/editor/directory) use rules files. These are always pulled into context.

But, if an agent is storing a larger [collection](https://langchain-ai.github.io/langgraph/concepts/memory/#collection) of facts and / or relationships ([semantic](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) memories), selection is harder. [ChatGPT](https://help.ollama.com/en/articles/8590148-memory-faq) is a good example of this. At the AIEngineer World’s Fair, [Simon Willison shared](https://simonwillison.net/2025/Jun/6/six-months-in-llms/) a good example of memory selection gone wrong: ChatGPT fetched his location and injected it into an image that he requested. This type of erroneous memory retrieval can make users feel like the context winder “*no longer belongs to them*”! Use of embeddings and / or [knowledge](https://arxiv.org/html/2501.13956v1#:~:text=In%20Zep%2C%20memory%20is%20powered,subgraph%2C%20and%20a%20community%20subgraph) [graphs](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/#:~:text=changes%20since%20updates%20can%20trigger,and%20holistic%20memory%20for%20agentic) for indexing of memories have been used to assist with selection.

### Memory selecting in LangGraph

In `1_write_context.ipynb`, we saw how to write to `InMemoryStore` in graph nodes. Now let's select state from it. We can use the [get](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-storage) method to select context from state.
"""
logger.info("## Memory")

f
r
o
m

l
a
n
g
g
r
a
p
h
.
s
t
o
r
e
.
m
e
m
o
r
y

i
m
p
o
r
t

I
n
M
e
m
o
r
y
S
t
o
r
e



I
n
i
t
i
a
l
i
z
e

t
h
e

m
e
m
o
r
y

s
t
o
r
e

s
t
o
r
e

=

I
n
M
e
m
o
r
y
S
t
o
r
e
(
)



D
e
f
i
n
e

n
a
m
e
s
p
a
c
e

f
o
r

o
r
g
a
n
i
z
i
n
g

m
e
m
o
r
i
e
s

n
a
m
e
s
p
a
c
e

=

(
"
r
l
m
"
,

"
j
o
k
e
_
g
e
n
e
r
a
t
o
r
"
)



S
t
o
r
e

t
h
e

g
e
n
e
r
a
t
e
d

j
o
k
e

i
n

m
e
m
o
r
y

s
t
o
r
e
.
p
u
t
(





n
a
m
e
s
p
a
c
e
,






























n
a
m
e
s
p
a
c
e

f
o
r

o
r
g
a
n
i
z
a
t
i
o
n





"
l
a
s
t
_
j
o
k
e
"
,



























k
e
y

i
d
e
n
t
i
f
i
e
r





{
"
j
o
k
e
"
:

j
o
k
e
_
g
e
n
e
r
a
t
o
r
_
s
t
a
t
e
[
"
j
o
k
e
"
]
}


v
a
l
u
e

t
o

s
t
o
r
e

)



S
e
l
e
c
t

(
r
e
t
r
i
e
v
e
)

t
h
e

j
o
k
e

f
r
o
m

m
e
m
o
r
y

r
e
t
r
i
e
v
e
d
_
j
o
k
e

=

s
t
o
r
e
.
g
e
t
(
n
a
m
e
s
p
a
c
e
,

"
l
a
s
t
_
j
o
k
e
"
)
.
v
a
l
u
e



D
i
s
p
l
a
y

t
h
e

r
e
t
r
i
e
v
e
d

c
o
n
t
e
x
t

c
o
n
s
o
l
e
.
p
r
i
n
t
(
"
\
n
[
b
o
l
d

g
r
e
e
n
]
R
e
t
r
i
e
v
e
d

C
o
n
t
e
x
t

f
r
o
m

M
e
m
o
r
y
:
[
/
b
o
l
d

g
r
e
e
n
]
"
)

p
p
r
i
n
t
(
r
e
t
r
i
e
v
e
d
_
j
o
k
e
)

f
r
o
m

l
a
n
g
g
r
a
p
h
.
c
h
e
c
k
p
o
i
n
t
.
m
e
m
o
r
y

i
m
p
o
r
t

I
n
M
e
m
o
r
y
S
a
v
e
r

f
r
o
m

l
a
n
g
g
r
a
p
h
.
s
t
o
r
e
.
b
a
s
e

i
m
p
o
r
t

B
a
s
e
S
t
o
r
e

f
r
o
m

l
a
n
g
g
r
a
p
h
.
s
t
o
r
e
.
m
e
m
o
r
y

i
m
p
o
r
t

I
n
M
e
m
o
r
y
S
t
o
r
e



I
n
i
t
i
a
l
i
z
e

s
t
o
r
a
g
e

c
o
m
p
o
n
e
n
t
s

c
h
e
c
k
p
o
i
n
t
e
r

=

I
n
M
e
m
o
r
y
S
a
v
e
r
(
)

m
e
m
o
r
y
_
s
t
o
r
e

=

I
n
M
e
m
o
r
y
S
t
o
r
e
(
)



d
e
f

g
e
n
e
r
a
t
e
_
j
o
k
e
(
s
t
a
t
e
:

S
t
a
t
e
,

s
t
o
r
e
:

B
a
s
e
S
t
o
r
e
)

-
>

d
i
c
t
[
s
t
r
,

s
t
r
]
:





"
"
"
G
e
n
e
r
a
t
e

a

j
o
k
e

w
i
t
h

m
e
m
o
r
y
-
a
w
a
r
e

c
o
n
t
e
x
t

s
e
l
e
c
t
i
o
n
.










T
h
i
s

f
u
n
c
t
i
o
n

d
e
m
o
n
s
t
r
a
t
e
s

s
e
l
e
c
t
i
n
g

c
o
n
t
e
x
t

f
r
o
m

m
e
m
o
r
y

b
e
f
o
r
e





g
e
n
e
r
a
t
i
n
g

n
e
w

c
o
n
t
e
n
t
,

e
n
s
u
r
i
n
g

c
o
n
s
i
s
t
e
n
c
y

a
n
d

a
v
o
i
d
i
n
g

d
u
p
l
i
c
a
t
i
o
n
.










A
r
g
s
:









s
t
a
t
e
:

C
u
r
r
e
n
t

s
t
a
t
e

c
o
n
t
a
i
n
i
n
g

t
h
e

t
o
p
i
c









s
t
o
r
e
:

M
e
m
o
r
y

s
t
o
r
e

f
o
r

p
e
r
s
i
s
t
e
n
t

c
o
n
t
e
x
t














R
e
t
u
r
n
s
:









D
i
c
t
i
o
n
a
r
y

w
i
t
h

t
h
e

g
e
n
e
r
a
t
e
d

j
o
k
e





"
"
"






S
e
l
e
c
t

p
r
i
o
r

j
o
k
e

f
r
o
m

m
e
m
o
r
y

i
f

i
t

e
x
i
s
t
s





p
r
i
o
r
_
j
o
k
e

=

s
t
o
r
e
.
g
e
t
(
n
a
m
e
s
p
a
c
e
,

"
l
a
s
t
_
j
o
k
e
"
)





i
f

p
r
i
o
r
_
j
o
k
e
:









p
r
i
o
r
_
j
o
k
e
_
t
e
x
t

=

p
r
i
o
r
_
j
o
k
e
.
v
a
l
u
e
[
"
j
o
k
e
"
]









p
r
i
n
t
(
f
"
P
r
i
o
r

j
o
k
e
:

{
p
r
i
o
r
_
j
o
k
e
_
t
e
x
t
}
"
)





e
l
s
e
:









p
r
i
n
t
(
"
P
r
i
o
r

j
o
k
e
:

N
o
n
e
# !
"
)







G
e
n
e
r
a
t
e

a

n
e
w

j
o
k
e

t
h
a
t

d
i
f
f
e
r
s

f
r
o
m

t
h
e

p
r
i
o
r

o
n
e





p
r
o
m
p
t

=

(









f
"
W
r
i
t
e

a

s
h
o
r
t

j
o
k
e

a
b
o
u
t

{
s
t
a
t
e
[
'
t
o
p
i
c
'
]
}
,

"









f
"
b
u
t

m
a
k
e

i
t

d
i
f
f
e
r
e
n
t

f
r
o
m

a
n
y

p
r
i
o
r

j
o
k
e

y
o
u
'
v
e

w
r
i
t
t
e
n
:

{
p
r
i
o
r
_
j
o
k
e
_
t
e
x
t

i
f

p
r
i
o
r
_
j
o
k
e

e
l
s
e

'
N
o
n
e
'
}
"





)





m
s
g

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
p
r
o
m
p
t
)







S
t
o
r
e

t
h
e

n
e
w

j
o
k
e

i
n

m
e
m
o
r
y

f
o
r

f
u
t
u
r
e

c
o
n
t
e
x
t

s
e
l
e
c
t
i
o
n





s
t
o
r
e
.
p
u
t
(
n
a
m
e
s
p
a
c
e
,

"
l
a
s
t
_
j
o
k
e
"
,

{
"
j
o
k
e
"
:

m
s
g
.
c
o
n
t
e
n
t
}
)






r
e
t
u
r
n

{
"
j
o
k
e
"
:

m
s
g
.
c
o
n
t
e
n
t
}




B
u
i
l
d

t
h
e

m
e
m
o
r
y
-
a
w
a
r
e

w
o
r
k
f
l
o
w

w
o
r
k
f
l
o
w

=

S
t
a
t
e
G
r
a
p
h
(
S
t
a
t
e
)

w
o
r
k
f
l
o
w
.
a
d
d
_
n
o
d
e
(
"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
,

g
e
n
e
r
a
t
e
_
j
o
k
e
)



C
o
n
n
e
c
t

t
h
e

w
o
r
k
f
l
o
w

w
o
r
k
f
l
o
w
.
a
d
d
_
e
d
g
e
(
S
T
A
R
T
,

"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
)

w
o
r
k
f
l
o
w
.
a
d
d
_
e
d
g
e
(
"
g
e
n
e
r
a
t
e
_
j
o
k
e
"
,

E
N
D
)



C
o
m
p
i
l
e

w
i
t
h

b
o
t
h

c
h
e
c
k
p
o
i
n
t
i
n
g

a
n
d

m
e
m
o
r
y

s
t
o
r
e

c
h
a
i
n

=

w
o
r
k
f
l
o
w
.
c
o
m
p
i
l
e
(
c
h
e
c
k
p
o
i
n
t
e
r
=
c
h
e
c
k
p
o
i
n
t
e
r
,

s
t
o
r
e
=
m
e
m
o
r
y
_
s
t
o
r
e
)

E
x
e
c
u
t
e

t
h
e

w
o
r
k
f
l
o
w

w
i
t
h

t
h
e

f
i
r
s
t

t
h
r
e
a
d

c
o
n
f
i
g

=

{
"
c
o
n
f
i
g
u
r
a
b
l
e
"
:

{
"
t
h
r
e
a
d
_
i
d
"
:

"
1
"
}
}

j
o
k
e
_
g
e
n
e
r
a
t
o
r
_
s
t
a
t
e

=

c
h
a
i
n
.
i
n
v
o
k
e
(
{
"
t
o
p
i
c
"
:

"
c
a
t
s
"
}
,

c
o
n
f
i
g
)

latest_state = chain.get_state(config)

console.logger.debug("\n[bold magenta]Latest Graph State:[/bold magenta]")
plogger.debug(latest_state)

"""
We fetch the prior joke from memory and pass it to an LLM to improve it!
"""
logger.info("We fetch the prior joke from memory and pass it to an LLM to improve it!")

E
x
e
c
u
t
e

t
h
e

w
o
r
k
f
l
o
w

w
i
t
h

a

s
e
c
o
n
d

t
h
r
e
a
d

t
o

d
e
m
o
n
s
t
r
a
t
e

m
e
m
o
r
y

p
e
r
s
i
s
t
e
n
c
e

c
o
n
f
i
g

=

{
"
c
o
n
f
i
g
u
r
a
b
l
e
"
:

{
"
t
h
r
e
a
d
_
i
d
"
:

"
2
"
}
}

j
o
k
e
_
g
e
n
e
r
a
t
o
r
_
s
t
a
t
e

=

c
h
a
i
n
.
i
n
v
o
k
e
(
{
"
t
o
p
i
c
"
:

"
c
a
t
s
"
}
,

c
o
n
f
i
g
)

"""
## Tools

Agents use tools, but can become overloaded if they are provided with too many. This is often because the tool descriptions can overlap, causing model confusion about which tool to use. One approach is to apply RAG to tool descriptions in order to fetch the most relevant tools for a task based upon semantic similarity, an idea that Drew Breunig calls “[tool loadout](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html).” Some [recent papers](https://arxiv.org/abs/2505.03275) have shown that this improve tool selection accuracy by 3-fold.

### Tool selecting in LangGraph

For tool selection, the [LangGraph Bigtool](https://github.com/langchain-ai/langgraph-bigtool) library is a great way to apply semantic similarity search over tool descriptions for selection of the most relevant tools for a task. It leverages LangGraph's long-term memory store to allow an agent to search for and retrieve relevant tools for a given problem. Lets demonstrate `langgraph-bigtool` by equipping an agent with all functions from Python's built-in math library.
"""
logger.info("## Tools")




# _set_env("OPENAI_API_KEY")

all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)

tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

embeddings = init_embeddings("ollama:nomic-embed-text")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)
agent

query = "Use available tools to calculate arc cosine of 0.5."
result = agent.invoke({"messages": query})
format_messages(result['messages'])

"""
### Learn more

* **Toolshed: Scale Tool-Equipped Agents with Advanced RAG-Tool Fusion** - Lumer, E., Subbiah, V.K., Burke, J.A., Basavaraju, P.H. & Huber, A. (2024). arXiv:2410.14594.

The paper introduces Toolshed Knowledge Bases and Advanced RAG-Tool Fusion to address challenges in scaling tool-equipped AI agents. The Toolshed Knowledge Base is a vector database designed to store enhanced tool representations and optimize tool selection for large-scale tool-equipped agents. The Advanced RAG-Tool Fusion technique applies retrieval-augmented generation across three phases: pre-retrieval (tool document enhancement), intra-retrieval (query planning and transformation), and post-retrieval (document refinement and self-reflection). The researchers demonstrated significant performance improvements, achieving 46%, 56%, and 47% absolute improvements on different benchmark datasets (Recall@5), all without requiring model fine-tuning.

* **Graph RAG-Tool Fusion** - Lumer, E., Basavaraju, P.H., Mason, M., Burke, J.A. & Subbiah, V.K. (2025). arXiv:2502.07223.

This paper addresses limitations in current RAG approaches for tool selection by introducing Graph RAG-Tool Fusion, which combines vector-based retrieval with graph traversal to capture tool dependencies. Traditional RAG methods fail to capture structured dependencies between tools (e.g., a "get stock price" API requiring a "stock ticker" parameter from another API). The authors present ToolLinkOS, a benchmark dataset with 573 fictional tools across 15 industries, each averaging 6.3 tool dependencies. Graph RAG-Tool Fusion achieved absolute improvements of 71.7% and 22.1% over naïve RAG on ToolLinkOS and ToolSandbox benchmarks, respectively, by understanding and navigating interconnected tool relationships within a predefined knowledge graph.

* **LLM-Tool-Survey** - https://github.com/quchangle1/LLM-Tool-Survey

This comprehensive survey repository explores Tool Learning with Large Language Models, presenting a systematic examination of how AI models can effectively use external tools to enhance their capabilities. The repository covers key aspects including benefits of tools (knowledge acquisition, expertise enhancement, interaction improvement) and technical workflows. It provides an extensive collection of research papers categorized by tool types, reasoning methods, and technological approaches, ranging from mathematical tools and programming interpreters to multi-modal and domain-specific applications. The repository serves as a valuable collaborative resource for researchers and practitioners interested in the evolving landscape of AI tool integration.

* **Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval** - Shi, Z., Wang, Y., Yan, L., Ren, P., Wang, S., Yin, D. & Ren, Z. arXiv:2503.01763.

The paper introduces ToolRet, a benchmark for evaluating tool retrieval capabilities of information retrieval (IR) models in LLM contexts. Unlike existing benchmarks that manually pre-annotate small sets of relevant tools, ToolRet comprises 7.6k diverse retrieval tasks and a corpus of 43k tools from existing datasets. The research found that even IR models with strong performance in conventional benchmarks exhibit poor performance on ToolRet, directly impacting task success rates of tool-using LLMs. As a solution, the researchers contributed a large-scale training dataset with over 200k instances that substantially optimizes tool retrieval ability, bridging the gap between existing approaches and real-world tool-learning scenarios.

## Knowledge 

[RAG](https://github.com/langchain-ai/rag-from-scratch) (retrieval augmented generation) is an extremely rich topic. Code agents are some of the best examples of agentic RAG in large-scale production. [In practice, RAG is can be a central context engineering challenge](https://x.com/_mohansolo/status/1899630246862966837). Varun from Windsurf captures some of these challenges well:

> Indexing code ≠ context retrieval … [We are doing indexing & embedding search … [with] AST parsing code and chunking along semantically meaningful boundaries … embedding search becomes unreliable as a retrieval heuristic as the size of the codebase grows … we must rely on a combination of techniques like grep/file search, knowledge graph based retrieval, and … a re-ranking step where [context] is ranked in order of relevance. 

### RAG in LangGraph

There are several [tutorials and videos](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/) that show how to use RAG with LangGraph. When combining RAG with agents in LangGraph, it's common to build a retrieval tool. Note that this tool could incorporate any combination of RAG techniques, as mentioned above.

Fetch documents to use in our RAG system. We will use three of the most recent pages from Lilian Weng's excellent blog. We'll start by fetching the content of the pages using WebBaseLoader utility.
"""
logger.info("### Learn more")


urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

"""
Split the fetched documents into smaller chunks for indexing into our vectorstore.
"""
logger.info("Split the fetched documents into smaller chunks for indexing into our vectorstore.")


docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

"""
Now that we have our split documents, we can index them into a vector store that we'll use for semantic search.
"""
logger.info("Now that we have our split documents, we can index them into a vector store that we'll use for semantic search.")


vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=embeddings
)
retriever = vectorstore.as_retriever()

"""
Create a retriever tool that we can use in our agent.
"""
logger.info("Create a retriever tool that we can use in our agent.")


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

"""
Now, implement an agent that can select context from the tool.
"""
logger.info("Now, implement an agent that can select context from the tool.")

tools = [retriever_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


rag_prompt = """You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng.
Clarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and
proceed until you have sufficient context to answer the user's research request."""

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=rag_prompt
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

agent = agent_builder.compile()

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

query = "What are the types of reward hacking discussed in the blogs?"
result = agent.invoke({"messages": query})
format_messages(result['messages'])

logger.info("\n\n[DONE]", bright=True)