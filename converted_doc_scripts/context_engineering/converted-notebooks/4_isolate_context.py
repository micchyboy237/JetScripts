from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_sandbox import PyodideSandboxTool
from utils import format_messages
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Isolating Context

*Isolating context involves splitting it up to help an agent perform a task.*

![Screenshot 2025-07-09 at 2.28.19 PM.png](attachment:96e3f693-02e4-47c2-9f03-3a1c3146c84a.png)

## Multi-Agent

One of the most popular and intuitive ways to isolate context is to split it across sub-agents. A motivation for the Ollama [Swarm](https://github.com/ollama/swarm) library was “[separation of concerns](https://ollama.github.io/ollama-agents-python/ref/agent/)”, where a team of agents can handle sub-tasks. Each agent has a specific set of tools, instructions, and its own context window.

Ollama’s [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system) makes a clear case for the benefit of this: many agents with isolated contexts outperformed single-agent by 90.2%, largely because each subagent context window can be allocated to a more narrow sub-task. As the blog said:

> [Subagents operate] in parallel with their own context windows, exploring different aspects of the question simultaneously. 

![image (3).webp](attachment:21c82d0f-1baa-48c1-a13a-628b2782d836.webp)

Of course, the challenge with multi-agent include token use (e.g., [15× more tokens](https://www.anthropic.com/engineering/built-multi-agent-research-system) than chat), the need for careful [prompt engineering](https://www.anthropic.com/engineering/built-multi-agent-research-system) to plan sub-agent work, and coordination of sub-agents.

### Multi-Agent in LangGraph

LangGraph supports multi-agent systems. A popular and intuitive way to implement this is the [supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) architecture, which is what is used in Ollama's [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system). This allows the supervisor to delegate tasks to sub-agents, each with their own context window.
"""
logger.info("# Isolating Context")

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
p
r
e
b
u
i
l
t

i
m
p
o
r
t

c
r
e
a
t
e
_
r
e
a
c
t
_
a
g
e
n
t

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
_
s
u
p
e
r
v
i
s
o
r

i
m
p
o
r
t

c
r
e
a
t
e
_
s
u
p
e
r
v
i
s
o
r



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

l
a
n
g
u
a
g
e

m
o
d
e
l

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

a
d
d
(
a
:

f
l
o
a
t
,

b
:

f
l
o
a
t
)

-
>

f
l
o
a
t
:





"
"
"
A
d
d

t
w
o

n
u
m
b
e
r
s
.










A
r
g
s
:









a
:

F
i
r
s
t

n
u
m
b
e
r









b
:

S
e
c
o
n
d

n
u
m
b
e
r














R
e
t
u
r
n
s
:









S
u
m

o
f

a

a
n
d

b





"
"
"





r
e
t
u
r
n

a

+

b



d
e
f

m
u
l
t
i
p
l
y
(
a
:

f
l
o
a
t
,

b
:

f
l
o
a
t
)

-
>

f
l
o
a
t
:





"
"
"
M
u
l
t
i
p
l
y

t
w
o

n
u
m
b
e
r
s
.










A
r
g
s
:









a
:

F
i
r
s
t

n
u
m
b
e
r









b
:

S
e
c
o
n
d

n
u
m
b
e
r














R
e
t
u
r
n
s
:









P
r
o
d
u
c
t

o
f

a

a
n
d

b





"
"
"





r
e
t
u
r
n

a

*

b



d
e
f

w
e
b
_
s
e
a
r
c
h
(
q
u
e
r
y
:

s
t
r
)

-
>

s
t
r
:





"
"
"
M
o
c
k

w
e
b

s
e
a
r
c
h

f
u
n
c
t
i
o
n

t
h
a
t

r
e
t
u
r
n
s

F
A
A
N
G

c
o
m
p
a
n
y

h
e
a
d
c
o
u
n
t
s
.










A
r
g
s
:









q
u
e
r
y
:

S
e
a
r
c
h

q
u
e
r
y

(
u
n
u
s
e
d

i
n

t
h
i
s

m
o
c
k
)














R
e
t
u
r
n
s
:









S
t
a
t
i
c

i
n
f
o
r
m
a
t
i
o
n

a
b
o
u
t

F
A
A
N
G

c
o
m
p
a
n
y

h
e
a
d
c
o
u
n
t
s





"
"
"





r
e
t
u
r
n

(









"
H
e
r
e

a
r
e

t
h
e

h
e
a
d
c
o
u
n
t
s

f
o
r

e
a
c
h

o
f

t
h
e

F
A
A
N
G

c
o
m
p
a
n
i
e
s

i
n

2
0
2
4
:
\
n
"









"
1
.

*
*
F
a
c
e
b
o
o
k

(
M
e
t
a
)
*
*
:

6
7
,
3
1
7

e
m
p
l
o
y
e
e
s
.
\
n
"









"
2
.

*
*
A
p
p
l
e
*
*
:

1
6
4
,
0
0
0

e
m
p
l
o
y
e
e
s
.
\
n
"









"
3
.

*
*
A
m
a
z
o
n
*
*
:

1
,
5
5
1
,
0
0
0

e
m
p
l
o
y
e
e
s
.
\
n
"









"
4
.

*
*
N
e
t
f
l
i
x
*
*
:

1
4
,
0
0
0

e
m
p
l
o
y
e
e
s
.
\
n
"









"
5
.

*
*
G
o
o
g
l
e

(
A
l
p
h
a
b
e
t
)
*
*
:

1
8
1
,
2
6
9

e
m
p
l
o
y
e
e
s
.
"





)




C
r
e
a
t
e

s
p
e
c
i
a
l
i
z
e
d

a
g
e
n
t
s

w
i
t
h

i
s
o
l
a
t
e
d

c
o
n
t
e
x
t
s

m
a
t
h
_
a
g
e
n
t

=

c
r
e
a
t
e
_
r
e
a
c
t
_
a
g
e
n
t
(





m
o
d
e
l
=
l
l
m
,





t
o
o
l
s
=
[
a
d
d
,

m
u
l
t
i
p
l
y
]
,





n
a
m
e
=
"
m
a
t
h
_
e
x
p
e
r
t
"
,





p
r
o
m
p
t
=
"
Y
o
u

a
r
e

a

m
a
t
h

e
x
p
e
r
t
.

A
l
w
a
y
s

u
s
e

o
n
e

t
o
o
l

a
t

a

t
i
m
e
.
"

)


r
e
s
e
a
r
c
h
_
a
g
e
n
t

=

c
r
e
a
t
e
_
r
e
a
c
t
_
a
g
e
n
t
(





m
o
d
e
l
=
l
l
m
,





t
o
o
l
s
=
[
w
e
b
_
s
e
a
r
c
h
]
,





n
a
m
e
=
"
r
e
s
e
a
r
c
h
_
e
x
p
e
r
t
"
,





p
r
o
m
p
t
=
"
Y
o
u

a
r
e

a

w
o
r
l
d

c
l
a
s
s

r
e
s
e
a
r
c
h
e
r

w
i
t
h

a
c
c
e
s
s

t
o

w
e
b

s
e
a
r
c
h
.

D
o

n
o
t

d
o

a
n
y

m
a
t
h
.
"

)



C
r
e
a
t
e

s
u
p
e
r
v
i
s
o
r

w
o
r
k
f
l
o
w

f
o
r

c
o
o
r
d
i
n
a
t
i
n
g

a
g
e
n
t
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

c
r
e
a
t
e
_
s
u
p
e
r
v
i
s
o
r
(





[
r
e
s
e
a
r
c
h
_
a
g
e
n
t
,

m
a
t
h
_
a
g
e
n
t
]
,





m
o
d
e
l
=
l
l
m
,





p
r
o
m
p
t
=
(









"
Y
o
u

a
r
e

a

t
e
a
m

s
u
p
e
r
v
i
s
o
r

m
a
n
a
g
i
n
g

a

r
e
s
e
a
r
c
h

e
x
p
e
r
t

a
n
d

a

m
a
t
h

e
x
p
e
r
t
.

"









"
F
o
r

c
u
r
r
e
n
t

e
v
e
n
t
s
,

u
s
e

r
e
s
e
a
r
c
h
_
a
g
e
n
t
.

"









"
F
o
r

m
a
t
h

p
r
o
b
l
e
m
s
,

u
s
e

m
a
t
h
_
a
g
e
n
t
.
"





)

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

m
u
l
t
i
-
a
g
e
n
t

a
p
p
l
i
c
a
t
i
o
n

a
p
p

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

m
u
l
t
i
-
a
g
e
n
t

w
o
r
k
f
l
o
w

r
e
s
u
l
t

=

a
p
p
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
m
e
s
s
a
g
e
s
"
:

[









{













"
r
o
l
e
"
:

"
u
s
e
r
"
,













"
c
o
n
t
e
n
t
"
:

"
w
h
a
t
'
s

t
h
e

c
o
m
b
i
n
e
d

h
e
a
d
c
o
u
n
t

o
f

t
h
e

F
A
A
N
G

c
o
m
p
a
n
i
e
s

i
n

2
0
2
4
?
"









}





]

}
)

format_messages(result['messages'])

"""
### Learn more

* **LangGraph Swarm** - https://github.com/langchain-ai/langgraph-swarm-py

LangGraph Swarm is a Python library for creating multi-agent AI systems with dynamic collaboration capabilities. Key features include agents that can dynamically hand off control based on specialization while maintaining conversation context between transitions. The library supports customizable handoff tools between agents, streaming, short-term and long-term memory, and human-in-the-loop interactions. Built on the LangGraph framework, it enables creating flexible, context-aware multi-agent systems where different AI agents can collaborate and seamlessly transfer conversation control based on their unique capabilities. Installation is simple with `pip install langgraph-swarm`.

* [See](https://www.youtube.com/watch?v=4nZl32FwU-o) [these](https://www.youtube.com/watch?v=JeyDrn1dSUQ) [videos](https://www.youtube.com/watch?v=B_0TNuYi56w) for more detail on on multi-agent systems.

## Sandboxed Environment

HuggingFace’s [deep researcher](https://huggingface.co/blog/open-deep-research#:~:text=From%20building%20,it%20can%20still%20use%20it) shows another interesting example of context isolation. Most agents use [tool calling APIs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview), which return JSON objects (tool arguments) that can be passed to tools (e.g., a search API) to get tool feedback (e.g., search results). HuggingFace uses a [CodeAgent](https://huggingface.co/papers/2402.01030), which outputs code to invoke tools. The code then runs in a [sandbox](https://e2b.dev/). Selected context (e.g., return values) from code execution is then passed back to the LLM.

This allows context to be isolated in the environment, outside of the LLM context window. Hugging Face noted that this is a great way to isolate token-heavy objects from the LLM:

> [Code Agents allow for] a better handling of state … Need to store this image / audio / other for later use? No problem, just assign it as a variable in your state and you [use it later].

### Sandboxed Environment in LangGraph

It's pretty easy to use Sandboxes with LangGraph agents. [LangChain Sandbox](https://github.com/langchain-ai/langchain-sandbox) provides a secure environment for executing untrusted Python code. It leverages Pyodide (Python compiled to WebAssembly) to run Python code in a sandboxed environment. This can simply be used as a tool in a LangGraph agent.

> NOTE: Install Deno (required): https://docs.deno.com/runtime/getting_started/installation/
"""
logger.info("### Learn more")

tool = PyodideSandboxTool()
result = await tool.ainvoke("logger.debug('Hello, world!')")
logger.success(format_json(result))

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
_
s
a
n
d
b
o
x

i
m
p
o
r
t

P
y
o
d
i
d
e
S
a
n
d
b
o
x
T
o
o
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
p
r
e
b
u
i
l
t

i
m
p
o
r
t

c
r
e
a
t
e
_
r
e
a
c
t
_
a
g
e
n
t



C
r
e
a
t
e

s
a
n
d
b
o
x

t
o
o
l

w
i
t
h

n
e
t
w
o
r
k

a
c
c
e
s
s

f
o
r

p
a
c
k
a
g
e

i
n
s
t
a
l
l
a
t
i
o
n

t
o
o
l

=

P
y
o
d
i
d
e
S
a
n
d
b
o
x
T
o
o
l
(






A
l
l
o
w

P
y
o
d
i
d
e

t
o

i
n
s
t
a
l
l

P
y
t
h
o
n

p
a
c
k
a
g
e
s

t
h
a
t

m
i
g
h
t

b
e

r
e
q
u
i
r
e
d





a
l
l
o
w
_
n
e
t
=
T
r
u
e

)



C
r
e
a
t
e

a

R
e
a
c
t

a
g
e
n
t

w
i
t
h

t
h
e

s
a
n
d
b
o
x

t
o
o
l

a
g
e
n
t

=

c
r
e
a
t
e
_
r
e
a
c
t
_
a
g
e
n
t
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
3
-
7
-
s
o
n
n
e
t
-
l
a
t
e
s
t
"
,





t
o
o
l
s
=
[
t
o
o
l
]
,

)



E
x
e
c
u
t
e

a

m
a
t
h
e
m
a
t
i
c
a
l

q
u
e
r
y

u
s
i
n
g

t
h
e

s
a
n
d
b
o
x

r
e
s
u
l
t

=

a
w
a
i
t

a
g
e
n
t
.
a
i
n
v
o
k
e
(





{
"
m
e
s
s
a
g
e
s
"
:

[
{
"
r
o
l
e
"
:

"
u
s
e
r
"
,

"
c
o
n
t
e
n
t
"
:

"
w
h
a
t
'
s

5

+

7
?
"
}
]
}
,

)



F
o
r
m
a
t

a
n
d

d
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
s
u
l
t
s

f
o
r
m
a
t
_
m
e
s
s
a
g
e
s
(
r
e
s
u
l
t
[
'
m
e
s
s
a
g
e
s
'
]
)

"""
### State 

An agent’s runtime state object can also be a great way to isolate context. This can serve the same purpose as sandboxing. A state object can be designed with a schema (e.g., a Pydantic model) that has various fields that context can be written to. One field of the schema (e.g., messages) can be exposed to the LLM at each turn of the agent, but the schema can isolate information in other fields for more selective use. 

### State Isolation in LangGraph

LangGraph is designed around a [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) object, allowing you to design a state schema and access different fields of that schema across trajectory of your agent. For example, you can easily store context from tool calls in certain fields of your state object, isolating from the LLM until that context is required. In these notebooks, you've seen numerous example of this.
"""
logger.info("### State")

logger.info("\n\n[DONE]", bright=True)