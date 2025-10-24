from jet.logger import logger
from rich.markdown import Markdown
from utils import format_messages
from utils import format_messages, format_message
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
# Compressing Context in LangGraph

*Compressing context involves retaining only the tokens required to perform a task.*

![Screenshot 2025-07-09 at 2.28.10 PM.png](attachment:81bde857-21da-4464-b4ff-156e6f7d7079.png)

## Summarization 

Agent interactions can span [hundreds of turns](https://www.anthropic.com/engineering/built-multi-agent-research-system) and use token-heavy tool calls. Summarization is one common way to manage these challenges. If you’ve used Claude Code, you’ve seen this in action. Claude Code runs “[auto-compact](https://docs.anthropic.com/en/docs/claude-code/costs)” after you exceed 95% of the context window and it will summarize the full trajectory of user-agent interactions. This type of compression across an [agent trajectory](https://langchain-ai.github.io/langgraph/concepts/memory/#manage-short-term-memory) can use various strategies such as [recursive](https://arxiv.org/pdf/2308.15022#:~:text=the%20retrieved%20utterances%20capture%20the,based%203) or [hierarchical](https://alignment.anthropic.com/2025/summarization-for-monitoring/#:~:text=We%20addressed%20these%20issues%20by,of%20our%20computer%20use%20capability) summarization.

It can also be useful to [add summarization](https://github.com/langchain-ai/open_deep_research/blob/e5a5160a398a3699857d00d8569cb7fd0ac48a4f/src/open_deep_research/utils.py#L1407) at points in an agent’s trajectory. For example, it can be used to post-process certain tool calls (e.g., token-heavy search tools). As a second example, [Cognition](https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents) mentioned summarization at agent-agent boundaries to knowledge hand-off. They also the challenge if specific events or decisions to be captured. They use a fine-tuned model for this in Devin, which underscores how much work can go into this step.

![image (2).webp](attachment:756744cf-234a-4a7a-bd44-4a47801af657.webp)

### Summarization in LangGraph

Because LangGraph is a low [is a low-level orchestration framework](https://blog.langchain.com/how-to-think-about-agent-frameworks/), you can [lay out your agent as a set of nodes](https://www.youtube.com/watch?v=aHCDrAbH_go), [explicitly define](https://blog.langchain.com/how-to-think-about-agent-frameworks/) the logic within each one, and define an state object that is passed between them. This low-level control gives several ways to compress context.

You can use a message list as your agent state and [summarize](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory) using [a few built-in utilities](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory).

#### Summarize Messages

Let's implement a RAG agent, and add summarization of the conversation history.
"""
logger.info("# Compressing Context in LangGraph")

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
t
o
o
l
s
.
r
e
t
r
i
e
v
e
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
r
e
t
r
i
e
v
e
r
_
t
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
c
h
a
i
n
_
c
o
m
m
u
n
i
t
y
.
d
o
c
u
m
e
n
t
_
l
o
a
d
e
r
s

i
m
p
o
r
t

W
e
b
B
a
s
e
L
o
a
d
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
c
h
a
i
n
_
c
o
r
e
.
v
e
c
t
o
r
s
t
o
r
e
s

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
V
e
c
t
o
r
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
c
h
a
i
n
_
t
e
x
t
_
s
p
l
i
t
t
e
r
s

i
m
p
o
r
t

R
e
c
u
r
s
i
v
e
C
h
a
r
a
c
t
e
r
T
e
x
t
S
p
l
i
t
t
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
c
h
a
i
n
.
e
m
b
e
d
d
i
n
g
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
e
m
b
e
d
d
i
n
g
s



D
e
f
i
n
e

U
R
L
s

f
o
r

d
o
c
u
m
e
n
t

l
o
a
d
i
n
g

u
r
l
s

=

[





"
h
t
t
p
s
:
/
/
l
i
l
i
a
n
w
e
n
g
.
g
i
t
h
u
b
.
i
o
/
p
o
s
t
s
/
2
0
2
5
-
0
5
-
0
1
-
t
h
i
n
k
i
n
g
/
"
,





"
h
t
t
p
s
:
/
/
l
i
l
i
a
n
w
e
n
g
.
g
i
t
h
u
b
.
i
o
/
p
o
s
t
s
/
2
0
2
4
-
1
1
-
2
8
-
r
e
w
a
r
d
-
h
a
c
k
i
n
g
/
"
,





"
h
t
t
p
s
:
/
/
l
i
l
i
a
n
w
e
n
g
.
g
i
t
h
u
b
.
i
o
/
p
o
s
t
s
/
2
0
2
4
-
0
7
-
0
7
-
h
a
l
l
u
c
i
n
a
t
i
o
n
/
"
,





"
h
t
t
p
s
:
/
/
l
i
l
i
a
n
w
e
n
g
.
g
i
t
h
u
b
.
i
o
/
p
o
s
t
s
/
2
0
2
4
-
0
4
-
1
2
-
d
i
f
f
u
s
i
o
n
-
v
i
d
e
o
/
"
,

]



L
o
a
d

d
o
c
u
m
e
n
t
s

f
r
o
m

t
h
e

s
p
e
c
i
f
i
e
d

U
R
L
s

d
o
c
s

=

[
W
e
b
B
a
s
e
L
o
a
d
e
r
(
u
r
l
)
.
l
o
a
d
(
)

f
o
r

u
r
l

i
n

u
r
l
s
]

d
o
c
s
_
l
i
s
t

=

[
i
t
e
m

f
o
r

s
u
b
l
i
s
t

i
n

d
o
c
s

f
o
r

i
t
e
m

i
n

s
u
b
l
i
s
t
]



S
p
l
i
t

d
o
c
u
m
e
n
t
s

i
n
t
o

m
a
n
a
g
e
a
b
l
e

c
h
u
n
k
s

t
e
x
t
_
s
p
l
i
t
t
e
r

=

R
e
c
u
r
s
i
v
e
C
h
a
r
a
c
t
e
r
T
e
x
t
S
p
l
i
t
t
e
r
.
f
r
o
m
_
t
i
k
t
o
k
e
n
_
e
n
c
o
d
e
r
(





c
h
u
n
k
_
s
i
z
e
=
2
0
0
0
,





c
h
u
n
k
_
o
v
e
r
l
a
p
=
5
0

)

d
o
c
_
s
p
l
i
t
s

=

t
e
x
t
_
s
p
l
i
t
t
e
r
.
s
p
l
i
t
_
d
o
c
u
m
e
n
t
s
(
d
o
c
s
_
l
i
s
t
)



C
r
e
a
t
e

e
m
b
e
d
d
i
n
g
s

a
n
d

v
e
c
t
o
r
s
t
o
r
e

e
m
b
e
d
d
i
n
g
s

=

i
n
i
t
_
e
m
b
e
d
d
i
n
g
s
(
"
o
p
e
n
a
i
:
t
e
x
t
-
e
m
b
e
d
d
i
n
g
-
3
-
s
m
a
l
l
"
)

v
e
c
t
o
r
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
V
e
c
t
o
r
S
t
o
r
e
.
f
r
o
m
_
d
o
c
u
m
e
n
t
s
(





d
o
c
u
m
e
n
t
s
=
d
o
c
_
s
p
l
i
t
s
,






e
m
b
e
d
d
i
n
g
=
e
m
b
e
d
d
i
n
g
s

)

r
e
t
r
i
e
v
e
r

=

v
e
c
t
o
r
s
t
o
r
e
.
a
s
_
r
e
t
r
i
e
v
e
r
(
)



C
r
e
a
t
e

r
e
t
r
i
e
v
e
r

t
o
o
l

f
o
r

t
h
e

a
g
e
n
t

r
e
t
r
i
e
v
e
r
_
t
o
o
l

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
t
r
i
e
v
e
r
_
t
o
o
l
(





r
e
t
r
i
e
v
e
r
,





"
r
e
t
r
i
e
v
e
_
b
l
o
g
_
p
o
s
t
s
"
,





"
S
e
a
r
c
h

a
n
d

r
e
t
u
r
n

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

L
i
l
i
a
n

W
e
n
g

b
l
o
g

p
o
s
t
s
.
"
,

)



T
e
s
t

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
r

t
o
o
l

t
e
s
t
_
r
e
s
u
l
t

=

r
e
t
r
i
e
v
e
r
_
t
o
o
l
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
q
u
e
r
y
"
:

"
t
y
p
e
s

o
f

r
e
w
a
r
d

h
a
c
k
i
n
g
"
}
)

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



S
e
t

u
p

t
o
o
l
s

a
n
d

b
i
n
d

t
h
e
m

t
o

t
h
e

L
L
M

t
o
o
l
s

=

[
r
e
t
r
i
e
v
e
r
_
t
o
o
l
]

t
o
o
l
s
_
b
y
_
n
a
m
e

=

{
t
o
o
l
.
n
a
m
e
:

t
o
o
l

f
o
r

t
o
o
l

i
n

t
o
o
l
s
}



B
i
n
d

t
o
o
l
s

t
o

L
L
M

f
o
r

a
g
e
n
t

f
u
n
c
t
i
o
n
a
l
i
t
y

l
l
m
_
w
i
t
h
_
t
o
o
l
s

=

l
l
m
.
b
i
n
d
_
t
o
o
l
s
(
t
o
o
l
s
)

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
_
e
x
t
e
n
s
i
o
n
s

i
m
p
o
r
t

L
i
t
e
r
a
l


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
_
c
o
r
e
.
m
e
s
s
a
g
e
s

i
m
p
o
r
t

S
y
s
t
e
m
M
e
s
s
a
g
e
,

T
o
o
l
M
e
s
s
a
g
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

M
e
s
s
a
g
e
s
S
t
a
t
e
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




D
e
f
i
n
e

e
x
t
e
n
d
e
d

s
t
a
t
e

w
i
t
h

s
u
m
m
a
r
y

f
i
e
l
d

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
M
e
s
s
a
g
e
s
S
t
a
t
e
)
:





"
"
"
E
x
t
e
n
d
e
d

s
t
a
t
e

t
h
a
t

i
n
c
l
u
d
e
s

a

s
u
m
m
a
r
y

f
i
e
l
d

f
o
r

c
o
n
t
e
x
t

c
o
m
p
r
e
s
s
i
o
n
.
"
"
"





s
u
m
m
a
r
y
:

s
t
r




D
e
f
i
n
e

t
h
e

R
A
G

a
g
e
n
t

s
y
s
t
e
m

p
r
o
m
p
t

r
a
g
_
p
r
o
m
p
t

=

"
"
"
Y
o
u

a
r
e

a

h
e
l
p
f
u
l

a
s
s
i
s
t
a
n
t

t
a
s
k
e
d

w
i
t
h

r
e
t
r
i
e
v
i
n
g

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

f
r
o
m

a

s
e
r
i
e
s

o
f

t
e
c
h
n
i
c
a
l

b
l
o
g

p
o
s
t
s

b
y

L
i
l
i
a
n

W
e
n
g
.


C
l
a
r
i
f
y

t
h
e

s
c
o
p
e

o
f

r
e
s
e
a
r
c
h

w
i
t
h

t
h
e

u
s
e
r

b
e
f
o
r
e

u
s
i
n
g

y
o
u
r

r
e
t
r
i
e
v
a
l

t
o
o
l

t
o

g
a
t
h
e
r

c
o
n
t
e
x
t
.

R
e
f
l
e
c
t

o
n

a
n
y

c
o
n
t
e
x
t

y
o
u

f
e
t
c
h
,

a
n
d

p
r
o
c
e
e
d

u
n
t
i
l

y
o
u

h
a
v
e

s
u
f
f
i
c
i
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

t
o

a
n
s
w
e
r

t
h
e

u
s
e
r
'
s

r
e
s
e
a
r
c
h

r
e
q
u
e
s
t
.
"
"
"



D
e
f
i
n
e

t
h
e

s
u
m
m
a
r
i
z
a
t
i
o
n

p
r
o
m
p
t

s
u
m
m
a
r
i
z
a
t
i
o
n
_
p
r
o
m
p
t

=

"
"
"
S
u
m
m
a
r
i
z
e

t
h
e

f
u
l
l

c
h
a
t

h
i
s
t
o
r
y

a
n
d

a
l
l

t
o
o
l

f
e
e
d
b
a
c
k

t
o


g
i
v
e

a
n

o
v
e
r
v
i
e
w

o
f

w
h
a
t

t
h
e

u
s
e
r

a
s
k
e
d

a
b
o
u
t

a
n
d

w
h
a
t

t
h
e

a
g
e
n
t

d
i
d
.
"
"
"



d
e
f

l
l
m
_
c
a
l
l
(
s
t
a
t
e
:

M
e
s
s
a
g
e
s
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
:





"
"
"
E
x
e
c
u
t
e

L
L
M

c
a
l
l

w
i
t
h

s
y
s
t
e
m

p
r
o
m
p
t

a
n
d

m
e
s
s
a
g
e

h
i
s
t
o
r
y
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

c
o
n
v
e
r
s
a
t
i
o
n

s
t
a
t
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

n
e
w

m
e
s
s
a
g
e
s





"
"
"





m
e
s
s
a
g
e
s

=

[
S
y
s
t
e
m
M
e
s
s
a
g
e
(
c
o
n
t
e
n
t
=
r
a
g
_
p
r
o
m
p
t
)
]

+

s
t
a
t
e
[
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
]





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
_
w
i
t
h
_
t
o
o
l
s
.
i
n
v
o
k
e
(
m
e
s
s
a
g
e
s
)





r
e
t
u
r
n

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
r
e
s
p
o
n
s
e
]
}



d
e
f

t
o
o
l
_
n
o
d
e
(
s
t
a
t
e
:

M
e
s
s
a
g
e
s
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
:





"
"
"
E
x
e
c
u
t
e

t
o
o
l

c
a
l
l
s

a
n
d

r
e
t
u
r
n

r
e
s
u
l
t
s
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

c
o
n
v
e
r
s
a
t
i
o
n

s
t
a
t
e

w
i
t
h

t
o
o
l

c
a
l
l
s














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
o
o
l

r
e
s
u
l
t
s





"
"
"





r
e
s
u
l
t

=

[
]





f
o
r

t
o
o
l
_
c
a
l
l

i
n

s
t
a
t
e
[
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
]
[
-
1
]
.
t
o
o
l
_
c
a
l
l
s
:









t
o
o
l

=

t
o
o
l
s
_
b
y
_
n
a
m
e
[
t
o
o
l
_
c
a
l
l
[
"
n
a
m
e
"
]
]









o
b
s
e
r
v
a
t
i
o
n

=

t
o
o
l
.
i
n
v
o
k
e
(
t
o
o
l
_
c
a
l
l
[
"
a
r
g
s
"
]
)









r
e
s
u
l
t
.
a
p
p
e
n
d
(
T
o
o
l
M
e
s
s
a
g
e
(
c
o
n
t
e
n
t
=
o
b
s
e
r
v
a
t
i
o
n
,

t
o
o
l
_
c
a
l
l
_
i
d
=
t
o
o
l
_
c
a
l
l
[
"
i
d
"
]
)
)





r
e
t
u
r
n

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

r
e
s
u
l
t
}



d
e
f

s
u
m
m
a
r
y
_
n
o
d
e
(
s
t
a
t
e
:

M
e
s
s
a
g
e
s
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

s
u
m
m
a
r
y

o
f

t
h
e

c
o
n
v
e
r
s
a
t
i
o
n

a
n
d

t
o
o
l

i
n
t
e
r
a
c
t
i
o
n
s
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

c
o
n
v
e
r
s
a
t
i
o
n

s
t
a
t
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

c
o
n
v
e
r
s
a
t
i
o
n

s
u
m
m
a
r
y





"
"
"





m
e
s
s
a
g
e
s

=

[
S
y
s
t
e
m
M
e
s
s
a
g
e
(
c
o
n
t
e
n
t
=
s
u
m
m
a
r
i
z
a
t
i
o
n
_
p
r
o
m
p
t
)
]

+

s
t
a
t
e
[
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
]





r
e
s
u
l
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
m
e
s
s
a
g
e
s
)





r
e
t
u
r
n

{
"
s
u
m
m
a
r
y
"
:

r
e
s
u
l
t
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

s
h
o
u
l
d
_
c
o
n
t
i
n
u
e
(
s
t
a
t
e
:

M
e
s
s
a
g
e
s
S
t
a
t
e
)

-
>

L
i
t
e
r
a
l
[
"
A
c
t
i
o
n
"
,

"
s
u
m
m
a
r
y
_
n
o
d
e
"
]
:





"
"
"
D
e
t
e
r
m
i
n
e

n
e
x
t

s
t
e
p

b
a
s
e
d

o
n

w
h
e
t
h
e
r

L
L
M

m
a
d
e

t
o
o
l

c
a
l
l
s
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

c
o
n
v
e
r
s
a
t
i
o
n

s
t
a
t
e














R
e
t
u
r
n
s
:









N
e
x
t

n
o
d
e

t
o

e
x
e
c
u
t
e





"
"
"





m
e
s
s
a
g
e
s

=

s
t
a
t
e
[
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
]





l
a
s
t
_
m
e
s
s
a
g
e

=

m
e
s
s
a
g
e
s
[
-
1
]











I
f

L
L
M

m
a
d
e

t
o
o
l

c
a
l
l
s
,

e
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





i
f

l
a
s
t
_
m
e
s
s
a
g
e
.
t
o
o
l
_
c
a
l
l
s
:









r
e
t
u
r
n

"
A
c
t
i
o
n
"






O
t
h
e
r
w
i
s
e
,

p
r
o
c
e
e
d

t
o

s
u
m
m
a
r
i
z
a
t
i
o
n





r
e
t
u
r
n

"
s
u
m
m
a
r
y
_
n
o
d
e
"




B
u
i
l
d

t
h
e

R
A
G

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

a
g
e
n
t
_
b
u
i
l
d
e
r

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

n
o
d
e
s

t
o

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

a
g
e
n
t
_
b
u
i
l
d
e
r
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
l
l
m
_
c
a
l
l
"
,

l
l
m
_
c
a
l
l
)

a
g
e
n
t
_
b
u
i
l
d
e
r
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
"
,

t
o
o
l
_
n
o
d
e
)

a
g
e
n
t
_
b
u
i
l
d
e
r
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
s
u
m
m
a
r
y
_
n
o
d
e
"
,

s
u
m
m
a
r
y
_
n
o
d
e
)



D
e
f
i
n
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

e
d
g
e
s

a
g
e
n
t
_
b
u
i
l
d
e
r
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
l
l
m
_
c
a
l
l
"
)

a
g
e
n
t
_
b
u
i
l
d
e
r
.
a
d
d
_
c
o
n
d
i
t
i
o
n
a
l
_
e
d
g
e
s
(





"
l
l
m
_
c
a
l
l
"
,





s
h
o
u
l
d
_
c
o
n
t
i
n
u
e
,





{









"
A
c
t
i
o
n
"
:

"
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
"
,









"
s
u
m
m
a
r
y
_
n
o
d
e
"
:

"
s
u
m
m
a
r
y
_
n
o
d
e
"
,





}
,

)

a
g
e
n
t
_
b
u
i
l
d
e
r
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
"
,

"
l
l
m
_
c
a
l
l
"
)

a
g
e
n
t
_
b
u
i
l
d
e
r
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
s
u
m
m
a
r
y
_
n
o
d
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

a
g
e
n
t

a
g
e
n
t

=

a
g
e
n
t
_
b
u
i
l
d
e
r
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
a
g
e
n
t
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
x
r
a
y
=
T
r
u
e
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


query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": query})
format_message(result['messages'])

Markdown(result["summary"])

"""
Nice, but it uses `115k tokens`!

See trace:  

https://smith.langchain.com/public/50d70503-1a8e-46c1-bbba-a1efb8626b05/r

This is often a challenge with agents that have token-heavy tool calls!

#### Summarize Tools

Let's update the RAG agent, and add summarization the tool call output.
"""
logger.info("#### Summarize Tools")

tool_summarization_prompt = """You will be provided a doc from a RAG system.
Summarize the docs, ensuring to retain all relevant / essential information.
Your goal is simply to reduce the size of the doc (tokens) to a more manageable size."""

def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "Action"
    return END

def tool_node_with_summarization(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        summary = llm.invoke([{"role":"system",
                              "content":tool_summarization_prompt},
                              {"role":"user",
                               "content":observation}])
        result.append(ToolMessage(content=summary.content, tool_call_id=tool_call["id"]))
    return {"messages": result}

agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment_with_summarization", tool_node_with_summarization)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment_with_summarization",
        END: END,
    },
)
agent_builder.add_edge("environment_with_summarization", "llm_call")

agent = agent_builder.compile()

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


query = "Why does RL improve LLM reasoning according to the blogs?"
result = agent.invoke({"messages": query})
format_messages(result['messages'])

"""
This uses 60k tokens. 

https://smith.langchain.com/public/994cdf93-e837-4708-9628-c83b397dd4b5/r

#### Learn More

* **Heuristic Compression and Message Trimming** - https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#trim-messages

LangGraph provides several message management strategies to handle context window limitations. The `trim_messages()` function allows you to limit token count by keeping the "last" messages and controlling maximum tokens and message boundaries. This can be implemented as pre-model hooks for agents with custom state management. Key benefits include preventing context window overflow, maintaining conversation context, optimizing memory usage, and enabling long-running conversations. The approach emphasizes flexible, programmatic management of conversational memory across different AI interaction scenarios.

* **SummarizationNode as Pre-Model Hook** - https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/

SummarizationNode helps manage conversation history by summarizing messages when token count exceeds specified limits. It can be implemented as a pre-model hook in ReAct agents, allowing you to keep original message history or overwrite it with summaries. The node uses `count_tokens_approximately()` to track message history size and supports configurable parameters including `max_tokens` (threshold), `max_summary_tokens` (summary length), and `output_messages_key` (storage location). This approach provides flexible mechanisms for managing conversation memory in AI agents while preventing context window overflow and maintaining conversation context.

* **LangMem Summarization** - https://langchain-ai.github.io/langmem/guides/summarization/

LangMem provides strategies for managing long context through message history summarization. It offers two primary approaches: direct summarization using `summarize_messages()` function with configurable token thresholds and "running summary" maintenance, and the SummarizationNode approach with dedicated nodes for automatic summary propagation. Key implementation considerations include configuring token limits, using separate state keys for full message history versus summaries, and maintaining conversation context across multiple interactions. LangMem integrates seamlessly with LangGraph state management for both simple chatbots and ReAct-style agents.
"""
logger.info("#### Learn More")


logger.info("\n\n[DONE]", bright=True)