from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from IPython.display import SVG
from langchain_experimental.cpal.base import CPALChain
from langchain_experimental.pal_chain import PALChain
from jet.llm.ollama.base_langchain import Ollama

initialize_ollama_settings()

"""
# Causal program-aided language (CPAL) chain

The CPAL chain builds on the recent PAL to stop LLM hallucination. The problem with the PAL approach is that it hallucinates on a math problem with a nested chain of dependence. The innovation here is that this new CPAL approach includes causal structure to fix hallucination.

The original [PR's description](https://github.com/langchain-ai/langchain/pull/6255) contains a full overview.

Using the CPAL chain, the LLM translated this

    "Tim buys the same number of pets as Cindy and Boris."
    "Cindy buys the same number of pets as Bill plus Bob."
    "Boris buys the same number of pets as Ben plus Beth."
    "Bill buys the same number of pets as Obama."
    "Bob buys the same number of pets as Obama."
    "Ben buys the same number of pets as Obama."
    "Beth buys the same number of pets as Obama."
    "If Obama buys one pet, how many pets total does everyone buy?"


into this

![complex-graph.png](/img/cpal_diagram.png).

Outline of code examples demoed in this notebook.

1. CPAL's value against hallucination: CPAL vs PAL  
    1.1 Complex narrative  
    1.2 Unanswerable math word problem  
2. CPAL's three types of causal diagrams ([The Book of Why](https://en.wikipedia.org/wiki/The_Book_of_Why)).   
    2.1 Mediator   
    2.2 Collider   
    2.3 Confounder
"""


llm = Ollama(temperature=0, max_tokens=512)
cpal_chain = CPALChain.from_univariate_prompt(llm=llm, verbose=True)
pal_chain = PALChain.from_math_prompt(llm=llm, verbose=True)

"""
## CPAL's value against hallucination: CPAL vs PAL

Like PAL, CPAL intends to reduce large language model (LLM) hallucination.

The CPAL chain is different from the PAL chain for a couple of reasons.

CPAL adds a causal structure (or DAG) to link entity actions (or math expressions).
The CPAL math expressions are modeling a chain of cause and effect relations, which can be intervened upon, whereas for the PAL chain math expressions are projected math identities.
"""

"""
### 1.1 Complex narrative

Takeaway: PAL hallucinates, CPAL does not hallucinate.
"""

question = (
    "Tim buys the same number of pets as Cindy and Boris."
    "Cindy buys the same number of pets as Bill plus Bob."
    "Boris buys the same number of pets as Ben plus Beth."
    "Bill buys the same number of pets as Obama."
    "Bob buys the same number of pets as Obama."
    "Ben buys the same number of pets as Obama."
    "Beth buys the same number of pets as Obama."
    "If Obama buys one pet, how many pets total does everyone buy?"
)

pal_chain.run(question)

cpal_chain.run(question)

cpal_chain.draw(path="web.svg")
SVG("web.svg")

"""
### Unanswerable math

Takeaway: PAL hallucinates, where CPAL, rather than hallucinate, answers with _"unanswerable, narrative question and plot are incoherent"_
"""

question = (
    "Jan has three times the number of pets as Marcia."
    "Marcia has two more pets than Cindy."
    "If Cindy has ten pets, how many pets does Barak have?"
)

pal_chain.run(question)

try:
    cpal_chain.run(question)
except Exception as e_msg:
    print(e_msg)

"""
### Basic math

#### Causal mediator
"""

question = (
    "Jan has three times the number of pets as Marcia. "
    "Marcia has two more pets than Cindy. "
    "If Cindy has four pets, how many total pets do the three have?"
)

"""
---
PAL
"""

pal_chain.run(question)

"""
---
CPAL
"""

cpal_chain.run(question)

cpal_chain.draw(path="web.svg")
SVG("web.svg")

"""
### Causal collider
"""

question = (
    "Jan has the number of pets as Marcia plus the number of pets as Cindy. "
    "Marcia has no pets. "
    "If Cindy has four pets, how many total pets do the three have?"
)

cpal_chain.run(question)

cpal_chain.draw(path="web.svg")
SVG("web.svg")

"""
### Causal confounder
"""

question = (
    "Jan has the number of pets as Marcia plus the number of pets as Cindy. "
    "Marcia has two more pets than Cindy. "
    "If Cindy has four pets, how many total pets do the three have?"
)

cpal_chain.run(question)

cpal_chain.draw(path="web.svg")
SVG("web.svg")

# %load_ext autoreload
# %autoreload 2

logger.info("\n\n[DONE]", bright=True)
