from llama_index.core import get_response_synthesizer
from jet.llm.ollama.base import Ollama
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/structured_refine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Refine with Structured Answer Filtering
When using our Refine response synthesizer for response synthesis, it's crucial to filter out non-answers. An issue often encountered is the propagation of a single unhelpful response like "I don't have the answer", which can persist throughout the synthesis process and lead to a final answer of the same nature. This can occur even when there are actual answers present in other, more relevant sections.

These unhelpful responses can be filtered out by setting `structured_answer_filtering` to `True`. It is set to `False` by default since this currently only works best if you are using an Ollama model that supports function calling.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Load Data
"""

texts = [
    "The president in the year 2040 is John Cena.",
    "The president in the year 2050 is Florence Pugh.",
    'The president in the year 2060 is Dwayne "The Rock" Johnson.',
]

"""
## Summarize
"""


# os.environ["OPENAI_API_KEY"] = "sk-..."


llm = Ollama(model="llama3.2")


summarizer = get_response_synthesizer(
    response_mode="refine", llm=llm, verbose=True
)

response = summarizer.get_response("who is president in the year 2050?", texts)

"""
### Failed Result
As you can see, we weren't able to get the correct answer from the input `texts` strings since the initial "I don't know" answer propogated through till the end of the response synthesis.
"""

logger.debug(response)

"""
Now we'll try again with `structured_answer_filtering=True`
"""


summarizer = get_response_synthesizer(
    response_mode="refine",
    llm=llm,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)

"""
### Successful Result
As you can see, we were able to determine the correct answer from the given context by filtering the `texts` strings for the ones that actually contained the answer to our question.
"""

logger.debug(response)

"""
## Non Function-calling LLMs
You may want to make use of this filtering functionality with an LLM that doesn't offer a function calling API.

In that case, the `Refine` module will automatically switch to using a structured output `Program` that doesn't rely on an external function calling API.
"""

instruct_llm = Ollama(
    model="llama3.2")


summarizer = get_response_synthesizer(
    response_mode="refine",
    llm=instruct_llm,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)
logger.debug(response)

"""
### `CompactAndRefine`
Since `CompactAndRefine` is built on top of `Refine`, this response mode also supports structured answer filtering.
"""


summarizer = get_response_synthesizer(
    response_mode="compact",
    llm=instruct_llm,
    verbose=True,
    structured_answer_filtering=True,
)

response = summarizer.get_response("who is president in the year 2050?", texts)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
