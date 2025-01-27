from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from typing import Any, List
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from jet.llm.ollama.base_langchain import ChatOllama
from langchain.globals import set_verbose
from langchain.chains import FlareChain

initialize_ollama_settings()

"""
# Retrieve as you generate with FLARE

This notebook is an implementation of Forward-Looking Active REtrieval augmented generation (FLARE).

Please see the original repo [here](https://github.com/jzbjyb/FLARE/tree/main).

The basic idea is:

- Start answering a question
- If you start generating tokens the model is uncertain about, look up relevant documents
- Use those documents to continue generating
- Repeat until finished

There is a lot of cool detail in how the lookup of relevant documents is done.
Basically, the tokens that model is uncertain about are highlighted, and then an LLM is called to generate a question that would lead to that answer. For example, if the generated text is `Joe Biden went to Harvard`, and the tokens the model was uncertain about was `Harvard`, then a good generated question would be `where did Joe Biden go to college`. This generated question is then used in a retrieval step to fetch relevant documents.

In order to set up this chain, we will need three things:

- An LLM to generate the answer
- An LLM to generate hypothetical questions to use in retrieval
- A retriever to use to look up answers for

The LLM that we use to generate the answer needs to return logprobs so we can identify uncertain tokens. For that reason, we HIGHLY recommend that you use the Ollama wrapper (NB: not the ChatOllama wrapper, as that does not return logprobs).

The LLM we use to generate hypothetical questions to use in retrieval can be anything. In this notebook we will use ChatOllama because it is fast and cheap.

The retriever can be anything. In this notebook we will use [SERPER](https://serper.dev/) search engine, because it is cheap.

Other important parameters to understand:

- `max_generation_len`: The maximum number of tokens to generate before stopping to check if any are uncertain
- `min_prob`: Any tokens generated with probability below this will be considered uncertain
"""

"""
## Imports
"""


# os.environ["SERPER_API_KEY"] = ""
# os.environ["OPENAI_API_KEY"] = ""


"""
## Retriever
"""


class SerperSearchRetriever(BaseRetriever):
    search: GoogleSerperAPIWrapper = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        return [Document(page_content=self.search.run(query))]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()


retriever = SerperSearchRetriever(search=GoogleSerperAPIWrapper())

"""
## FLARE Chain
"""


set_verbose(True)


flare = FlareChain.from_llm(
    ChatOllama(model="llama3.1"),
    retriever=retriever,
    max_generation_len=164,
    min_prob=0.3,
)

query = "explain in great detail the difference between the langchain framework and baby agi"
result = flare.run(query)

logger.newline()
logger.info("Flare Query 1:")
logger.debug(query)
logger.success(result)


llm = ChatOllama(model="llama3.1")
result = llm.invoke(query)

logger.newline()
logger.info("LLM Query 1:")
logger.debug(query)
logger.success(result)


query = "how are the origin stories of langchain and bitcoin similar or different?"
result = flare.run(query)

logger.newline()
logger.info("Flare Query 2:")
logger.debug(query)
logger.success(result)


logger.info("\n\n[DONE]", bright=True)
