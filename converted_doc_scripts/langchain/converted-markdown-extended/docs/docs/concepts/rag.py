from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import SystemMessage, HumanMessage
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
# Retrieval augmented generation (RAG)

:::info[Prerequisites]

* [Retrieval](/docs/concepts/retrieval/)

:::

## Overview

Retrieval Augmented Generation (RAG) is a powerful technique that enhances [language models](/docs/concepts/chat_models/) by combining them with external knowledge bases.
RAG addresses [a key limitation of models](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise): models rely on fixed training datasets, which can lead to outdated or incomplete information.
When given a query, RAG systems first search a knowledge base for relevant information.
The system then incorporates this retrieved information into the model's prompt.
The model uses the provided context to generate a response to the query.
By bridging the gap between vast language models and dynamic, targeted information retrieval, RAG is a powerful technique for building more capable and reliable AI systems.

## Key concepts

![Conceptual Overview](/img/rag_concepts.png)

(1) **Retrieval system**: Retrieve relevant information from a knowledge base.

(2) **Adding external knowledge**: Pass retrieved information to a model.

## Retrieval system

Model's have internal knowledge that is often fixed, or at least not updated frequently due to the high cost of training.
This limits their ability to answer questions about current events, or to provide specific domain knowledge.
To address this, there are various knowledge injection techniques like [fine-tuning](https://hamel.dev/blog/posts/fine_tuning_valuable.html) or continued pre-training.
Both are [costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise) and often [poorly suited](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) for factual retrieval.
Using a retrieval system offers several advantages:

- **Up-to-date information**: RAG can access and utilize the latest data, keeping responses current.
- **Domain-specific expertise**: With domain-specific knowledge bases, RAG can provide answers in specific domains.
- **Reduced hallucination**: Grounding responses in retrieved facts helps minimize false or invented information.
- **Cost-effective knowledge integration**: RAG offers a more efficient alternative to expensive model fine-tuning.

:::info[Further reading]

See our conceptual guide on [retrieval](/docs/concepts/retrieval/).

:::

## Adding external knowledge

With a retrieval system in place, we need to pass knowledge from this system to the model.
A RAG pipeline typically achieves this following these steps:

- Receive an input query.
- Use the retrieval system to search for relevant information based on the query.
- Incorporate the retrieved information into the prompt sent to the LLM.
- Generate a response that leverages the retrieved context.

As an example, here's a simple RAG workflow that passes information from a [retriever](/docs/concepts/retrievers/) to a [chat model](/docs/concepts/chat_models/):
"""
logger.info("# Retrieval augmented generation (RAG)")


system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}:"""

question = """What are the main components of an LLM-powered autonomous agent system?"""

docs = retriever.invoke(question)

docs_text = "".join(d.page_content for d in docs)

system_prompt_fmt = system_prompt.format(context=docs_text)

model = ChatOllama(model="llama3.2")

questions = model.invoke([SystemMessage(content=system_prompt_fmt),
                          HumanMessage(content=question)])

"""
:::info[Further reading]

RAG a deep area with many possible optimization and design choices:

* See [this excellent blog](https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval?utm_source=profile&utm_medium=reader2) from Cameron Wolfe for a comprehensive overview and history of RAG.
* See our [RAG how-to guides](/docs/how_to/#qa-with-rag).
* See our RAG [tutorials](/docs/tutorials/).
* See our RAG from Scratch course, with [code](https://github.com/langchain-ai/rag-from-scratch) and [video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x).
* Also, see our RAG from Scratch course [on Freecodecamp](https://youtu.be/sVcwVQRHIc8?feature=shared).

:::
"""
logger.info("RAG a deep area with many possible optimization and design choices:")

logger.info("\n\n[DONE]", bright=True)