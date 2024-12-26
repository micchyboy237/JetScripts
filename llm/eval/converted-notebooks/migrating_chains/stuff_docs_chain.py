from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")
from langchain_core.documents import Document

documents = [
Document(page_content="Apples are red", metadata={"title": "apple_book"}),
Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

document_prompt = PromptTemplate(
input_variables=["page_content"], template="{page_content}"
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")

"""
# Migrating from StuffDocumentsChain

[StuffDocumentsChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.StuffDocumentsChain.html) combines documents by concatenating them into a single context window. It is a straightforward and effective strategy for combining documents for question-answering, summarization, and other purposes.

[create_stuff_documents_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) is the recommended alternative. It functions the same as `StuffDocumentsChain`, with better support for streaming and batch functionality. Because it is a simple combination of [LCEL primitives](/docs/concepts/lcel), it is also easier to extend and incorporate into other LangChain applications.

Below we will go through both `StuffDocumentsChain` and `create_stuff_documents_chain` on a simple example for illustrative purposes.

Let's first load a chat model:

import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm" />
"""



"""
## Example

Let's go through an example where we analyze a set of documents. We first generate some simple documents for illustrative purposes:
"""



"""
### Legacy

<details open>

Below we show an implementation with `StuffDocumentsChain`. We define the prompt template for a summarization task and instantiate a [LLMChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html) object for this purpose. We define how documents are formatted into the prompt and ensure consistency among the keys in the various prompts.
"""

document_variable_name = "context"
prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")

llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

"""
We can now invoke our chain:
"""

result = chain.invoke(documents)
result["output_text"]

for chunk in chain.stream(documents):
    print(chunk)

"""
</details>

### LCEL

<details open>

Below we show an implementation using `create_stuff_documents_chain`:
"""

chain = create_stuff_documents_chain(llm, prompt)

"""
Invoking the chain, we obtain a similar result as before:
"""

result = chain.invoke({"context": documents})
result

"""
Note that this implementation supports streaming of output tokens:
"""

for chunk in chain.stream({"context": documents}):
    print(chunk, end=" | ")

"""
</details>

## Next steps

Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.

See these [how-to guides](/docs/how_to/#qa-with-rag) for more on question-answering tasks with RAG.

See [this tutorial](/docs/tutorials/summarization/) for more LLM-based summarization strategies.
"""

logger.info("\n\n[DONE]", bright=True)