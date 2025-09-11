from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import ChatModelTabs from "@theme/ChatModelTabs";
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
# Migrating from StuffDocumentsChain

[StuffDocumentsChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.StuffDocumentsChain.html) combines documents by concatenating them into a single context window. It is a straightforward and effective strategy for combining documents for question-answering, summarization, and other purposes.

[create_stuff_documents_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) is the recommended alternative. It functions the same as `StuffDocumentsChain`, with better support for streaming and batch functionality. Because it is a simple combination of [LCEL primitives](/docs/concepts/lcel), it is also easier to extend and incorporate into other LangChain applications.

Below we will go through both `StuffDocumentsChain` and `create_stuff_documents_chain` on a simple example for illustrative purposes.

Let's first load a chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("# Migrating from StuffDocumentsChain")


llm = ChatOllama(model="llama3.2")

"""
## Example

Let's go through an example where we analyze a set of documents. We first generate some simple documents for illustrative purposes:
"""
logger.info("## Example")


documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]

"""
### Legacy

<details open>

Below we show an implementation with `StuffDocumentsChain`. We define the prompt template for a summarization task and instantiate a [LLMChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html) object for this purpose. We define how documents are formatted into the prompt and ensure consistency among the keys in the various prompts.
"""
logger.info("### Legacy")


document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
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
logger.info("We can now invoke our chain:")

result = chain.invoke(documents)
result["output_text"]

for chunk in chain.stream(documents):
    logger.debug(chunk)

"""
</details>

### LCEL

<details open>

Below we show an implementation using `create_stuff_documents_chain`:
"""
logger.info("### LCEL")


prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
chain = create_stuff_documents_chain(llm, prompt)

"""
Invoking the chain, we obtain a similar result as before:
"""
logger.info("Invoking the chain, we obtain a similar result as before:")

result = chain.invoke({"context": documents})
result

"""
Note that this implementation supports streaming of output tokens:
"""
logger.info("Note that this implementation supports streaming of output tokens:")

for chunk in chain.stream({"context": documents}):
    logger.debug(chunk, end=" | ")

"""
</details>

## Next steps

Check out the [LCEL conceptual docs](/docs/concepts/lcel) for more background information.

See these [how-to guides](/docs/how_to/#qa-with-rag) for more on question-answering tasks with RAG.

See [this tutorial](/docs/tutorials/summarization/) for more LLM-based summarization strategies.
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)