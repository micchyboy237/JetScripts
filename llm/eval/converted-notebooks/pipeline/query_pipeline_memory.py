from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Query Pipeline Chat Engine
# 
# By combining a query pipeline with a memory buffer, we can design our own custom chat engine loop.

# %pip install llama-index-core
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-ollama
# %pip install llama-index-postprocessor-colbert-rerank
# %pip install llama-index-readers-web

import os

# os.environ["OPENAI_API_KEY"] = "sk-..."

## Index Construction
# 
# As a test, we will index Anthropic's latest documentation about tool/function calling.

from llama_index.readers.web import BeautifulSoupWebReader

reader = BeautifulSoupWebReader()

documents = reader.load_data(
    ["https://docs.anthropic.com/claude/docs/tool-use"]
)

# If you inspected the document text, you'd notice that there are way too many blank lines, lets clean that up a bit.

lines = documents[0].text.split("\n")

fixed_lines = [lines[0]]
for idx in range(1, len(lines)):
    if lines[idx].strip() == "" and lines[idx - 1].strip() == "":
        continue
    fixed_lines.append(lines[idx])

documents[0].text = "\n".join(fixed_lines)

# Now, we can create our index using Ollama embeddings.

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(
        model="text-embedding-3-large", embed_batch_size=256
    ),
)

## Query Pipeline Contruction
# 
# As a demonstration, lets make a robust query pipeline with HyDE for retrieval and Colbert for reranking.

from llama_index.core.query_pipeline import (
    QueryPipeline,
    InputComponent,
    ArgPackComponent,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.colbert_rerank import ColbertRerank

input_component = InputComponent()

rewrite = (
    "Please write a query to a semantic search engine using the current conversation.\n"
    "\n"
    "\n"
    "{chat_history_str}"
    "\n"
    "\n"
    "Latest message: {query_str}\n"
    'Query:"""\n'
)
rewrite_template = PromptTemplate(rewrite)
llm = Ollama(
    model="llama3.1", request_timeout=300.0, context_window=4096,
    temperature=0.2,
)

argpack_component = ArgPackComponent()

retriever = index.as_retriever(similarity_top_k=6)

reranker = ColbertRerank(top_n=3)

# For generating a response using chat history + retrieved nodes, lets create a custom component.

from typing import Any, Dict, List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import ChatMessage
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.core.schema import NodeWithScore

DEFAULT_CONTEXT_PROMPT = (
    "Here is some context that may be relevant:\n"
    "-----\n"
    "{node_context}\n"
    "-----\n"
    "Please write a response to the following question, using the above context:\n"
    "{query_str}\n"
)


class ResponseWithChatHistory(CustomQueryComponent):
    llm: Ollama = Field(..., description="Ollama LLM")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use for the LLM"
    )
    context_prompt: str = Field(
        default=DEFAULT_CONTEXT_PROMPT,
        description="Context prompt to use for the LLM",
    )

    def _validate_component_inputs(
        self, input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"chat_history", "nodes", "query_str"}

    @property
    def _output_keys(self) -> set:
        return {"response"}

    def _prepare_context(
        self,
        chat_history: List[ChatMessage],
        nodes: List[NodeWithScore],
        query_str: str,
    ) -> List[ChatMessage]:
        node_context = ""
        for idx, node in enumerate(nodes):
            node_text = node.get_content(metadata_mode="llm")
            node_context += f"Context Chunk {idx}:\n{node_text}\n\n"

        formatted_context = self.context_prompt.format(
            node_context=node_context, query_str=query_str
        )
        user_message = ChatMessage(role="user", content=formatted_context)

        chat_history.append(user_message)

        if self.system_prompt is not None:
            chat_history = [
                ChatMessage(role="system", content=self.system_prompt)
            ] + chat_history

        return chat_history

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            chat_history, nodes, query_str
        )

        response = llm.chat(prepared_context)

        return {"response": response}

    async def _arun_component(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the component asynchronously."""
        chat_history = kwargs["chat_history"]
        nodes = kwargs["nodes"]
        query_str = kwargs["query_str"]

        prepared_context = self._prepare_context(
            chat_history, nodes, query_str
        )

        response = llm.chat(prepared_context)

        return {"response": response}


response_component = ResponseWithChatHistory(
    llm=llm,
    system_prompt=(
        "You are a Q&A system. You will be provided with the previous chat history, "
        "as well as possibly relevant context, to assist in answering a user message."
    ),
)

# With our modules created, we can link them together in a query pipeline.

pipeline = QueryPipeline(
    modules={
        "input": input_component,
        "rewrite_template": rewrite_template,
        "llm": llm,
        "rewrite_retriever": retriever,
        "query_retriever": retriever,
        "join": argpack_component,
        "reranker": reranker,
        "response_component": response_component,
    },
    verbose=False,
)

pipeline.add_link(
    "input", "rewrite_template", src_key="query_str", dest_key="query_str"
)
pipeline.add_link(
    "input",
    "rewrite_template",
    src_key="chat_history_str",
    dest_key="chat_history_str",
)
pipeline.add_link("rewrite_template", "llm")
pipeline.add_link("llm", "rewrite_retriever")
pipeline.add_link("input", "query_retriever", src_key="query_str")

pipeline.add_link("rewrite_retriever", "join", dest_key="rewrite_nodes")
pipeline.add_link("query_retriever", "join", dest_key="query_nodes")

pipeline.add_link("join", "reranker", dest_key="nodes")
pipeline.add_link(
    "input", "reranker", src_key="query_str", dest_key="query_str"
)

pipeline.add_link("reranker", "response_component", dest_key="nodes")
pipeline.add_link(
    "input", "response_component", src_key="query_str", dest_key="query_str"
)
pipeline.add_link(
    "input",
    "response_component",
    src_key="chat_history",
    dest_key="chat_history",
)

# Lets test the pipeline to confirm it works!

## Running the Pipeline with Memory
# 
# The above pipeline uses two inputs -- a query string and a chat_history list.
# 
# The query string is simply the string input/query.
# 
# The chat history list is a list of ChatMessage objects. We can use a memory module from llama-index to directly manage and create the memory!

from llama_index.core.memory import ChatMemoryBuffer

pipeline_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

# Lets pre-create a "chat session" and watch it play out.

user_inputs = [
    "Hello!",
    "How does tool-use work with Claude-3 work?",
    "What models support it?",
    "Thanks, that what I needed to know!",
]

for msg in user_inputs:
    chat_history = pipeline_memory.get()

    chat_history_str = "\n".join([str(x) for x in chat_history])

    response = pipeline.run(
        query_str=msg,
        chat_history=chat_history,
        chat_history_str=chat_history_str,
    )

    user_msg = ChatMessage(role="user", content=msg)
    pipeline_memory.put(user_msg)
    print(str(user_msg))

    pipeline_memory.put(response.message)
    print(str(response.message))
    print()

logger.info("\n\n[DONE]", bright=True)