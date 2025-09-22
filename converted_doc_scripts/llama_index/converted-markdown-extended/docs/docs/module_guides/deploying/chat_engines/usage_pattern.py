from jet.logger import logger
from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
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
# Usage Pattern

## Get Started

Build a chat engine from index:
"""
logger.info("# Usage Pattern")

chat_engine = index.as_chat_engine()

"""
<Aside type="tip">
To learn how to build an index, see [Indexing](/python/framework/module_guides/indexing/index_guide)
</Aside>

Have a conversation with your data:
"""
logger.info("To learn how to build an index, see [Indexing](/python/framework/module_guides/indexing/index_guide)")

response = chat_engine.chat("Tell me a joke.")

"""
Reset chat history to start a new conversation:
"""
logger.info("Reset chat history to start a new conversation:")

chat_engine.reset()

"""
Enter an interactive chat REPL:
"""
logger.info("Enter an interactive chat REPL:")

chat_engine.chat_repl()

"""
## Configuring a Chat Engine

Configuring a chat engine is very similar to configuring a query engine.

### High-Level API

You can directly build and configure a chat engine from an index in 1 line of code:
"""
logger.info("## Configuring a Chat Engine")

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

"""
> Note: you can access different chat engines by specifying the `chat_mode` as a kwarg. `condense_question` corresponds to `CondenseQuestionChatEngine`, `react` corresponds to `ReActChatEngine`, `context` corresponds to a `ContextChatEngine`.

> Note: While the high-level API optimizes for ease-of-use, it does _NOT_ expose full range of configurability.

#### Available Chat Modes

- `best` - Turn the query engine into a tool, for use with a `ReAct` data agent or an `Ollama` data agent, depending on what your LLM supports. `Ollama` data agents require `gpt-3.5-turbo` or `gpt-4` as they use the function calling API from Ollama.
- `condense_question` - Look at the chat history and re-write the user message to be a query for the index. Return the response after reading the response from the query engine.
- `context` - Retrieve nodes from the index using every user message. The retrieved text is inserted into the system prompt, so that the chat engine can either respond naturally or use the context from the query engine.
- `condense_plus_context` - A combination of `condense_question` and `context`. Look at the chat history and re-write the user message to be a retrieval query for the index. The retrieved text is inserted into the system prompt, so that the chat engine can either respond naturally or use the context from the query engine.
- `simple` - A simple chat with the LLM directly, no query engine involved.
- `react` - Same as `best`, but forces a `ReAct` data agent.
- `ollama` - Same as `best`, but forces an `Ollama` data agent.

### Low-Level Composition API

You can use the low-level composition API if you need more granular control.
Concretely speaking, you would explicitly construct `ChatEngine` object instead of calling `index.as_chat_engine(...)`.

> Note: You may need to look at API references or example notebooks.

Here's an example where we configure the following:

- configure the condense question prompt,
- initialize the conversation with some existing history,
- print verbose debug message.
"""
logger.info("#### Available Chat Modes")


custom_prompt = PromptTemplate(
    """\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""
)

custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hello assistant, we are having a insightful discussion about Paul Graham today.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
]

query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    chat_history=custom_chat_history,
    verbose=True,
)

"""
### Streaming

To enable streaming, you simply need to call the `stream_chat` endpoint instead of the `chat` endpoint.

!!! warning
This somewhat inconsistent with query engine (where you pass in a `streaming=True` flag). We are working on making the behavior more consistent!
"""
logger.info("### Streaming")

chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Tell me a joke.")
for token in streaming_response.response_gen:
    logger.debug(token, end="")

"""
See an [end-to-end tutorial](/python/examples/customization/streaming/chat_engine_condense_question_stream_response)
"""
logger.info("See an [end-to-end tutorial](/python/examples/customization/streaming/chat_engine_condense_question_stream_response)")

logger.info("\n\n[DONE]", bright=True)