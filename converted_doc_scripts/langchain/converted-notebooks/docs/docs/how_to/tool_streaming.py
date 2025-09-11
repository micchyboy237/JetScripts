from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.tools import tool
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
# How to stream tool calls

When [tools](/docs/concepts/tools/) are called in a streaming context, 
[message chunks](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
will be populated with [tool call chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolCallChunk.html#langchain_core.messages.tool.ToolCallChunk) 
objects in a list via the `.tool_call_chunks` attribute. A `ToolCallChunk` includes 
optional string fields for the tool `name`, `args`, and `id`, and includes an optional 
integer field `index` that can be used to join chunks together. Fields are optional 
because portions of a tool call may be streamed across different chunks (e.g., a chunk 
that includes a substring of the arguments may have null values for the tool name and id).

Because message chunks inherit from their parent message class, an 
[AIMessageChunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
with tool call chunks will also include `.tool_calls` and `.invalid_tool_calls` fields. 
These fields are parsed best-effort from the message's tool call chunks.

Note that not all providers currently support streaming for tool calls. Before we start let's define our tools and our model.
"""
logger.info("# How to stream tool calls")



@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

# from getpass import getpass


# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools(tools)

"""
Now let's define our query and stream our output:
"""
logger.info("Now let's define our query and stream our output:")

query = "What is 3 * 12? Also, what is 11 + 49?"

for chunk in llm_with_tools.stream(query):
    logger.debug(chunk.tool_call_chunks)

"""
Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.

For example, below we accumulate tool call chunks:
"""
logger.info("Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    logger.debug(gathered.tool_call_chunks)

logger.debug(type(gathered.tool_call_chunks[0]["args"]))

"""
And below we accumulate tool calls to demonstrate partial parsing:
"""
logger.info("And below we accumulate tool calls to demonstrate partial parsing:")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    logger.debug(gathered.tool_calls)

logger.debug(type(gathered.tool_calls[0]["args"]))

"""
Note the key difference: accumulating `tool_call_chunks` captures the raw tool arguments as an unparsed string as they are streamed. In contrast, **accumulating** `tool_calls` demonstrates partial parsing by progressively converting the streamed argument string into a valid, usable dictionary at each step of the process.
"""
logger.info("Note the key difference: accumulating `tool_call_chunks` captures the raw tool arguments as an unparsed string as they are streamed. In contrast, **accumulating** `tool_calls` demonstrates partial parsing by progressively converting the streamed argument string into a valid, usable dictionary at each step of the process.")

logger.info("\n\n[DONE]", bright=True)