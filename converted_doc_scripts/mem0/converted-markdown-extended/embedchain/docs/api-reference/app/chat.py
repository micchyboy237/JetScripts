from embedchain import App
from embedchain.config import BaseLlmConfig
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '💬 chat'
---

`chat()` method allows you to chat over your data sources using a user-friendly chat API. You can find the signature below:

### Parameters

<ParamField path="input_query" type="str">
    Question to ask
</ParamField>
<ParamField path="config" type="BaseLlmConfig" optional>
    Configure different llm settings such as prompt, temprature, number_documents etc.
</ParamField>
<ParamField path="dry_run" type="bool" optional>
    The purpose is to test the prompt structure without actually running LLM inference. Defaults to `False`
</ParamField>
<ParamField path="where" type="dict" optional>
    A dictionary of key-value pairs to filter the chunks from the vector database. Defaults to `None`
</ParamField>
<ParamField path="session_id" type="str" optional>
    Session ID of the chat. This can be used to maintain chat history of different user sessions. Default value: `default`
</ParamField>
<ParamField path="citations" type="bool" optional>
    Return citations along with the LLM answer. Defaults to `False`
</ParamField>

### Returns

<ResponseField name="answer" type="str | tuple">
  If `citations=False`, return a stringified answer to the question asked. <br />
  If `citations=True`, returns a tuple with answer and citations respectively.
</ResponseField>

## Usage

### With citations

If you want to get the answer to question and return both answer and citations, use the following code snippet:
"""
logger.info("### Parameters")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

answer, sources = app.chat("What is the net worth of Elon?", citations=True)
logger.debug(answer)

logger.debug(sources)

"""
<Note>
When `citations=True`, note that the returned `sources` are a list of tuples where each tuple has two elements (in the following order):
1. source chunk
2. dictionary with metadata about the source chunk
    - `url`: url of the source
    - `doc_id`: document id (used for book keeping purposes)
    - `score`: score of the source chunk with respect to the question
    - other metadata you might have added at the time of adding the source
</Note>


### Without citations

If you just want to return answers and don't want to return citations, you can use the following example:
"""
logger.info("### Without citations")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

answer = app.chat("What is the net worth of Elon?")
logger.debug(answer)

"""
### With session id

If you want to maintain chat sessions for different users, you can simply pass the `session_id` keyword argument. See the example below:
"""
logger.info("### With session id")


app = App()
app.add("https://www.forbes.com/profile/elon-musk")

app.chat("What is the net worth of Elon Musk?", session_id="user1")
app.chat("What is the net worth of Bill Gates?", session_id="user2")
app.chat("What was my last question", session_id="user1")

"""
### With custom context window

If you want to customize the context window that you want to use during chat (default context window is 3 document chunks), you can do using the following code snippet:
"""
logger.info("### With custom context window")


app = App()
app.add("https://www.forbes.com/profile/elon-musk")

query_config = BaseLlmConfig(number_documents=5)
app.chat("What is the net worth of Elon Musk?", config=query_config)

"""
### With Mem0 to store chat history

Mem0 is a cutting-edge long-term memory for LLMs to enable personalization for the GenAI stack. It enables LLMs to remember past interactions and provide more personalized responses. 

In order to use Mem0 to enable memory for personalization in your apps:
- Install the [`mem0`](https://docs.mem0.ai/) package using `pip install mem0ai`. 
- Prepare config for `memory`, refer [Configurations](docs/api-reference/advanced/configuration.mdx).
"""
logger.info("### With Mem0 to store chat history")


config = {
  "memory": {
    "top_k": 5
  }
}

app = App.from_config(config=config)
app.add("https://www.forbes.com/profile/elon-musk")

app.chat("What is the net worth of Elon Musk?")

"""
## How Mem0 works:
- Mem0 saves context derived from each user question into its memory.
- When a user poses a new question, Mem0 retrieves relevant previous memories.
- The `top_k` parameter in the memory configuration specifies the number of top memories to consider during retrieval.
- Mem0 generates the final response by integrating the user's question, context from the data source, and the relevant memories.
"""
logger.info("## How Mem0 works:")

logger.info("\n\n[DONE]", bright=True)