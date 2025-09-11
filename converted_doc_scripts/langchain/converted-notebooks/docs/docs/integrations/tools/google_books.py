from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.google_books import GoogleBooksQueryRun
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
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
# Google Books

The Google Books tool that supports the ReAct pattern and allows you to search the Google Books API. Google Books is the largest API in the world that keeps track of books in a curated manner. It has over 40 million entries, which can give users a significant amount of data.

### Tool features

Currently the tool has the following capabilities:
- Gathers the relevant information from the Google Books API using a key word search
- Formats the information into a readable output, and return the result to the agent

## Setup

Make sure `langchain-community` is installed.
"""
logger.info("# Google Books")

# %pip install --upgrade --quiet  langchain-community

"""
### Credentials

You will need an API key from Google Books. You can do this by visiting and following the steps at [https://developers.google.com/books/docs/v1/using#APIKey](https://developers.google.com/books/docs/v1/using#APIKey).

Then you will need to set the environment variable `GOOGLE_BOOKS_API_KEY` to your Google Books API key.

## Instantiation

To instantiate the tool import the Google Books tool and set your credentials.
"""
logger.info("### Credentials")



os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"
tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())

"""
## Invocation

You can invoke the tool by calling the `run` method.
"""
logger.info("## Invocation")

tool.run("ai")

"""
### [Invoke directly with args](/docs/concepts/tools)

See below for an direct invocation example.
"""
logger.info("### [Invoke directly with args](/docs/concepts/tools)")



os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"
tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())

tool.run("ai")

"""
### [Invoke with ToolCall](/docs/concepts/tools)

See below for a tool call example.
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"

tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
llm = ChatOllama(model="llama3.2")
prompt = PromptTemplate.from_template(
    "Return the keyword, and only the keyword, that the user is looking for from this text: {text}"
)


def suggest_books(query):
    chain = prompt | llm | StrOutputParser()
    keyword = chain.invoke({"text": query})
    return tool.run(keyword)


suggestions = suggest_books("I need some information on AI")
logger.debug(suggestions)

"""
## Chaining

See the below example for chaining.
"""
logger.info("## Chaining")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GOOGLE_BOOKS_API_KEY"] = "<your Google Books API key>"

tool = GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper())
llm = ChatOllama(model="llama3.2")

instructions = """You are a book suggesting assistant."""
base_prompt = hub.pull("langchain-ai/ollama-functions-template")
prompt = base_prompt.partial(instructions=instructions)

tools = [tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke({"input": "Can you recommend me some books related to ai?"})

"""
## API reference

The Google Books API can be found here: [https://developers.google.com/books](https://developers.google.com/books)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)