from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
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
# GraphQL

>[GraphQL](https://graphql.org/) is a query language for APIs and a runtime for executing those queries against your data. `GraphQL` provides a complete and understandable description of the data in your API, gives clients the power to ask for exactly what they need and nothing more, makes it easier to evolve APIs over time, and enables powerful developer tools.

By including a `BaseGraphQLTool` in the list of tools provided to an Agent, you can grant your Agent the ability to query data from GraphQL APIs for any purposes you need.

This Jupyter Notebook demonstrates how to use the `GraphQLAPIWrapper` component with an Agent.

In this example, we'll be using the public `Star Wars GraphQL API` available at the following endpoint: https://swapi-graphql.netlify.app/graphql .

First, you need to install `httpx` and `gql` Python packages.
"""
logger.info("# GraphQL")

pip install httpx gql > /dev/null

# %pip install --upgrade --quiet  langchain-community

"""
Now, let's create a BaseGraphQLTool instance with the specified Star Wars API endpoint and initialize an Agent with the tool.
"""
logger.info("Now, let's create a BaseGraphQLTool instance with the specified Star Wars API endpoint and initialize an Agent with the tool.")


# os.environ["OPENAI_API_KEY"] = ""


tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://swapi-graphql.netlify.app/graphql",
)


agent = create_react_agent("ollama:gpt-4.1-mini", tools)

"""
Now, we can use the Agent to run queries against the Star Wars GraphQL API. Let's ask the Agent to list all the Star Wars films and their release dates.
"""
logger.info("Now, we can use the Agent to run queries against the Star Wars GraphQL API. Let's ask the Agent to list all the Star Wars films and their release dates.")

graphql_fields = """allFilms {
    films {
      title
      director
      releaseDate
      speciesConnection {
        species {
          name
          classification
          homeworld {
            name
          }
        }
      }
    }
  }

"""

suffix = "Search for the titles of all the stawars films stored in the graphql database that has this schema "

input_message = {
    "role": "user",
    "content": suffix + graphql_fields,
}

for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

logger.info("\n\n[DONE]", bright=True)