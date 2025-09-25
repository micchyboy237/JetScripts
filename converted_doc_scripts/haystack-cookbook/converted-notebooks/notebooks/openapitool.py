from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_experimental.components.tools.openapi import OpenAPITool, LLMProvider
from jet.logger import logger
import json
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
# 🧪 Invoking APIs with `OpenAPITool`

> [`OpenAPITool`](https://docs.haystack.deepset.ai/v2.8/reference/experimental-tools-api#openapitool) is discontinued and removed in `haystack-experimental==0.4.0`. As an alternative, you can use [openapi-llm](https://github.com/vblagoje/openapi-llm). 

Many APIs available on the Web provide an OpenAPI specification that describes their structure and syntax.

[`OpenAPITool`](https://docs.haystack.deepset.ai/v2.8/reference/experimental-tools-api#openapitool) is an experimental Haystack component that allows you to call an API using payloads generated from human instructions.

Here's a brief overview of how it works:
- At initialization, it loads the OpenAPI specification from a URL or a file.
- At runtime:
  - Converts human instructions into a suitable API payload using a Chat Language Model (LLM).
  - Invokes the API.
  - Returns the API response, wrapped in a Chat Message.

Let's see this component in action...

## Setup
"""
logger.info("# 🧪 Invoking APIs with `OpenAPITool`")

# ! pip install "haystack-experimental==0.3.0" haystack-ai jsonref

"""
In this notebook, we will be using some APIs that require an API key. Let's set them as environment variables.
"""
logger.info("In this notebook, we will be using some APIs that require an API key. Let's set them as environment variables.")


# os.environ["OPENAI_API_KEY"]="..."

os.environ["FIRECRAWL_API_KEY"]="..."

os.environ["SERPERDEV_API_KEY"]="..."

"""
## Call an API without credentials

In the first example, we use Open-Meteo, a Free Weather API that does not require authentication.

We use `OPENAI` as LLM provider. Other supported providers are `ANTHROPIC` and `COHERE`.
"""
logger.info("## Call an API without credentials")


tool = OpenAPITool(generator_api=LLMProvider.OPENAI,
                   spec="https://raw.githubusercontent.com/open-meteo/open-meteo/main/openapi.yml")

tool.run(messages=[ChatMessage.from_user("Weather in San Francisco, US")])

"""
## Incorporate `OpenAPITool` in a Pipeline

Next, let's create a simple Pipeline where the service response is translated into a human-understandable format using the Language Model.

We use a [`ChatPromptBuilder`](https://docs.haystack.deepset.ai/docs/chatpromptbuilder) to create a list of Chat Messages for the LM.
"""
logger.info("## Incorporate `OpenAPITool` in a Pipeline")


messages = [ChatMessage.from_user("{{user_message}}"), ChatMessage.from_user("{{service_response}}")]
builder = ChatPromptBuilder(template=messages)

pipe = Pipeline()
pipe.add_component("meteo", tool)
pipe.add_component("builder", builder)
pipe.add_component("llm", OpenAIChatGenerator(generation_kwargs={"max_tokens": 1024}))

pipe.connect("meteo", "builder.service_response")
pipe.connect("builder", "llm.messages")

result = pipe.run(data={"meteo": {"messages": [ChatMessage.from_user("weather in San Francisco, US")]},
                        "builder": {"user_message": [ChatMessage.from_user("Explain the weather in San Francisco in a human understandable way")]}})

logger.debug(result["llm"]["replies"][0].content)

"""
## Use an API with credentials in a Pipeline

In this example, we use [Firecrawl](https://www.firecrawl.dev/): a project that scrape Web pages (and Web sites) and convert them into clean text. Firecrawl has an API that requires an API key.

In the following Pipeline, we use Firecrawl to scrape a news article, which is then summarized using a Language Model.
"""
logger.info("## Use an API with credentials in a Pipeline")

messages = [ChatMessage.from_user("{{user_message}}"), ChatMessage.from_user("{{service_response}}")]
builder = ChatPromptBuilder(template=messages)


pipe = Pipeline()
pipe.add_component("firecrawl", OpenAPITool(generator_api=LLMProvider.OPENAI,
                                            spec="https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json",
                                            credentials=Secret.from_env_var("FIRECRAWL_API_KEY")))
pipe.add_component("builder", builder)
pipe.add_component("llm", OpenAIChatGenerator(generation_kwargs={"max_tokens": 1024}))

pipe.connect("firecrawl", "builder.service_response")
pipe.connect("builder", "llm.messages")

user_prompt = "Given the article below, list the most important facts in a bulleted list. Do not include repetitions. Max 5 points."

result = pipe.run(data={"firecrawl": {"messages": [ChatMessage.from_user("Scrape https://lite.cnn.com/2024/07/18/style/rome-ancient-papal-palace/index.html")]},
                        "builder": {"user_message": [ChatMessage.from_user(user_prompt)]}})

logger.debug(result["llm"]["replies"][0].content)

"""
## Create a Pipeline with multiple `OpenAPITool` components

In this example, we show a Pipeline where multiple alternative APIs can be invoked depending on the user query. In particular, a Google Search (via Serper.dev) can be performed or a single page can be scraped using Firecrawl.

⚠️ The approach shown is just one way to achieve this using [conditional routing](https://docs.haystack.deepset.ai/docs/conditionalrouter). We are currently experimenting with tool support in Haystack, and there may be simpler ways to achieve the same result in the future.
"""
logger.info("## Create a Pipeline with multiple `OpenAPITool` components")


decision_prompt_template = """
You are a virtual assistant, equipped with the following tools:

- `{"tool_name": "search_web", "tool_description": "Access to Google search, use this tool whenever information on recents events is needed"}`
- `{"tool_name": "scrape_page", "tool_description": "Use this tool to scrape and crawl web pages"}`

Select the most appropriate tool to resolve the user's query. Respond in JSON format, specifying the user request and the chosen tool for the response.
If you can't match user query to an above listed tools, respond with `none`.


Here are some examples:

```json
{
  "query": "Why did Elon Musk recently sue Ollama?",
  "response": "search_web"
}
{
  "query": "What is on the front-page of hackernews today?",
  "response": "scrape_page"
}
{
  "query": "Tell me about Berlin",
  "response": "none"
}
```

Choose the best tool (or none) for each user request, considering the current context of the conversation specified above.

{"query": {{query}}, "response": }
"""

def get_tool_name(replies):
  try:
    tool_name = json.loads(replies)["response"]
    return tool_name
  except:
    return "error"


routes = [
    {
        "condition": "{{replies[0] | get_tool_name == 'search_web'}}",
        "output": "{{query}}",
        "output_name": "search_web",
        "output_type": str,
    },
    {
        "condition": "{{replies[0] | get_tool_name == 'scrape_page'}}",
        "output": "{{query}}",
        "output_name": "scrape_page",
        "output_type": str,
    },
    {
        "condition": "{{replies[0] | get_tool_name == 'none'}}",
        "output": "{{replies[0]}}",
        "output_name": "no_tools",
        "output_type": str,
    },
    {
        "condition": "{{replies[0] | get_tool_name == 'error'}}",
        "output": "{{replies[0]}}",
        "output_name": "error",
        "output_type": str,
    },
]



messages = [ChatMessage.from_user("{{query}}")]

search_web_chat_builder = ChatPromptBuilder(template=messages)
scrape_page_chat_builder = ChatPromptBuilder(template=messages)

search_web_tool = OpenAPITool(generator_api=LLMProvider.OPENAI,
                   spec="https://bit.ly/serper_dev_spec_yaml",
                   credentials=Secret.from_env_var("SERPERDEV_API_KEY"))

scrape_page_tool = OpenAPITool(generator_api=LLMProvider.OPENAI,
                   spec="https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json",
                   credentials=Secret.from_env_var("FIRECRAWL_API_KEY"))

pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=decision_prompt_template))
pipe.add_component("llm", OpenAIGenerator())
pipe.add_component("router", ConditionalRouter(routes, custom_filters={"get_tool_name": get_tool_name}))
pipe.add_component("search_web_chat_builder", search_web_chat_builder)
pipe.add_component("scrape_page_chat_builder", scrape_page_chat_builder)
pipe.add_component("search_web_tool", search_web_tool)
pipe.add_component("scrape_page_tool", scrape_page_tool)

pipe.connect("prompt_builder", "llm")
pipe.connect("llm.replies", "router.replies")
pipe.connect("router.search_web", "search_web_chat_builder")
pipe.connect("router.scrape_page", "scrape_page_chat_builder")
pipe.connect("search_web_chat_builder", "search_web_tool")
pipe.connect("scrape_page_chat_builder", "scrape_page_tool")

query = "Who won the UEFA European Football Championship?"

pipe.run({"prompt_builder": {"query": query}, "router": {"query": query}})

query = "What is on the front-page of BBC today?"

pipe.run({"prompt_builder": {"query": query}, "router": {"query": query}})

logger.info("\n\n[DONE]", bright=True)