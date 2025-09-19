from email.mime.text import MIMEText
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.tools import create_tool_from_function
from haystack.tools.from_function import _remove_title_from_schema
from jet.logger import CustomLogger
from langchain_community.agent_toolkits import FileManagementToolkit
from pprint import pp
from pydantic import create_model
from trafilatura import fetch_url, extract
from typing import Annotated
from typing import List
import os
import requests
import shutil
import smtplib, ssl


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# 🗞️ Newsletter Sending Agent with Haystack Tools

🧑‍🍳 **Demo by Stefano Fiorucci ([X](https://x.com/theanakin87), [LinkedIn](https://www.linkedin.com/in/stefano-fiorucci/))  and Tuana Celik([X](https://x.com/tuanacelik), [LinkedIn](https://www.linkedin.com/in/tuanacelik/))**


In this recipe, we will build a newsletter sending agent with 3 tools:
- A tool that fetches the top stories from Hacker News
- A tool that creates newsletters for a particular audience
- A tool that can send emails (with Gmail)

> This notebook is updated after [Haystack 2.9.0](https://github.com/deepset-ai/haystack/releases/tag/v2.9.0). Experimental features in the old version of this notebook are merged into Haystack core package.

## 📺 Watch Along

<iframe width="560" height="315" src="https://www.youtube.com/embed/QWx3OzW2Pvo?si=Zk-eW2sT_tOzBY_V" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Install dependencies

Install the latest versions of `haystack-ai` and `trafilatura`
"""
logger.info("# 🗞️ Newsletter Sending Agent with Haystack Tools")

# ! pip install haystack-ai trafilatura

"""
#### **Importing Features**

In this demo, we are using Haystack's latest features: [`Tool`](https://docs.haystack.deepset.ai/docs/tool), [`ToolInvoker`](https://docs.haystack.deepset.ai/docs/toolinvoker) with extended [`ChatMessage`](https://docs.haystack.deepset.ai/docs/chatmessage) and [`OllamaFunctionCallingAdapterChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator).
"""
logger.info("#### **Importing Features**")

# from getpass import getpass


"""
## Hacker News Fetcher Tool

In a previous article and recipe, we had shown how you can create a custom component for Haystack called the `HackerNewsFetcher`.

Here, we are doing something very similar, but instead we are creating a function and using that as a `Tool` instead.

📚 [Hacker News Summaries with Custom Components](https://haystack.deepset.ai/cookbook/hackernews-custom-component-rag?utm_campaign=developer-relations&utm_source=tools-livestream)

This tool expects `top_k` as input, and returns that many of the _current_ top stories on Hacker News 🚀
"""
logger.info("## Hacker News Fetcher Tool")

def hacker_news_fetcher(top_k: int = 3):
    newest_list = requests.get(url='https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty')
    urls = []
    articles = []
    for id_ in newest_list.json()[0:top_k]:
        article = requests.get(url=f"https://hacker-news.firebaseio.com/v0/item/{id_}.json?print=pretty")
        if 'url' in article.json():
            urls.append(article.json()['url'])
        elif 'text' in article.json():
            articles.append(article.json()['text'])

    for url in urls:
        try:
            downloaded = fetch_url(url)
            text = extract(downloaded)
            if text is not None:
                articles.append(text[:500])
        except Exception as e:
            logger.debug(e)
            logger.debug(f"Couldn't download {url}, skipped")

    return articles

hacker_news_fetcher_tool = Tool(name="hacker_news_fetcher",
                                description="Fetch the top k articles from hacker news",
                                function=hacker_news_fetcher,
                                parameters={
                                    "type": "object",
                                    "properties": {
                                        "top_k": {
                                            "type": "integer",
                                            "description": "The number of articles to fetch"
                                        }
                                    },
                                })

"""
## Newsletter generation Pipeline and Tool

For the Newsletter gnereation tool, we will be creating a Haystack pipeline, and making our pipeline itself a tool.

Our tool will expect the following inputs:

- `articles`: Content to base the newsletter off of
- `target_people`: The audience we want to target, for example "engineers" may be our target audience
- `n_words`: The number of words we want to limit our newsletter to
"""
logger.info("## Newsletter generation Pipeline and Tool")

# if not "OPENAI_API_KEY" in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter your OllamaFunctionCalling API key: ")

template = [ChatMessage.from_user("""
Create a entertaining newsletter for {{target_people}} based on the following articles.
The newsletter should be well structured, with a unique angle and a maximum of {{n_words}} words.

Articles:
{% for article in articles %}
    {{ article }}
    ---
{% endfor %}
""")]

newsletter_pipe = Pipeline()
newsletter_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
newsletter_pipe.add_component("llm", OllamaFunctionCallingAdapterChatGenerator(model="llama3.2"))
newsletter_pipe.connect("prompt_builder", "llm")

def newsletter_pipeline_func(articles: List[str], target_people: str = "programmers", n_words: int = 100):
    result = newsletter_pipe.run({"prompt_builder": {"articles": articles, "target_people": target_people, "n_words": n_words}})

    return {"reply": result["llm"]["replies"][0].text}

newsletter_tool = Tool(name="newsletter_generator",
                          description="Generate a newsletter based on some articles",
                            function=newsletter_pipeline_func,
                            parameters={
                                "type": "object",
                                "properties": {
                                    "articles": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "description": "The articles to base the newsletter on",
                                        }
                                    },
                                    "target_people": {
                                        "type": "string",
                                        "description": "The target audience for the newsletter",
                                    },
                                    "n_words": {
                                        "type": "integer",
                                        "description": "The number of words to summarize the newsletter to",
                                    }
                                },
                                "required": ["articles"],
                            })

"""
## Send Email Tool

Here, we are creating a Gmail tool. You login with your gmail account, allowing the final Agent to send emails from your email, to others.

> ⚠️ Note: To be able to use the gmail too, you have to create an app password for your Gmail account, which will be the sender. You can delete this after.

To configure our `email` Tool, you have to provide the following information about the sender email account 👇
"""
logger.info("## Send Email Tool")

if not "NAME" in os.environ:
    os.environ["NAME"] = input("What's your name? ")
if not "SENDER_EMAIL" in os.environ:
#     os.environ["SENDER_EMAIL"] = getpass("Enter your Gmail e-mail: ")
if not "GMAIL_APP_PASSWORD" in os.environ:
#     os.environ["GMAIL_APP_PASSWORD"] = getpass("Enter your Gmail App Password: ")

"""
Next, we create a `Tool` that expects the following input:
- `receiver`: The email address that we want to send an email to
- `body`: The body of the email
- `subject`: The subject line for the email.
"""
logger.info("Next, we create a `Tool` that expects the following input:")


def send_email(receiver: str, body: str, subject: str):
  msg = MIMEText(body)
  sender_email = os.environ['SENDER_EMAIL']
  sender_name = os.environ['NAME']
  sender = f"{sender_name} <{sender_email}>"
  msg['Subject'] = subject
  msg['From'] = sender
  port = 465  # For SSL
  smtp_server = "smtp.gmail.com"
  password = os.environ["GMAIL_APP_PASSWORD"]
  context = ssl.create_default_context()
  with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
      server.login(sender_email, password)
      server.sendmail(sender_email, receiver, msg.as_string())
  return 'Email sent!'

email_tool = Tool(name="email",
                  description="Send emails with specific content",
                  function=send_email,
                  parameters={
                      "type": "object",
                      "properties": {
                          "receiver": {
                              "type": "string",
                              "description": "The email of the receiver"
                          },
                          "body": {
                              "type": "string",
                              "description": "The content of the email"
                          },
                          "subject": {
                              "type": "string",
                              "description": "The subject of the email"
                          }
                      },
                  })

"""
## Newsletter Sending Chat Agent

Now, we build a Newsletter creating chat agent which we can use to ask for newsletters, as well as sending them to given email addresses.
"""
logger.info("## Newsletter Sending Chat Agent")

chat_generator = OllamaFunctionCallingAdapterChatGenerator(tools=[hacker_news_fetcher_tool, newsletter_tool, email_tool])

tool_invoker = ToolInvoker(tools=[hacker_news_fetcher_tool, newsletter_tool, email_tool])

messages = [
        ChatMessage.from_system(
            """Prepare a tool call if needed, otherwise use your knowledge to respond to the user.
            If the invocation of a tool requires the result of another tool, prepare only one call at a time.

            Each time you receive the result of a tool call, ask yourself: "Am I done with the task?".
            If not and you need to invoke another tool, prepare the next tool call.
            If you are done, respond with just the final result."""
        )
    ]

while True:
    user_input = input("\n\nwaiting for input (type 'exit' or 'quit' to stop)\n🧑: ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        break
    messages.append(ChatMessage.from_user(user_input))

    while True:
        logger.debug("⌛ iterating...")

        replies = chat_generator.run(messages=messages)["replies"]
        messages.extend(replies)

        if not replies[0].tool_calls:
            break
        tool_calls = replies[0].tool_calls

        for tc in tool_calls:
            logger.debug("\n TOOL CALL:")
            logger.debug(f"\t{tc.id}")
            logger.debug(f"\t{tc.tool_name}")
            for k,v in tc.arguments.items():
                v_truncated = str(v)[:50]
                logger.debug(f"\t{k}: {v_truncated}{'' if len(v_truncated) == len(str(v)) else '...'}")

        tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
        messages.extend(tool_messages)


    logger.debug(f"🤖: {messages[-1].text}")

"""
## Extras: Converting Tools

### Convert functions into Tools
"""
logger.info("## Extras: Converting Tools")


"""
Writing the JSON schema is not fun... 🤔
"""
logger.info("Writing the JSON schema is not fun... 🤔")

def newsletter_pipeline_func(articles: List[str], target_people: str = "programmers", n_words: int = 100):
    result = newsletter_pipe.run({"prompt_builder": {"articles": articles, "target_people": target_people, "n_words": n_words}})

    return {"reply": result["llm"]["replies"][0].text}

newsletter_tool = Tool(name="newsletter_generator",
                          description="Generate a newsletter based on some articles",
                            function=newsletter_pipeline_func,
                            parameters={
                                "type": "object",
                                "properties": {
                                    "articles": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "description": "The articles to include in the newsletter",
                                        }
                                    },
                                    "target_people": {
                                        "type": "string",
                                        "description": "The target audience for the newsletter",
                                    },
                                    "n_words": {
                                        "type": "integer",
                                        "description": "The number of words to summarize the newsletter to",
                                    }
                                },
                                "required": ["articles"],
                            })

"""
We can do this instead 👇
"""
logger.info("We can do this instead 👇")


def newsletter_pipeline_func(
    articles: Annotated[List[str], "The articles to include in the newsletter"],
    target_people: Annotated[str, "The target audience for the newsletter"] = "programmers",
    n_words: Annotated[int, "The number of words to summarize the newsletter to"] = 100
    ):
    """Generate a newsletter based on some articles"""

    result = newsletter_pipe.run({"prompt_builder": {"articles": articles, "target_people": target_people, "n_words": n_words}})

    return {"reply": result["llm"]["replies"][0].text}

newsletter_tool = create_tool_from_function(newsletter_pipeline_func)

pp(newsletter_tool, width=200)

"""
### Convert Pre-Existing Tools into Haystack Tools

Haystack is quite flexible. This means if you have tools already defined elsewhere, you are able to convert them to Haystack tools. For example,
 [LangChain has several interesting tools](https://python.langchain.com/docs/integrations/tools/) that we can seamlessly convert into Haystack tools.
"""
logger.info("### Convert Pre-Existing Tools into Haystack Tools")

# ! pip install langchain-community


def convert_langchain_tool_to_haystack_tool(langchain_tool):
    tool_name = langchain_tool.name
    tool_description = langchain_tool.description

    def invocation_adapter(**kwargs):
        return langchain_tool.invoke(input=kwargs)

    tool_function = invocation_adapter

    model_fields = langchain_tool.args_schema.model_fields

    fields = {name: (field.annotation, field.default) for name, field in model_fields.items()}
    descriptions = {name: field.description for name, field in model_fields.items()}

    model = create_model(tool_name, **fields)
    schema = model.model_json_schema()

    _remove_title_from_schema(schema)

    for name, description in descriptions.items():
        if name in schema["properties"]:
            schema["properties"][name]["description"] = description

    return Tool(name=tool_name, description=tool_description, parameters=schema, function=tool_function)

toolkit = FileManagementToolkit(
    root_dir="/"
)  # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()

langchain_listdir_tool = toolkit.get_tools()[-1]

haystack_listdir_tool = convert_langchain_tool_to_haystack_tool(langchain_listdir_tool)


chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2", tools=[haystack_listdir_tool])
tool_invoker = ToolInvoker(tools=[haystack_listdir_tool])

user_message = ChatMessage.from_user("List the files in /content/sample_data")

replies = chat_generator.run(messages=[user_message])["replies"]

if replies[0].tool_calls:

    tool_messages = tool_invoker.run(messages=replies)["tool_messages"]

    messages = [user_message] + replies + tool_messages
    final_replies = chat_generator.run(messages=messages)["replies"]
    logger.debug(f"{final_replies[0].text}")

logger.info("\n\n[DONE]", bright=True)