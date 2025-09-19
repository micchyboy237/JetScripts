from google.colab import userdata
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.dataclasses import ChatMessage, ChatRole
from jet.logger import CustomLogger
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Function Calling with OllamaFunctionCallingAdapterChatGenerator 📞

> ⚠️ As of Haystack 2.9.0, this recipe has been deprecated. For the same example, follow [Tutorial: Building a Chat Agent with Function Calling](https://haystack.deepset.ai/tutorials/40_building_chat_application_with_function_calling)

*Notebook by Bilge Yucel ([LI](https://www.linkedin.com/in/bilge-yucel/) & [X (Twitter)](https://twitter.com/bilgeycl))*

A guide to understand function calling and how to use OllamaFunctionCalling function calling feature with [Haystack](https://github.com/deepset-ai/haystack).

📚 Useful Sources:
* [OllamaFunctionCallingAdapterChatGenerator Docs](https://docs.haystack.deepset.ai/v2.0/docs/openaichatgenerator)
* [OllamaFunctionCallingAdapterChatGenerator API Reference](https://docs.haystack.deepset.ai/v2.0/reference/generator-api#openaichatgenerator)

## Overview

Here are some use cases of function calling from [OllamaFunctionCalling Docs](https://platform.openai.com/docs/guides/function-calling):
* **Create assistants that answer questions by calling external APIs** (e.g. like ChatGPT Plugins)
e.g. define functions like send_email(to: string, body: string), or get_current_weather(location: string, unit: 'celsius' | 'fahrenheit')
* **Convert natural language into API calls**
e.g. convert "Who are my top customers?" to get_customers(min_revenue: int, created_before: string, limit: int) and call your internal API
* **Extract structured data from text**
e.g. define a function called extract_data(name: string, birthday: string), or sql_query(query: string)

## Set up the Development Environment
"""
logger.info("# Function Calling with OllamaFunctionCallingAdapterChatGenerator 📞")

# %%bash

pip install haystack-ai==2.8.1

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') or getpass("OPENAI_API_KEY: ")

"""
## Learn about the OllamaFunctionCallingAdapterChatGenerator

`OllamaFunctionCallingAdapterChatGenerator` is a component that supports the function calling feature of OllamaFunctionCalling.

The way to communicate with `OllamaFunctionCallingAdapterChatGenerator` is through [`ChatMessage`](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage) list. Therefore, create a `ChatMessage` with "USER" role using `ChatMessage.from_user()` and send it to OllamaFunctionCallingAdapterChatGenerator:
"""
logger.info("## Learn about the OllamaFunctionCallingAdapterChatGenerator")


client = OllamaFunctionCallingAdapterChatGenerator()
response = client.run(
    [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
)
logger.debug(response)

"""
### Basic Streaming

OllamaFunctionCallingAdapterChatGenerator supports streaming, provide a `streaming_callback` function and run the client again to see the difference.
"""
logger.info("### Basic Streaming")


client = OllamaFunctionCallingAdapterChatGenerator(streaming_callback=print_streaming_chunk)
response = client.run(
    [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
)

"""
## Function Calling with OllamaFunctionCallingAdapterChatGenerator

We'll try to recreate the [example on OllamaFunctionCalling docs](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models).

### Define a Function

We'll define a `get_current_weather` function that mocks a Weather API call in the response:
"""
logger.info("## Function Calling with OllamaFunctionCallingAdapterChatGenerator")

def get_current_weather(location: str, unit: str = "celsius"):
  return {"weather": "sunny", "temperature": 21.8, "unit": unit}

available_functions = {
  "get_current_weather": get_current_weather
}

"""
### Create the `tools`

We'll then add information about this function to our `tools` list by following [OllamaFunctionCalling's tool schema](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)
"""
logger.info("### Create the `tools`")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "unit"],
            },
        }
    }
]

"""
### Run OllamaFunctionCallingAdapterChatGenerator with tools

We'll pass the list of tools in the `run()` method as `generation_kwargs`.

Let's define messages and run the generator:
"""
logger.info("### Run OllamaFunctionCallingAdapterChatGenerator with tools")


messages = []
messages.append(ChatMessage.from_system("Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."))
messages.append(ChatMessage.from_user("What's the weather like in Berlin?"))

client = OllamaFunctionCallingAdapterChatGenerator(streaming_callback=print_streaming_chunk)
response = client.run(
    messages=messages,
    generation_kwargs={"tools":tools}
)

"""
It's a function call! 📞 The response gives us information about the function name and arguments to use to call that function:
"""
logger.info("It's a function call! 📞 The response gives us information about the function name and arguments to use to call that function:")

response

"""
Optionally, add the message with function information to the message list
"""
logger.info("Optionally, add the message with function information to the message list")

messages.append(response["replies"][0])

"""
See how we can extract the `function_name` and `function_args` from the message
"""
logger.info("See how we can extract the `function_name` and `function_args` from the message")


function_call = json.loads(response["replies"][0].content)[0]
function_name = function_call["function"]["name"]
function_args = json.loads(function_call["function"]["arguments"])
logger.debug("function_name:", function_name)
logger.debug("function_args:", function_args)

"""
### Make a Tool Call

Let's locate the corresponding function for `function_name` in our `available_functions` dictionary and use `function_args` when calling it. Once we receive the response from the tool, we'll append it to our `messages` for later sending to OllamaFunctionCalling.
"""
logger.info("### Make a Tool Call")

function_to_call = available_functions[function_name]
function_response = function_to_call(**function_args)
function_message = ChatMessage.from_function(content=json.dumps(function_response), name=function_name)
messages.append(function_message)

"""
Make the last call to OllamaFunctionCalling with response coming from the function and see how OllamaFunctionCalling uses the provided information
"""
logger.info("Make the last call to OllamaFunctionCalling with response coming from the function and see how OllamaFunctionCalling uses the provided information")

response = client.run(
    messages=messages,
    generation_kwargs={"tools":tools}
)

"""
## Improve the Example

Let's add more tool to our example and improve the user experience 👇

We'll add one more tool `use_haystack_pipeline` for OllamaFunctionCalling to use when there's a question about countries and capitals:
"""
logger.info("## Improve the Example")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "unit"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "use_haystack_pipeline",
            "description": "Use for search about countries and capitals",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to use in the search. Infer this from the user's message",
                    },
                },
                "required": ["query"]
            },
        }
    },
]

def get_current_weather(location: str, unit: str = "celsius"):
  return {"weather": "sunny", "temperature": 21.8, "unit": unit}

def use_haystack_pipeline(query: str):
  return {"documents": "Cutopia is the capital of Utopia", "query": query}

available_functions = {
  "get_current_weather": get_current_weather,
  "use_haystack_pipeline": use_haystack_pipeline,
}

"""
### Start the Application

Have fun having a chat with OllamaFunctionCalling 🎉

Example queries you can try:
* "***What's the capital of Utopia***", "***Is it sunny there?***": To test the messages are being recorded and sent
* "***What's the weather like in the capital of Utopia?***": To force two function calls
* "***What's the weather like today?***": To force OllamaFunctionCalling to ask more clarification
"""
logger.info("### Start the Application")


messages = []
messages.append(ChatMessage.from_system("Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."))

logger.debug(messages[-1].content)

while True:
  if response and response["replies"][0].meta["finish_reason"] == 'tool_calls':
    function_calls = json.loads(response["replies"][0].content)
    for function_call in function_calls:
      function_name = function_call["function"]["name"]
      function_to_call = available_functions[function_name]
      function_args = json.loads(function_call["function"]["arguments"])

      function_response = function_to_call(**function_args)
      function_message = ChatMessage.from_function(content=json.dumps(function_response), name=function_name)
      messages.append(function_message)

  else:
    if not messages[-1].is_from(ChatRole.SYSTEM):
      messages.append(ChatMessage.from_assistant(response["replies"][0].content))

    user_input = input("INFO: Type 'exit' or 'quit' to stop\n")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
      break
    else:
      messages.append(ChatMessage.from_user(user_input))

  response = client.run(
    messages=messages,
    generation_kwargs={"tools":tools}
  )

"""
### Print the summary of the conversation

This part can help you understand the message order
"""
logger.info("### Print the summary of the conversation")

logger.debug("\n=== SUMMARY ===")
for m in messages:
  logger.debug(f"\n - {m.content}")

logger.info("\n\n[DONE]", bright=True)