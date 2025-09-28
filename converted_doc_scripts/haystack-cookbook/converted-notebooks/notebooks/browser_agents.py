# from codecs import ignore_errors
# from IPython.display import display
from IPython.display import Markdown
from PIL import Image
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
# from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo
from io import BytesIO
# from ipywidgets import Video
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.file.utils import save_file
from jet.logger import logger
import gdown
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# 🕵️🌐 Build Browser Agents with Gemini + Playwright MCP

In many real-world scenarios, websites and applications do not offer APIs for programmatic access.

This is where Browser Agents become especially useful: they can interact with web pages just like a human would, by clicking buttons, filling out forms, scrolling, and extracting content.

In this notebook, we'll explore how to build Browser Agents that can perform various tasks, mostly focused on information gathering, and even extending to image generation.

🧰 Our stack:
- Haystack Agentic framework
- Google Gemini 2.5 Flash, a capable model with generous free tier
- Playwright MCP server, which offers the browser automation tools

## Setup

First, we'll install all the necessary dependencies:
- Haystack integrations: [MCP](https://docs.haystack.deepset.ai/docs/mcptoolset) and [Google GenAI](https://docs.haystack.deepset.ai/docs/googlegenaichatgenerator)
- Playwright dependencies: these are needed to successfully run it on Colab
"""
logger.info("# 🕵️🌐 Build Browser Agents with Gemini + Playwright MCP")

# ! uv pip install mcp-haystack google-genai-haystack gdown

# ! npm i -D @playwright/test && npx playwright install --with-deps chrome

"""
To get a free Gemini API key, visit [Google AI Studio](https://aistudio.google.com/).
"""
logger.info("To get a free Gemini API key, visit [Google AI Studio](https://aistudio.google.com/).")

# from getpass import getpass

# os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API Key: ")

"""
## Start Playwright MCP server and create a Toolset

We first start the [Playwright MCP server](https://github.com/microsoft/playwright-mcp), which will give us the tools needed for Web navigation and interactions.

Note that we prepend `nohup` to run the server in Colab in a non-blocking way, and also some other options are required for it to work correctly in a Colab environment.
"""
logger.info("## Start Playwright MCP server and create a Toolset")

# ! nohup npx @playwright/mcp@latest  --headless --isolated --no-sandbox --port 8931 > playwright.log &

"""
After a few moments, if the server started correctly, we should see a successful log message.
"""
logger.info("After a few moments, if the server started correctly, we should see a successful log message.")

# ! cat playwright.log

"""
We then create a [`MCPToolset`](https://docs.haystack.deepset.ai/docs/mcptoolset) connected to the server we started.

📌 When working with tools for LLMs, it's a good practice to only select the ones you actually need. This helps avoid confusing the LLM with too many options.

For that reason, we'll start with just a single tool: `browser_navigate`, which navigates to a given URL.


You can find the full list of tools available in the [Playwright MCP server documentation](https://github.com/microsoft/playwright-mcp?tab=readme-ov-file#tools).
"""
logger.info("We then create a [`MCPToolset`](https://docs.haystack.deepset.ai/docs/mcptoolset) connected to the server we started.")


server_info = StreamableHttpServerInfo(url="http://localhost:8931/mcp")

toolset = MCPToolset(server_info=server_info, tool_names=["browser_navigate"])

logger.debug(toolset.tools)

"""
## Our first Browser Agent with URL navigation

We can now build the Browser Agent, using the [Haystack Agent component](https://docs.haystack.deepset.ai/docs/agent).

We need to specify:
- A Chat Generator, the LLM (`gemini-2.5-flash`).
- A good system message. This model is powerful, but it can sometimes be overly cautious and refuse to perform tasks it is capable of, so we have included a system message to encourage it.
- Tools: we pass the Toolset defined earlier.

We also specify some optional parameters:
- `exit_conditions=["text"]`. The Agent will exit as soon as the LLM replies only with a text response and no tool calls.
- `streaming_callback=print_streaming_chunk`. Using the utility function, text and Tool Calls coming from the LLM will be streamed, along with Tool Call Results from Tools.
"""
logger.info("## Our first Browser Agent with URL navigation")


chat_generator = OllamaChatGenerator(model="qwen3:4b-q4_K_M", agent_name="Agent_1")

system_message = """
You are an intelligent assistant equipped with tools for navigating the web.

You can use tools when appropriate, but not every task requires them — you also have strong reasoning and language capabilities.
If a request seems challenging, don't default to refusal due to perceived tool limitations. Instead, think creatively and attempt a solution using the skills you do have.
You are more capable than you might assume. Trust your abilities.
"""

agent = Agent(
        chat_generator=chat_generator,
        tools=toolset,
        system_prompt=system_message,
        exit_conditions=["text"],
        streaming_callback=print_streaming_chunk
)

"""
### 📰 Find US-related news

Let's start with a simple task: navigate to a website and filter US-related news.
"""
logger.info("### 📰 Find US-related news")

messages = [
    ChatMessage.from_user("navigate to https://www.theguardian.com/world and list all US-related news")
]
result = agent.run(messages=messages)

"""
Nice.

### 📅 Who was born on this day?

Another simple task. Let's see how our agent behaves.
"""
logger.info("### 📅 Who was born on this day?")

messages = [
    ChatMessage.from_user("I was born on October 17th. Find five notable people born on the same day using Wikipedia.")
]
result = agent.run(messages=messages)

save_file(Markdown(result["last_message"].text), f"{OUTPUT_DIR}/results/last_message_agent_1.md")

"""
## Browser Agent with URL navigation + accessibility snapshot

Sometimes the raw version of a web page is not easily interpretable or misses relevant information. In these cases, a textual accessibility snapshot can help the LLM.

According to the Playwright MCP server maintainers, this approach often works better than providing screenshots.

Let's rebuild our agent.
"""
logger.info("## Browser Agent with URL navigation + accessibility snapshot")

toolset = MCPToolset(server_info=server_info, tool_names=["browser_navigate", "browser_snapshot"])

chat_generator = OllamaChatGenerator(model="qwen3:4b-q4_K_M", agent_name="Agent_2")

agent = Agent(
        chat_generator=chat_generator,
        tools=toolset,
        system_prompt=system_message,
        exit_conditions=["text"],
        streaming_callback=print_streaming_chunk
)

"""
### 🗞️🗞️ Compare online news websites

Let's now use our agent to find similar news stories appearing on different web sites.
"""
logger.info("### 🗞️🗞️ Compare online news websites")

messages = [ChatMessage.from_user("""
1. Visit www.nbcnews.com
2. Visit www.bbc.com
3. Identify news stories that appear on both sites.
4. Your final response must contain only a Markdown table listing the shared news stories and their respective links from each site.
""")
]
result = agent.run(messages=messages)

# display(Markdown(result["last_message"].text))
save_file(Markdown(result["last_message"].text), f"{OUTPUT_DIR}/results/last_message_agent_2.md")

"""
Definitely not bad!

If you look at streamed tool calls, you can also notice that the agent is sometimes able to correct its own actions leading to errors.

### 👨🏻‍💻 Find information about a GitHub contributor

This example is not original: I took it from a [Hugging Face tutorial](https://huggingface.co/docs/smolagents/en/examples/web_browser), but it's nice.
"""
logger.info("### 👨🏻‍💻 Find information about a GitHub contributor")

messages = [ChatMessage.from_user("""
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo,
and give me their total number of contributions over the last year?
""")
]

result = agent.run(messages=messages)

"""
## More tools for more advanced Browser Agents!

To unlock advanced use cases, we need more tools: the ability to click, type something, navigate back and wait.
"""
logger.info("## More tools for more advanced Browser Agents!")

toolset = MCPToolset(server_info=server_info,
                     tool_names=["browser_navigate", "browser_snapshot",
                                "browser_click", "browser_type",
                                 "browser_navigate_back","browser_wait_for",
                                ])

chat_generator = OllamaChatGenerator(model="qwen3:4b-q4_K_M", agent_name="Agent_3")

agent = Agent(
        chat_generator=chat_generator,
        tools=toolset,
        system_prompt=system_message,
        exit_conditions=["text"],
        streaming_callback=print_streaming_chunk
)

"""
### 🖱️ Find a product's price range on Amazon

I want to buy a new mouse. Let's use our agent to discover price ranges on Amazon for this product.
"""
logger.info("### 🖱️ Find a product's price range on Amazon")

prompt="""
1. Go to Amazon Italy and find all available prices for the Logitech MX Master 3S mouse.
2. Exclude any items that:
- are not computer mice;
- are not the exact Logitech MX Master 3S model;
- do not display a price;
- are bundled with other products or accessories.
3. Your final message must only contain a Markdown table with two columns: Name and Price.
"""

result = agent.run(messages=[ChatMessage.from_user(prompt)])


# display(Markdown(result["last_message"].text))
save_file(Markdown(result["last_message"].text), f"{OUTPUT_DIR}/results/last_message_agent_3.md")

"""
### 🖥️ GitHub exploration
"""
logger.info("### 🖥️ GitHub exploration")

prompt = "List some recent PRs merged by anakin87 on the deepset-ai/haystack repo on GitHub. Max 5."
result = agent.run(messages=[ChatMessage.from_user(prompt)])

"""
### ▶️ Find content creator social profiles from YouTube

Starting from a YouTube video, look for the social profiles of a creator.
Since sometimes search engines block automation tools, we ask the model to find intelligent alternative ways to look for information.
"""
logger.info("### ▶️ Find content creator social profiles from YouTube")

prompt="""
1. Open this YouTube video: https://www.youtube.com/watch?v=axmaslLO4-4 and extract the channel author’s username.
2. Then, try to find all their social media profiles by searching the web using the username.
If you cannot perform a web search, try other available methods to find the profiles.
3. Return only the links to their social media accounts along with the platform names.
"""

result = agent.run(messages=[ChatMessage.from_user(prompt)])

"""
### 🗺️ Use Google Maps to find a location
"""
logger.info("### 🗺️ Use Google Maps to find a location")

result = agent.run(messages=[ChatMessage.from_user("Use Google Maps to find the deepset HQ in Berlin")])

"""
### 🚂 🚌 Find public transportation travel options

Using Google Maps, let's ask a harder question, which requires more navigation.
"""
logger.info("### 🚂 🚌 Find public transportation travel options")

prompt = """
1. Using Google Maps, find the next 3 available public transportation travel options from Paris to Berlin departing today.
2. For each option, provide a detailed description of the route (e.g., transfers, stations, duration).
3. Include a direct link to the corresponding Google Maps route for each travel option.
"""

result = agent.run(messages=[ChatMessage.from_user(prompt)])

"""
Ready to go!

### 🖼️ 🤗 Image generation via Hugging Face spaces

Let's now try something more advanced: access a Hugging Face space to generate an image; note that our agent is also asked to create a good prompt for the image generation model.
"""
logger.info("### 🖼️ 🤗 Image generation via Hugging Face spaces")

prompt="""
1. Visit Hugging Face Spaces, search the Space named exactly "FLUX.1 [schnell]" and enter it.
2. Craft a detailed, descriptive prompt using your language skills to depict: "my holiday on Lake Como".
3. Use this prompt to generate an image within the Space.
4. After prompting, wait 5 seconds to allow the image to fully generate. Repeat until the image is generated.
5. Your final response must contain only the direct link to the generated image — no additional text.
"""

result = agent.run(messages=[ChatMessage.from_user(prompt)])


response = requests.get(result["last_message"].text)
image = Image.open(BytesIO(response.content))
save_file(image, f"{OUTPUT_DIR}/images/image.png")

"""
🏞️ Impressive!

The video below demonstrates a similar local setup. It's fascinating to watch the Agent navigate the web similarly to a human.
"""
logger.info("The video below demonstrates a similar local setup. It's fascinating to watch the Agent navigate the web similarly to a human.")


url = "https://drive.google.com/drive/folders/1HyPdNxpzi3svmPVYGXak7mqAjVWztwsH"

downloads_dir = f"{OUTPUT_DIR}/downloads"
os.makedirs(downloads_dir, ignore_errors=True)
gdown.download_folder(url, quiet=True, output=downloads_dir)

# Video.from_file('/content/browser_agent.mp4', autoplay=False, loop=False)

"""
## What's next?

When I first started exploring Browser Agents, I expected to hit roadblocks  quickly. But with the right framework, a solid LLM, and the Playwright MCP server (even without screenshot capabilities), you can actually get quite far. This notebook shows some fun and sometimes impressive demos that emerged from that process.

We also want to explore how to move these agents from notebooks to production: deploying in Docker environment, exposing via an API, and integrating into a simple UI. More material on this will come in the future.

Of course, there are use cases where vision capabilities or authentication handling are essential. If you are interested in support for those features, feel free to open a [discussion on GitHub](https://github.com/deepset-ai/haystack/discussions).

*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*
"""
logger.info("## What's next?")

logger.info("\n\n[DONE]", bright=True)