# from codecs import ignore_errors
# from IPython.display import display
# from IPython.display import Markdown
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
import shutil

from jet.transformers.formatters import format_json


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

# SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")
# os.makedirs(SCREENSHOT_DIR, exist_ok=True)

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

system_message = """
You are an intelligent assistant equipped with tools for navigating the web.

You can use tools when appropriate, but not every task requires them — you also have strong reasoning and language capabilities.
If a request seems challenging, don't default to refusal due to perceived tool limitations. Instead, think creatively and attempt a solution using the skills you do have.
You are more capable than you might assume. Trust your abilities.
"""

def create_agent(agent_name: str, tool_names: list[str]) -> Agent:
    toolset = MCPToolset(server_info=server_info, tool_names=tool_names)
    logger.debug("Toolset:")
    logger.debug(format_json(toolset.tools))
    chat_generator = OllamaChatGenerator(model="llama3.2", agent_name=agent_name, verbose=False, generation_kwargs={"temperature": 0.1})

    agent = Agent(
        chat_generator=chat_generator,
        tools=toolset,
        system_prompt=system_message,
        exit_conditions=["text"],
        streaming_callback=print_streaming_chunk
    )

    return agent

# 1st agent toolset
tool_names1 = ["browser_navigate"]

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
def example1_find_us_news_from_guardian():
    logger.info("## Our first Browser Agent with URL navigation")

    """
    ### 📰 Find US-related news

    Let's start with a simple task: navigate to a website and filter US-related news.
    """
    logger.info("### 📰 Find US-related news")

    # Update Agent 1 task with screenshot
    messages = [
        ChatMessage.from_user("navigate to https://www.theguardian.com/world and list all US-related news")
    ]
    agent = create_agent("Agent1_News", tool_names=tool_names1)
    result = agent.run(messages=messages)
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_1.md")

def example2_find_notable_people_born_on_october_17():
    logger.info("### 📅 Who was born on this day?")

    messages = [
        ChatMessage.from_user("I was born on October 17th. Find five notable people born on the same day using Wikipedia.")
    ]
    agent = create_agent("Agent2_Birthday", tool_names=tool_names1)
    result = agent.run(messages=messages)
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_2.md")

# 2nd agent toolset
tool_names2 = ["browser_navigate", "browser_snapshot"]

def example3_compare_news_websites_nbc_bbc():
    logger.info("### 🗞️🗞️ Compare online news websites")
    # Update Agent 3 tasks with snapshots
    messages = [ChatMessage.from_user("""
    1. Navigate to https://www.nbcnews.com and capture an accessibility snapshot of the page.
    2. Navigate to https://www.bbc.com and capture an accessibility snapshot of the page.
    3. Identify news stories that appear on both sites.
    4. Your final response must contain only a Markdown table listing the shared news stories and their respective links from each site.
    """)]
    agent = create_agent("Agent3_Compare", tool_names=tool_names2)
    result = agent.run(messages=messages)
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_3.md")

def example4_find_github_contributor_contributions():
    logger.info("### 👨🏻‍💻 Find information about a GitHub contributor")
    messages = [ChatMessage.from_user("""
I'm trying to find how hard I have to work to get a repo in https://github.com/trending.
Can you navigate to the profile for the top author of the top trending repo,
and give me their total number of contributions over the last year?
""")
    ]
    agent = create_agent("Agent4_GitHub", tool_names=tool_names2)
    result = agent.run(messages=messages)
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_4.md")

tool_names3 = [
    "browser_navigate", "browser_snapshot", "browser_click",
    "browser_type", "browser_navigate_back", "browser_wait_for"
]

def example5_amazon_price():
    logger.info("### 🖱️ Find a product's price range on Amazon")
    agent = create_agent("Agent5_Amazon", tool_names=tool_names3)
    prompt = """
1. Go to Amazon Italy and find all available prices for the Logitech MX Master 3S mouse.
2. Exclude any items that:
- are not computer mice;
- are not the exact Logitech MX Master 3S model;
- do not display a price;
- are bundled with other products or accessories.
3. Your final message must only contain a Markdown table with two columns: Name and Price.
"""
    result = agent.run(messages=[ChatMessage.from_user(prompt)])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_5.md")

def example6_github_prs():
    logger.info("### 🖥️ GitHub exploration")
    agent = create_agent("Agent6_Prs", tool_names=tool_names3)
    prompt = "List some recent PRs merged by anakin87 on the deepset-ai/haystack repo on GitHub. Max 5."
    result = agent.run(messages=[ChatMessage.from_user(prompt)])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_6.md")

def example7_youtube_socials():
    logger.info("### ▶️ Find content creator social profiles from YouTube")
    agent = create_agent("Agent7_YouTube", tool_names=tool_names3)
    prompt = """
1. Open this YouTube video: https://www.youtube.com/watch?v=axmaslLO4-4 and extract the channel author’s username.
2. Then, try to find all their social media profiles by searching the web using the username.
If you cannot perform a web search, try other available methods to find the profiles.
3. Return only the links to their social media accounts along with the platform names.
"""
    result = agent.run(messages=[ChatMessage.from_user(prompt)])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_7.md")

def example8_google_maps_location():
    logger.info("### 🗺️ Use Google Maps to find a location")
    agent = create_agent("Agent8_Maps", tool_names=tool_names3)
    result = agent.run(messages=[ChatMessage.from_user("Use Google Maps to find the deepset HQ in Berlin")])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_8.md")

def example9_public_transport():
    logger.info("### 🚂 🚌 Find public transportation travel options")
    agent = create_agent("Agent9_Transport", tool_names=tool_names3)
    prompt = """
1. Using Google Maps, find the next 3 available public transportation travel options from Paris to Berlin departing today.
2. For each option, provide a detailed description of the route (e.g., transfers, stations, duration).
3. Include a direct link to the corresponding Google Maps route for each travel option.
"""
    result = agent.run(messages=[ChatMessage.from_user(prompt)])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_9.md")

def example10_hf_image_generation():
    logger.info("### 🖼️ 🤗 Image generation via Hugging Face spaces")
    agent = create_agent("Agent10_Image", tool_names=tool_names3)
    prompt = """
1. Visit Hugging Face Spaces, search the Space named exactly "FLUX.1 [schnell]" and enter it.
2. Craft a detailed, descriptive prompt using your language skills to depict: "my holiday on Lake Como".
3. Use this prompt to generate an image within the Space.
4. After prompting, wait 5 seconds to allow the image to fully generate. Repeat until the image is generated.
5. Your final response must contain only the direct link to the generated image — no additional text.
"""
    result = agent.run(messages=[ChatMessage.from_user(prompt)])
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message_agent_10.md")

    # Try to download and save the image if possible
    response = result["last_message"].text
    try:
        # If the response is a URL, try to download the image
        import requests
        img_resp = requests.get(response)
        img_resp.raise_for_status()
        image = Image.open(BytesIO(img_resp.content))
        os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
        save_file(image, f"{OUTPUT_DIR}/images/image.png")
    except Exception as e:
        logger.error(f"Could not download or save image: {e}")

def example11_demo_video():
    logger.info("The video below demonstrates a similar local setup. It's fascinating to watch the Agent navigate the web similarly to a human.")

    url = "https://drive.google.com/drive/folders/1HyPdNxpzi3svmPVYGXak7mqAjVWztwsH"
    downloads_dir = f"{OUTPUT_DIR}/downloads"
    os.makedirs(downloads_dir, exist_ok=True)
    gdown.download_folder(url, quiet=True, output=downloads_dir)
    # Video.from_file('/content/browser_agent.mp4', autoplay=False, loop=False)

def main():
    example1_find_us_news_from_guardian()
    example2_find_notable_people_born_on_october_17()
    example3_compare_news_websites_nbc_bbc()
    example4_find_github_contributor_contributions()
    example5_amazon_price()
    example6_github_prs()
    example7_youtube_socials()
    example8_google_maps_location()
    example9_public_transport()
    example10_hf_image_generation()
    example11_demo_video()
    logger.info("## What's next?")
    logger.info("\n\n[DONE]", bright=True)

if __name__ == "__main__":
    main()