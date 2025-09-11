from jet.logger import logger
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
# Yeager.ai

This page covers how to use [Yeager.ai](https://yeager.ai) to generate LangChain tools and agents.

## What is Yeager.ai?
Yeager.ai is an ecosystem designed to simplify the process of creating AI agents and tools.

It features yAgents, a No-code LangChain Agent Builder, which enables users to build, test, and deploy AI solutions with ease. Leveraging the LangChain framework, yAgents allows seamless integration with various language models and resources, making it suitable for developers, researchers, and AI enthusiasts across diverse applications.

## yAgents
Low code generative agent designed to help you build, prototype, and deploy Langchain tools with ease.

### How to use?

# pip install yeagerai-agent
yeagerai-agent

Go to http://127.0.0.1:7860

This will install the necessary dependencies and set up yAgents on your system. After the first run, yAgents will create a .env file where you can input your Ollama API key. You can do the same directly from the Gradio interface under the tab "Settings".

# `OPENAI_API_KEY=<your_ollama_api_key_here>`

We recommend using GPT-4,. However, the tool can also work with GPT-3 if the problem is broken down sufficiently.

### Creating and Executing Tools with yAgents
yAgents makes it easy to create and execute AI-powered tools. Here's a brief overview of the process:
1. Create a tool: To create a tool, provide a natural language prompt to yAgents. The prompt should clearly describe the tool's purpose and functionality. For example:
`create a tool that returns the n-th prime number`

2. Load the tool into the toolkit: To load a tool into yAgents, simply provide a command to yAgents that says so. For example:
`load the tool that you just created it into your toolkit`

3. Execute the tool: To run a tool or agent, simply provide a command to yAgents that includes the name of the tool and any required parameters. For example:
`generate the 50th prime number`

You can see a video of how it works [here](https://www.youtube.com/watch?v=KA5hCM3RaWE).

As you become more familiar with yAgents, you can create more advanced tools and agents to automate your work and enhance your productivity.

For more information, see [yAgents' Github](https://github.com/yeagerai/yeagerai-agent) or our [docs](https://yeagerai.gitbook.io/docs/general/welcome-to-yeager.ai)
"""
logger.info("# Yeager.ai")

logger.info("\n\n[DONE]", bright=True)