from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.llms import EdenAI
from langchain_community.tools.edenai import (
EdenAiExplicitImageTool,
EdenAiObjectDetectionTool,
EdenAiParsingIDTool,
EdenAiParsingInvoiceTool,
EdenAiSpeechToTextTool,
EdenAiTextModerationTool,
EdenAiTextToSpeechTool,
)
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
# Eden AI

This Jupyter Notebook demonstrates how to use Eden AI tools with an Agent.

Eden AI is revolutionizing the AI landscape by uniting the best AI providers, empowering users to unlock limitless possibilities and tap into the true potential of artificial intelligence. With an all-in-one comprehensive and hassle-free platform, it allows users to deploy AI features to production lightning fast, enabling effortless access to the full breadth of AI capabilities via a single API. (website: https://edenai.co/ )


By including an Edenai tool in the list of tools provided to an Agent, you can grant your Agent the ability to do multiple tasks, such as:

- speech to text
- text to speech
- text explicit content detection 
- image explicit content detection
- object detection
- OCR invoice parsing
- OCR ID parsing


In this example, we will go through the process of utilizing the Edenai tools to create an Agent that can perform some of the tasks listed above.

---------------------------------------------------------------------------
Accessing the EDENAI's API requires an API key, 

which you can get by creating an account https://app.edenai.run/user/register  and heading here https://app.edenai.run/admin/account/settings

Once we have a key we'll want to set it as the environment variable ``EDENAI_API_KEY`` or you can pass the key in directly via the edenai_api_key named parameter when initiating the EdenAI tools, e.g. ``EdenAiTextModerationTool(edenai_)``
"""
logger.info("# Eden AI")

# %pip install --upgrade --quiet langchain-community



llm = EdenAI(
    feature="text", provider="ollama", params={"temperature": 0.2, "max_tokens": 250}
)

tools = [
    EdenAiTextModerationTool(providers=["ollama"], language="en"),
    EdenAiObjectDetectionTool(providers=["google", "api4ai"]),
    EdenAiTextToSpeechTool(providers=["amazon"], language="en", voice="MALE"),
    EdenAiExplicitImageTool(providers=["amazon", "google"]),
    EdenAiSpeechToTextTool(providers=["amazon"]),
    EdenAiParsingIDTool(providers=["amazon", "klippa"], language="en"),
    EdenAiParsingInvoiceTool(providers=["amazon", "google"], language="en"),
]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)

"""
## Example with text
"""
logger.info("## Example with text")

input_ = """i have this text : 'i want to slap you'
first : i want to know if this text contains explicit content or not .
second : if it does contain explicit content i want to know what is the explicit content in this text,
third : i want to make the text into speech .
if there is URL in the observations , you will always put it in the output (final answer) .
"""
result = agent_chain(input_)

"""
you can have more details of the execution by printing the result
"""
logger.info("you can have more details of the execution by printing the result")

result["output"]

result

"""
## Example with images
"""
logger.info("## Example with images")

input_ = """i have this url of an image : "https://static.javatpoint.com/images/objects.jpg"
first : i want to know if the image contain objects .
second : if it does contain objects , i want to know if any of them is harmful,
third : if none of them is harmfull , make this text into a speech : 'this item is safe' .
if there is URL in the observations , you will always put it in the output (final answer) .
"""
result = agent_chain(input_)

result["output"]

"""
you can have more details of the execution by printing the result
"""
logger.info("you can have more details of the execution by printing the result")

result

"""
## Example with OCR images
"""
logger.info("## Example with OCR images")

input_ = """i have this url of an id: "https://www.citizencard.com/images/citizencard-uk-id-card-2023.jpg"
i want to extract the information in it.
create a text welcoming the person by his name and make it into speech .
if there is URL in the observations , you will always put it in the output (final answer) .
"""
result = agent_chain(input_)

result["output"]

input_ = """i have this url of an invoice document: "https://app.edenai.run/assets/img/data_1.72e3bdcc.png"
i want to extract the information in it.
and answer these questions :
who is the customer ?
what is the company name ?
"""
result = agent_chain()

result["output"]

logger.info("\n\n[DONE]", bright=True)