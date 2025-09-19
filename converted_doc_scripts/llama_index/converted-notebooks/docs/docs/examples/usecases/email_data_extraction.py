from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import ChatPromptTemplate
from llama_index.core import download_loader
from llama_index.core.llms import ChatMessage
from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
from llama_index.readers.file import UnstructuredReader
from pydantic import BaseModel, Field
from typing import List
import logging
import openai
import os
import shutil
import sys
import json


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/usecases/email_data_extraction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Email Data Extraction
OllamaFunctionCalling functions can be used to extract data from Email. This is another example of getting structured data from unstructured conent using LLamaIndex. 

The primary objective of this example is to transform raw email content into an easily interpretable JSON format, exemplifying a practical application of language models in data extraction. Extracted structued JSON data can then be used in any downstream application. 

We will use a sample email as shown in below image. This email mimics a typical daily communication sent by ARK Investment to its subscribers. This sample email includes detailed information about trades under their Exchange-Traded Funds (ETFs). By using this specific example, we aim to showcase how we can effectively extract and structure complex financial data from a real-world email scenario, transforming it into a comprehensible JSON format 

![Ark Daily Trades](../data/images/ark_email_sample.PNG "Sample Email of ARK Investment Daily trading")

### Add required packages 

You will need following libraries along with LlamaIndex 🦙.

- `unstructured[msg]`: A package for handling unstructured data, required to get content from `.eml` and `.msg` format.
"""
logger.info("# Email Data Extraction")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file
# %pip install llama-index-program-openai

# !pip install llama-index

# !pip install "unstructured[msg]"

"""
### Enable Logging and Set up OllamaFunctionCalling API Key

In this step, we set up logging to monitor the program's execution and debug if needed. We also configure the OllamaFunctionCalling API key, essential for utilizing OllamaFunctionCalling services. Replace "YOUR_KEY_HERE" with your actual OllamaFunctionCalling API key.
"""
logger.info("### Enable Logging and Set up OllamaFunctionCalling API Key")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
### Set Up Expected JSON Output Definition (JSON Schema)

Here we define a Python class named `EmailData` using the Pydantic library. This class models the structure of the data we expect to extract from emails, including sender, receiver, the date and time of the email, etfs having list of shares traded under that ETF.
"""
logger.info("### Set Up Expected JSON Output Definition (JSON Schema)")


class Instrument(BaseModel):
    """Datamodel for ticker trading details."""

    direction: str = Field(description="ticker trading - Buy, Sell, Hold etc")
    ticker: str = Field(
        description="Stock Ticker. 1-4 character code. Example: AAPL, TSLS, MSFT, VZ"
    )
    company_name: str = Field(
        description="Company name corresponding to ticker"
    )
    shares_traded: float = Field(description="Number of shares traded")
    percent_of_etf: float = Field(description="Percentage of ETF")


class Etf(BaseModel):
    """ETF trading data model"""

    etf_ticker: str = Field(
        description="ETF Ticker code. Example: ARKK, FSPTX"
    )
    trade_date: str = Field(description="Date of trading")
    stocks: List[Instrument] = Field(
        description="List of instruments or shares traded under this etf"
    )


class EmailData(BaseModel):
    """Data model for email extracted information."""

    etfs: List[Etf] = Field(
        description="List of ETFs described in email having list of shares traded under it"
    )
    trade_notification_date: str = Field(
        description="Date of trade notification"
    )
    sender_email_id: str = Field(description="Email Id of the email sender.")
    email_date_time: str = Field(description="Date and time of email")


"""
### Load content from .eml / .msg file

In this step, we will use the `UnstructuredReader` from the `llama-hub` to load the content of an .eml email file or .msg Outlook file. This file's contents are then stored in a variable for further processing.
"""
logger.info("### Load content from .eml / .msg file")


loader = UnstructuredReader()

eml_documents = loader.load_data("../data/email/ark-trading-jan-12-2024.eml")
email_content = eml_documents[0].text
logger.debug("\n\n Email contents")
logger.debug(email_content)

msg_documents = loader.load_data("../data/email/ark-trading-jan-12-2024.msg")
msg_content = msg_documents[0].text
logger.debug("\n\n Outlook contents")
logger.debug(msg_content)

"""
### Use LLM function to extract content in JSON format

In the final step, we utilize the `llama_index` package to create a prompt template for extracting insights from the loaded email. An instance of the `OllamaFunctionCalling` model is used to interpret the email content and extract the relevant information based on our predefined `EmailData` schema. The output is then converted to a dictionary format for easy viewing and processing.
"""
logger.info("### Use LLM function to extract content in JSON format")


prompt = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role="system",
            content=(
                "You are an expert assitant for extracting insights from email in JSON format. \n"
                "You extract data and returns it in JSON format, according to provided JSON schema, from given email message. \n"
                "REMEMBER to return extracted data only from provided email message."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Email Message: \n" "------\n" "{email_msg_content}\n" "------"
            ),
        ),
    ]
)

llm = OllamaFunctionCalling(model="llama3.2")

program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=EmailData,
    llm=llm,
    prompt=prompt,
    verbose=True,
)

output = program(email_msg_content=email_content)
logger.debug("Output JSON From .eml File: ")
logger.debug(json.dumps(output.dict(), indent=2))

"""
### For outlook message
"""
logger.info("### For outlook message")

output = program(email_msg_content=msg_content)

logger.debug("Output JSON from .msg file: ")
logger.debug(json.dumps(output.dict(), indent=2))

logger.info("\n\n[DONE]", bright=True)
