from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.logger import logger
from langchain_community.document_transformers import DoctranTextTranslator
from langchain_core.documents import Document
import asyncio
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
# Doctran: language translation

Comparing documents through embeddings has the benefit of working across multiple languages. "Harrison says hello" and "Harrison dice hola" will occupy similar positions in the vector space because they have the same meaning semantically.

However, it can still be useful to use an LLM to **translate documents into other languages** before vectorizing them. This is especially helpful when users are expected to query the knowledge base in different languages, or when state-of-the-art embedding models are not available for a given language.

We can accomplish this using the [Doctran](https://github.com/psychic-api/doctran) library, which uses Ollama's function calling feature to translate documents between languages.
"""
logger.info("# Doctran: language translation")

# %pip install --upgrade --quiet  doctran



load_dotenv()

"""
## Input
This is the document we'll translate
"""
logger.info("## Input")

sample_text = """[Generated with ChatGPT]

Confidential Document - For Internal Use Only

Date: July 1, 2023

Subject: Updates and Discussions on Various Topics

Dear Team,

I hope this email finds you well. In this document, I would like to provide you with some important updates and discuss various topics that require our attention. Please treat the information contained herein as highly confidential.

Security and Privacy Measures
As part of our ongoing commitment to ensure the security and privacy of our customers' data, we have implemented robust measures across all our systems. We would like to commend John Doe (email: john.doe@example.com) from the IT department for his diligent work in enhancing our network security. Moving forward, we kindly remind everyone to strictly adhere to our data protection policies and guidelines. Additionally, if you come across any potential security risks or incidents, please report them immediately to our dedicated team at security@example.com.

HR Updates and Employee Benefits
Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: 049-45-5928) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: 418-492-3850, email: michael.johnson@example.com).

Marketing Initiatives and Campaigns
Our marketing team has been actively working on developing new strategies to increase brand awareness and drive customer engagement. We would like to thank Sarah Thompson (phone: 415-555-1234) for her exceptional efforts in managing our social media platforms. Sarah has successfully increased our follower base by 20% in the past month alone. Moreover, please mark your calendars for the upcoming product launch event on July 15th. We encourage all team members to attend and support this exciting milestone for our company.

Research and Development Projects
In our pursuit of innovation, our research and development department has been working tirelessly on various projects. I would like to acknowledge the exceptional work of David Rodriguez (email: david.rodriguez@example.com) in his role as project lead. David's contributions to the development of our cutting-edge technology have been instrumental. Furthermore, we would like to remind everyone to share their ideas and suggestions for potential new projects during our monthly R&D brainstorming session, scheduled for July 10th.

Please treat the information in this document with utmost confidentiality and ensure that it is not shared with unauthorized individuals. If you have any questions or concerns regarding the topics discussed, please do not hesitate to reach out to me directly.

Thank you for your attention, and let's continue to work together to achieve our goals.

Best regards,

Jason Fan
Cofounder & CEO
Psychic
jason@psychic.dev
"""

documents = [Document(page_content=sample_text)]
qa_translator = DoctranTextTranslator(language="spanish")

"""
## Output using Sync version
After translating a document, the result will be returned as a new document with the page_content translated into the target language
"""
logger.info("## Output using Sync version")

translated_document = qa_translator.transform_documents(documents)

logger.debug(translated_document[0].page_content)

"""
## Output using the Async version

After translating a document, the result will be returned as a new document with the page_content translated into the target language. The async version will improve performance when the documents are chunked in multiple parts. It will also make sure to return the output in the correct order.
"""
logger.info("## Output using the Async version")


result = await qa_translator.atransform_documents(documents)
logger.success(format_json(result))

logger.debug(result[0].page_content)

logger.info("\n\n[DONE]", bright=True)