from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.prompts import RichPromptTemplate
from pydantic import BaseModel
from pydantic import Field
from typing import Dict
from typing import Optional
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Structured Input for LLMs

It has been observed that most LLMs perfom better when prompted with XML-like content (you can see it in [Anthropic's prompting guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags), for instance).

We could refer to this kind of prompting as _structured input_, and LlamaIndex offers you the possibility of chatting with LLMs exactly through this technique - let's go through an example in this notebook!

### 1. Install Needed Dependencies

> _Make sure to have `llama-index>=0.12.34` installed if you wish to follow this tutorial along without any problem😄_
"""
logger.info("# Structured Input for LLMs")

# ! pip install -q llama-index

# ! pip show llama-index | grep "Version"

"""
### 2. Create a Prompt Template

In order to use the structured input, we need to create a prompt template that would have a [Jinja](https://jinja.palletsprojects.com/en/stable/) expression (recognizable by the `{{}}`) with a specific filter (`to_xml`) that will turn inputs such as Pydantic `BaseModel` subclasses, dictionaries or JSON-like strings into XML representations.
"""
logger.info("### 2. Create a Prompt Template")


template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ data | to_xml }}\n```\n\n"
prompt = RichPromptTemplate(template_str)

"""
Let's now try to format the input as a string, using different objects as `data`.
"""
logger.info(
    "Let's now try to format the input as a string, using different objects as `data`.")


class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: Dict[str, str]


user = User(
    name="John",
    surname="Doe",
    age=30,
    email="john.doe@example.com",
    phone="123-456-7890",
    social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
)

display(Markdown(prompt.format(data=user)))

user_dict = {
    "name": "John",
    "surname": "Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "phone": "123-456-7890",
    "social_accounts": {"bluesky": "john.doe", "instagram": "johndoe1234"},
}

display(Markdown(prompt.format(data=user_dict)))

user_str = '{"name":"John","surname":"Doe","age":30,"email":"john.doe@example.com","phone":"123-456-7890","social_accounts":{"bluesky":"john.doe","instagram":"johndoe1234"}}'

display(Markdown(prompt.format(data=user_str)))

"""
### 3. Chat With an LLM

Now that we know how to produce structured input, let's employ it to chat with an LLM!
"""
logger.info("### 3. Chat With an LLM")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass()


llm = OllamaFunctionCallingAdapter(model="llama3.2")

response = llm.chat(prompt.format_messages(data=user))
logger.success(format_json(response))

logger.debug(response.message.content)

"""
### 4. Use Structured Input and Structured Output

Combining structured input and structured output might really help to boost the reliability of the outputs of your LLMs - so let's give it a go!
"""
logger.info("### 4. Use Structured Input and Structured Output")


class SocialAccounts(BaseModel):
    instagram: Optional[str] = Field(default=None)
    bluesky: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    mastodon: Optional[str] = Field(default=None)


class ContactDetails(BaseModel):
    email: str
    phone: str
    social_accounts: SocialAccounts


sllm = llm.as_structured_llm(ContactDetails)

structured_response = sllm.chat(prompt.format_messages(data=user))
logger.success(format_json(structured_response))

logger.debug(structured_response.raw.email)
logger.debug(structured_response.raw.phone)
logger.debug(structured_response.raw.social_accounts.instagram)
logger.debug(structured_response.raw.social_accounts.bluesky)

logger.info("\n\n[DONE]", bright=True)
