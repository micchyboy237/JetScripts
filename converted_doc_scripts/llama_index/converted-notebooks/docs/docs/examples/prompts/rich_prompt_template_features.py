from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.prompts import RichPromptTemplate
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Build with RichPromptTemplate

Introduced in `llama-index-core==0.12.27`, `RichPromptTemplate` is a new prompt template that allows you to build prompts with rich formatting using Jinja syntax.

Using this, you can build:
- basic prompts with variables
- chat prompt templates in a single string
- prompts that accept text, images, and audio
- advanced prompts that loop or parse objects
- and more!

Let's look at some examples.
"""
logger.info("# Build with RichPromptTemplate")

# %pip install llama-index

"""
## Basic Prompt with Variables

In `RichPromptTemplate`, you can use the `{{ }}` syntax to insert variables into your prompt.
"""
logger.info("## Basic Prompt with Variables")


prompt = RichPromptTemplate("Hello, {{ name }}!")

"""
You can format the prompt into either a string or list of chat messages.
"""
logger.info("You can format the prompt into either a string or list of chat messages.")

logger.debug(prompt.format(name="John"))

logger.debug(prompt.format_messages(name="John"))

"""
## Chat Prompt Templates

You can also define chat message blocks directly in the prompt template.
"""
logger.info("## Chat Prompt Templates")

prompt = RichPromptTemplate(
    """
{% chat role="system" %}
You are now chatting with {{ user }}
{% endchat %}

{% chat role="user" %}
{{ user_msg }}
{% endchat %}
"""
)

logger.debug(prompt.format_messages(user="John", user_msg="Hello!"))

"""
## Prompts with Images and Audio

Assuming the LLM you are using supports it, you can also include images and audio in your prompts!

### Images
"""
logger.info("## Prompts with Images and Audio")

# !wget https://cdn.pixabay.com/photo/2016/07/07/16/46/dice-1502706_640.jpg -O image.png


llm = OllamaFunctionCallingAdapter(model="llama3.2", api_key="sk-...")

prompt = RichPromptTemplate(
    """
Describe the following image:
{{ image_path | image}}
"""
)

messages = prompt.format_messages(image_path="./image.png")
response = llm.chat(messages)
logger.debug(response.message.content)

"""
### Audio
"""
logger.info("### Audio")

# !wget AUDIO_URL = "https://science.nasa.gov/wp-content/uploads/2024/04/sounds-of-mars-one-small-step-earth.wav" -O audio.wav

prompt = RichPromptTemplate(
    """
Describe the following audio:
{{ audio_path | audio }}
"""
)
messages = prompt.format_messages(audio_path="./audio.wav")

llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, api_key="sk-...")
response = llm.chat(messages)
logger.debug(response.message.content)

"""
## [Advanced] Loops and Objects

Now, we can take this a step further. Lets assume we have a list of images and text that we want to include in our prompt.

We can use the `{% for x in y %}` loop syntax to loop through the list and include the images and text in our prompt.
"""
logger.info("## [Advanced] Loops and Objects")

text_and_images = [
    ("This is a test", "./image.png"),
    ("This is another test", "./image.png"),
]

prompt = RichPromptTemplate(
    """
{% for text, image_path in text_and_images %}
Here is some text:
{{ text }}
Here is an image:
{{ image_path | image }}
{% endfor %}
"""
)

messages = prompt.format_messages(text_and_images=text_and_images)

"""
Lets inspect the messages to see what we have.
"""
logger.info("Lets inspect the messages to see what we have.")

for message in messages:
    logger.debug(message.role.value)
    for block in message.blocks:
        logger.debug(str(block)[:100])
    logger.debug("\n")

"""
As you can see, we have a single message with a list of blocks, each representing a new block of content (text or image).

(Note: the images are resolved as base64 encoded strings when rendering the prompt)
"""
logger.info("As you can see, we have a single message with a list of blocks, each representing a new block of content (text or image).")

logger.info("\n\n[DONE]", bright=True)